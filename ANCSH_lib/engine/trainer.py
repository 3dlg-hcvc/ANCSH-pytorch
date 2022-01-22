import os
import h5py
import logging
import numpy as np
from time import time

import torch
import torch.optim as optim
from torch.optim import optimizer
from torch.utils.tensorboard import SummaryWriter

from ANCSH_lib.model import ANCSH
from ANCSH_lib.data import ANCSHDataset
from ANCSH_lib import utils
from ANCSH_lib.utils import AvgRecorder, NetworkType
from tools.utils import io
from tools import visualization as viz


class ANCSHTrainer:
    def __init__(self, cfg, data_path, network_type, num_parts):
        self.cfg = cfg
        self.log = logging.getLogger("Network")
        # data_path is a dictionary {'train', 'test'}
        if cfg.device == "cuda:0" and torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        self.device = device
        self.log.info(f"Using device {self.device}")

        self.network_type = NetworkType[network_type] if isinstance(network_type, str) else network_type

        self.num_parts = num_parts
        self.max_epochs = cfg.network.max_epochs
        self.model = self.build_model()
        self.model.to(device)
        self.log.info(f"Below is the network structure:\n {self.model}")

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=cfg.network.lr, betas=(0.9, 0.99)
        )
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.7)

        self.data_path = data_path
        self.writer = None

        self.train_loader = None
        self.test_loader = None
        self.init_data_loader(self.cfg.eval_only)
        self.test_result = None

    def build_model(self):
        model = ANCSH(self.network_type, self.num_parts)
        return model

    def init_data_loader(self, eval_only):
        if not eval_only:
            self.train_loader = torch.utils.data.DataLoader(
                ANCSHDataset(
                    self.data_path["train"], num_points=self.cfg.network.num_points
                ),
                batch_size=self.cfg.network.batch_size,
                shuffle=True,
                num_workers=self.cfg.network.num_workers,
            )

            self.log.info(f'Num {len(self.train_loader)} batches in train loader')

        self.test_loader = torch.utils.data.DataLoader(
            ANCSHDataset(
                self.data_path["test"], num_points=self.cfg.network.num_points
            ),
            batch_size=self.cfg.network.batch_size,
            shuffle=False,
            num_workers=self.cfg.network.num_workers,
        )
        self.log.info(f'Num {len(self.test_loader)} batches in test loader')

    def train_epoch(self, epoch):
        self.log.info(f'>>>>>>>>>>>>>>>> Train Epoch {epoch} >>>>>>>>>>>>>>>>')

        self.model.train()

        iter_time = AvgRecorder()
        io_time = AvgRecorder()
        to_gpu_time = AvgRecorder()
        network_time = AvgRecorder()
        start_time = time()
        end_time = time()
        remain_time = ''

        epoch_loss = {
            'total_loss': AvgRecorder()
        }

        # if self.train_loader.sampler is not None:
        #     self.train_loader.sampler.set_epoch(epoch)
        for i, (camcs_per_point, gt_dict, id) in enumerate(self.train_loader):
            io_time.update(time() - end_time)
            # Move the tensors to the device
            s_time = time()
            camcs_per_point = camcs_per_point.to(self.device)
            gt = {}
            for k, v in gt_dict.items():
                gt[k] = v.to(self.device)
            to_gpu_time.update(time() - s_time)

            # Get the loss
            s_time = time()
            pred = self.model(camcs_per_point)
            loss_dict = self.model.losses(pred, gt)
            network_time.update(time() - s_time)

            loss = torch.tensor(0.0, device=self.device)
            loss_weight = self.cfg.network.loss_weight
            # use different loss weight to calculate the final loss
            for k, v in loss_dict.items():
                if k not in loss_weight:
                    raise ValueError(f"No loss weight for {k}")
                loss += loss_weight[k] * v

            # Used to calculate the avg loss
            for k, v in loss_dict.items():
                if k not in epoch_loss.keys():
                    epoch_loss[k] = AvgRecorder()
                epoch_loss[k].update(v)
            epoch_loss['total_loss'].update(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # time and print
            current_iter = epoch * len(self.train_loader) + i + 1
            max_iter = (self.max_epochs + 1) * len(self.train_loader)
            remain_iter = max_iter - current_iter

            iter_time.update(time() - end_time)
            end_time = time()

            remain_time = remain_iter * iter_time.avg
            remain_time = utils.duration_in_hours(remain_time)

        self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
        # self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], epoch)
        # self.scheduler.step()
        # Add the loss values into the tensorboard
        for k, v in epoch_loss.items():
            if k == "total_loss":
                self.writer.add_scalar(f"{k}", epoch_loss[k].avg, epoch)
            else:
                self.writer.add_scalar(f"loss/{k}", epoch_loss[k].avg, epoch)

        if epoch % self.cfg.train.log_frequency == 0:
            loss_log = ''
            for k, v in epoch_loss.items():
                loss_log += '{}: {:.5f}  '.format(k, v.avg)

            self.log.info(
                'Epoch: {}/{} Loss: {} io_time: {:.2f}({:.4f}) to_gpu_time: {:.2f}({:.4f}) network_time: {:.2f}({:.4f}) \
                duration: {:.2f} remain_time: {}'
                    .format(epoch, self.max_epochs, loss_log, io_time.sum, io_time.avg, to_gpu_time.sum,
                            to_gpu_time.avg, network_time.sum, network_time.avg, time() - start_time, remain_time))

    def eval_epoch(self, epoch, save_results=False):
        self.log.info(f'>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        val_error = {
            'total_loss': AvgRecorder()
        }
        if save_results:
            io.ensure_dir_exists(self.cfg.paths.network.test.output_dir)
            inference_path = os.path.join(self.cfg.paths.network.test.output_dir,
                                          self.network_type.value + '_' + self.cfg.paths.network.test.inference_result)
            self.test_result = h5py.File(inference_path, "w")
            self.test_result.attrs["network_type"] = self.network_type.value

        # test the model on the val set and write the results into tensorboard
        self.model.eval()
        with torch.no_grad():
            start_time = time()
            for i, (camcs_per_point, gt_dict, id) in enumerate(self.test_loader):
                # Move the tensors to the device
                camcs_per_point = camcs_per_point.to(self.device)
                gt = {}
                for k, v in gt_dict.items():
                    gt[k] = v.to(self.device)

                pred = self.model(camcs_per_point)
                if save_results:
                    self.save_results(pred, camcs_per_point, gt, id)
                loss_dict = self.model.losses(pred, gt)
                loss_weight = self.cfg.network.loss_weight
                loss = torch.tensor(0.0, device=self.device)
                # use different loss weight to calculate the final loss
                for k, v in loss_dict.items():
                    if k not in loss_weight:
                        raise ValueError(f"No loss weight for {k}")
                    loss += loss_weight[k] * v

                # Used to calculate the avg loss
                for k, v in loss_dict.items():
                    if k not in val_error.keys():
                        val_error[k] = AvgRecorder()
                    val_error[k].update(v)
                val_error['total_loss'].update(loss)
        # write the val_error into the tensorboard
        if self.writer is not None:
            for k, v in val_error.items():
                self.writer.add_scalar(f"val_error/{k}", val_error[k].avg, epoch)

        loss_log = ''
        for k, v in val_error.items():
            loss_log += '{}: {:.5f}  '.format(k, v.avg)

        self.log.info(
            'Eval Epoch: {}/{} Loss: {} duration: {:.2f}'
                .format(epoch, self.max_epochs, loss_log, time() - start_time))
        if save_results:
            self.test_result.close()
        return val_error

    def train(self, start_epoch=0):
        self.model.train()
        self.writer = SummaryWriter(self.cfg.paths.network.train.output_dir)

        io.ensure_dir_exists(self.cfg.paths.network.train.output_dir)

        best_model = None
        best_result = np.inf
        for epoch in range(start_epoch, self.max_epochs + 1):
            self.train_epoch(epoch)

            if epoch % self.cfg.train.save_frequency == 0 or epoch == self.max_epochs:
                # Save the model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    os.path.join(self.cfg.paths.network.train.output_dir,
                                 self.cfg.paths.network.train.model_filename % epoch),
                )

                val_error = self.eval_epoch(epoch)

                if best_model is None or val_error["total_loss"].avg < best_result:
                    best_model = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    }
                    best_result = val_error["total_loss"].avg
                    torch.save(
                        best_model,
                        os.path.join(self.cfg.paths.network.train.output_dir,
                                     self.cfg.paths.network.train.best_model_filename)
                    )
        self.writer.close()

    def get_latest_model_path(self, with_best=False):
        train_result_dir = os.path.dirname(self.cfg.paths.network.train.output_dir)
        folder, filename = utils.get_latest_file_with_datetime(train_result_dir,
                                                               self.network_type.value + '_', ext='.pth')
        model_path = os.path.join(train_result_dir, folder, filename)
        if with_best:
            model_path = os.path.join(train_result_dir, folder, self.cfg.paths.network.train.best_model_filename)
        return model_path

    def test(self, inference_model=None):
        if not inference_model or not io.file_exist(inference_model):
            inference_model = self.get_latest_model_path(with_best=True)
        if not io.file_exist(inference_model):
            raise IOError(f'Cannot open inference model {inference_model}')
        # Load the model
        self.log.info(f"Load model from {inference_model}")
        checkpoint = torch.load(inference_model, map_location=self.device)
        epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        self.eval_epoch(epoch, save_results=True)

        # create visualizations of evaluation results
        if self.cfg.test.render.render:
            export_dir = os.path.join(self.cfg.paths.network.test.output_dir,
                                      self.cfg.paths.network.test.visualization_folder)
            io.ensure_dir_exists(export_dir)
            inference_path = os.path.join(self.cfg.paths.network.test.output_dir,
                                          self.network_type.value + '_' + self.cfg.paths.network.test.inference_result)
            with h5py.File(inference_path, "r") as inference_h5:
                visualizer = viz.ANCSHVisualizer(inference_h5, network_type=self.network_type)
                visualizer.render(self.cfg.test.render.show, export=export_dir, export_mesh=self.cfg.test.render.export)

    def save_results(self, pred, camcs_per_point, gt, id):
        # Save the results and gt into hdf5 for further optimization
        batch_size = pred["seg_per_point"].shape[0]
        for b in range(batch_size):
            group = self.test_result.create_group(f"{id[b]}")
            group.create_dataset(
                "camcs_per_point",
                data=camcs_per_point[b].detach().cpu().numpy(),
                compression="gzip",
            )

            # save prediction results
            raw_segmentations = pred['seg_per_point'][b].detach().cpu().numpy()
            raw_npcs_points = pred['npcs_per_point'][b].detach().cpu().numpy()
            segmentations, npcs_points = utils.get_prediction_vertices(raw_segmentations, raw_npcs_points)
            group.create_dataset('pred_seg_per_point', data=segmentations, compression="gzip")
            group.create_dataset('pred_npcs_per_point', data=npcs_points, compression="gzip")
            if self.network_type == NetworkType.ANCSH:
                raw_naocs_points = pred['naocs_per_point'][b].detach().cpu().numpy()
                _, naocs_points = utils.get_prediction_vertices(raw_segmentations, raw_naocs_points)
                raw_joint_associations = pred['joint_cls_per_point'][b].detach().cpu().numpy()
                joint_associations = np.argmax(raw_joint_associations, axis=1)
                joint_axes = pred['axis_per_point'][b].detach().cpu().numpy()
                point_heatmaps = pred['heatmap_per_point'][b].detach().cpu().numpy().flatten()
                unit_vectors = pred['unitvec_per_point'][b].detach().cpu().numpy()

                group.create_dataset('pred_naocs_per_point', data=naocs_points, compression="gzip")
                group.create_dataset('pred_joint_cls_per_point', data=joint_associations, compression="gzip")
                group.create_dataset('pred_axis_per_point', data=joint_axes, compression="gzip")
                group.create_dataset('pred_heatmap_per_point', data=point_heatmaps, compression="gzip")
                group.create_dataset('pred_unitvec_per_point', data=unit_vectors, compression="gzip")

            # Save the gt
            for k, v in gt.items():
                group.create_dataset(
                    f"gt_{k}", data=gt[k][b].detach().cpu().numpy(), compression="gzip"
                )

    def resume_train(self, model_path=None):
        if not model_path or not io.file_exist(model_path):
            model_path = self.get_latest_model_path()
        # Load the model
        if io.is_non_zero_file(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            epoch = checkpoint["epoch"]
            self.log.info(f"Continue training with model from {model_path} at epoch {epoch}")
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.model.to(self.device)
        else:
            epoch = 0

        self.train(epoch)
