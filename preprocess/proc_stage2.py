import os
import h5py
import pandas as pd
import logging
import numpy as np
from progress.bar import Bar
from multiprocessing import Pool, cpu_count
from omegaconf import OmegaConf

from tools.utils import io
from ANCSH_lib.utils import NetworkType
from tools.visualization import Viewer, ANCSHVisualizer

import utils
from utils import JointType

log = logging.getLogger('proc_stage2')


class ProcStage2Impl:
    def __init__(self, cfg):
        self.output_path = cfg.output_path
        self.input_h5_path = cfg.input_h5_path
        self.stage1_tmp_dir = cfg.stage1_tmp_dir
        self.tmp_output_dir = cfg.tmp_output_dir
        self.rest_state_data_filename = cfg.rest_state_data_filename
        self.object_infos_path = cfg.object_infos_path
        self.heatmap_threshold = cfg.heatmap_threshold
        self.epsilon = 10e-8
        self.export = cfg.export

    @staticmethod
    def get_joint_params(vertices, joint, selected_vertices):
        heatmap = -np.ones((vertices.shape[0]))
        unitvec = np.zeros((vertices.shape[0], 3))
        joint_pos = joint['abs_position']
        joint_axis = joint['axis']
        joint_axis = joint_axis / np.linalg.norm(joint_axis)
        joint_axis = joint_axis.reshape((3, 1))
        if joint['type'] == JointType.revolute.value:
            vec1 = vertices - joint_pos
            # project to joint axis
            proj_len = np.dot(vec1, joint_axis)
            # np.clip(proj_len, a_min=self.epsilon, a_max=None, out=proj_len)
            proj_vec = proj_len * joint_axis.transpose()
            orthogonal_vec = - vec1 + proj_vec
            tmp_heatmap = np.linalg.norm(orthogonal_vec, axis=1).reshape(-1, 1)
            tmp_unitvec = orthogonal_vec / tmp_heatmap
            heatmap[selected_vertices] = tmp_heatmap[selected_vertices].reshape(-1)
            unitvec[selected_vertices] = tmp_unitvec[selected_vertices]
        elif joint['type'] == JointType.prismatic.value:
            heatmap[selected_vertices] = 0
            unitvec[selected_vertices] = joint_axis.transpose()
        else:
            log.error(f'Invalid joint type {joint["axis"]}')

        heatmap = np.where(heatmap >= 0, heatmap, np.inf)
        return heatmap, unitvec

    def __call__(self, idx, input_data):
        input_h5 = h5py.File(self.input_h5_path, 'r')
        object_infos = io.read_json(self.object_infos_path)
        output_filepath = os.path.splitext(self.output_path)[0] + f'_{idx}' + os.path.splitext(self.output_path)[-1]
        h5file = h5py.File(output_filepath, 'w')
        bar = Bar(f'Stage2 Processing chunk {idx}', max=len(input_data))
        for index, row in input_data.iterrows():
            instance_name = f'{row["objectCat"]}_{row["objectId"]}_{row["articulationId"]}_{row["frameId"]}'
            in_h5frame = input_h5[instance_name]
            mask = in_h5frame['mask'][:]
            points_camera = in_h5frame['points_camera'][:]
            points_rest_state = in_h5frame['points_rest_state'][:]
            parts_camera2rest_state = in_h5frame['parts_transformation'][:]
            camera2base = in_h5frame['base_transformation'][:]

            stage1_tmp_data_dir = os.path.join(self.stage1_tmp_dir, row['objectCat'], row['objectId'])
            rest_state_data_path = os.path.join(stage1_tmp_data_dir, self.rest_state_data_filename)
            rest_state_data = io.read_json(rest_state_data_path)

            part_info = object_infos[row['objectCat']][row['objectId']]['part']
            num_parts = len(part_info)

            # process points related ground truth
            object_info = object_infos[row['objectCat']][row['objectId']]['object']
            # diagonal axis aligned bounding box length to 1
            # (0.5, 0.5, 0.5) centered
            naocs_translation = - np.asarray(object_info['center']) + 0.5 * object_info['scale']
            naocs_scale = 1.0 / object_info['scale']
            naocs = points_rest_state + naocs_translation
            naocs *= naocs_scale

            naocs_transformation = np.reshape(camera2base, (4, 4), order='F')
            naocs_transformation[:3, 3] += naocs_translation
            naocs2cam_transformation = np.linalg.inv(naocs_transformation).flatten('F')
            naocs2cam_scale = 1.0 / naocs_scale

            points_class = np.empty_like(mask)
            npcs = np.empty_like(points_rest_state)
            parts_npcs2cam_transformation = np.empty_like(parts_camera2rest_state)
            parts_npcs2cam_scale = np.empty(num_parts)
            parts_min_bounds = np.empty((num_parts, 3))
            parts_max_bounds = np.empty((num_parts, 3))
            for link_index, link in enumerate(rest_state_data['links']):
                if link['virtual']:
                    continue
                link_index_key = str(link_index)
                part_points = points_rest_state[mask == link_index]
                center = np.asarray(part_info[link_index_key]['center'])
                # diagonal axis aligned bounding box length to 1
                # (0.5, 0.5, 0.5) centered
                npcs_translation = - center + 0.5 * part_info[link_index_key]['scale']
                npcs_scale = 1.0 / part_info[link_index_key]['scale']
                part_points_norm = part_points + npcs_translation
                part_points_norm *= npcs_scale

                npcs[mask == link_index] = part_points_norm
                part_class = part_info[link_index_key]['part_class']
                points_class[mask == link_index] = part_class
                npcs_transformation = np.reshape(parts_camera2rest_state[link['part_index']], (4, 4), order='F')
                npcs_transformation[:3, 3] += npcs_translation
                npcs2cam_transformation = np.linalg.inv(npcs_transformation)
                parts_npcs2cam_transformation[part_class] = npcs2cam_transformation.flatten('F')
                parts_npcs2cam_scale[part_class] = 1.0 / npcs_scale
                parts_min_bounds[part_class] = np.asarray(part_info[link_index_key]['min_bound'])
                parts_max_bounds[part_class] = np.asarray(part_info[link_index_key]['max_bound'])

            # process joints related ground truth
            link_names = [link['name'] for link in rest_state_data['links']]
            # transform joints to naocs space
            viewer = Viewer()
            naocs_joints = rest_state_data['joints']
            for i, joint in enumerate(rest_state_data['joints']):
                if not joint:
                    continue
                joint_pose = np.asarray(joint['pose2link']).reshape((4, 4), order='F')
                joint_parent = joint['parent']
                parent_link = rest_state_data['links'][link_names.index(joint_parent)]
                parent_link_abs_pose = np.asarray(parent_link['abs_pose']).reshape((4, 4), order='F')
                joint_abs_pose = np.dot(parent_link_abs_pose, joint_pose)
                joint_pos = joint_abs_pose[:3, 3]
                naocs_joint_pos = joint_pos + naocs_translation
                naocs_joint_pos *= naocs_scale
                joint_axis = np.dot(joint_abs_pose[:3, :3], joint['axis'])
                joint_axis = joint_axis / np.linalg.norm(joint_axis)

                naocs_joints[i]['abs_position'] = naocs_joint_pos
                naocs_joints[i]['axis'] = joint_axis

                joint_child = joint['child']
                child_link_class = part_info[str(link_names.index(joint_child))]['part_class']
                joint_class = child_link_class
                naocs_joints[i]['class'] = joint_class

                joint_type = JointType[joint['type']].value
                naocs_joints[i]['type'] = joint_type

                if self.export:
                    viewer.add_trimesh_arrows([naocs_joint_pos], [joint_axis], color=Viewer.rgba_by_index(joint_class))

            if self.export:
                tmp_data_dir = os.path.join(self.tmp_output_dir, row['objectCat'], row['objectId'],
                                            row['articulationId'])
                io.ensure_dir_exists(tmp_data_dir)
                viewer.export(os.path.join(tmp_data_dir, instance_name+'_naocs_arrows.ply'))

            valid_joints = [joint for joint in naocs_joints if joint if joint['type'] >= 0]
            num_valid_joints = len(valid_joints)
            tmp_heatmap = np.empty((num_valid_joints, naocs.shape[0]))
            tmp_unitvec = np.empty((num_valid_joints, naocs.shape[0], 3))
            for i, joint in enumerate(valid_joints):
                joint_class = joint['class']
                parent_links = [i for i, link in enumerate(rest_state_data['links'])
                                if link if not link['virtual'] if joint['parent'] == link['name']]
                child_links = [i for i, link in enumerate(rest_state_data['links'])
                               if link if not link['virtual'] if joint['child'] == link['name']]
                connected_links = parent_links + child_links
                part_classes = [part_info[str(link_index)]['part_class'] for link_index in connected_links]
                if joint['type'] == JointType.prismatic.value:
                    part_classes = [part_class for part_class in part_classes if part_class == joint_class]
                selected_vertex_indices = np.isin(points_class, part_classes)
                part_heatmap, part_unitvec = ProcStage2Impl.get_joint_params(naocs, joint, selected_vertex_indices)

                tmp_heatmap[joint_class - 1] = part_heatmap
                tmp_unitvec[joint_class - 1] = part_unitvec

            joints_association = tmp_heatmap.argmin(axis=0)
            points_heatmap = tmp_heatmap[joints_association, np.arange(naocs.shape[0])]
            points_unitvec = tmp_unitvec[joints_association, np.arange(naocs.shape[0])]
            points_unitvec[points_heatmap >= self.heatmap_threshold] = np.zeros(3)
            joints_association[points_heatmap >= self.heatmap_threshold] = -1
            points_heatmap_result = 1.0 - points_heatmap / self.heatmap_threshold
            points_heatmap_result[points_heatmap >= self.heatmap_threshold] = -1
            # points with no joint association has value 0
            joints_association += 1
            joints_axis = np.zeros((naocs.shape[0], 3))
            joint_types = -np.ones(num_parts)
            for joint in naocs_joints:
                if joint:
                    joints_axis[joints_association == joint['class']] = joint['axis']
                    joint_types[joint['class']] = joint['type']

            h5frame = h5file.require_group(instance_name)
            h5frame.attrs['objectCat'] = row["objectCat"]
            h5frame.attrs['objectId'] = row["objectId"]
            h5frame.attrs['articulationId'] = row["articulationId"]
            h5frame.attrs['frameId'] = row["frameId"]
            h5frame.attrs['numParts'] = num_parts
            h5frame.attrs['id'] = instance_name
            h5frame.create_dataset("seg_per_point", shape=points_class.shape, data=points_class, compression="gzip")
            h5frame.create_dataset("camcs_per_point", shape=points_camera.shape, data=points_camera, compression="gzip")
            h5frame.create_dataset("npcs_per_point", shape=npcs.shape, data=npcs, compression="gzip")
            h5frame.create_dataset("naocs_per_point", shape=naocs.shape, data=naocs, compression="gzip")
            h5frame.create_dataset("heatmap_per_point", shape=points_heatmap_result.shape, data=points_heatmap_result,
                                   compression="gzip")
            h5frame.create_dataset("unitvec_per_point", shape=points_unitvec.shape, data=points_unitvec,
                                   compression="gzip")
            h5frame.create_dataset("axis_per_point", shape=joints_axis.shape, data=joints_axis,
                                   compression="gzip")
            h5frame.create_dataset("joint_cls_per_point", shape=joints_association.shape, data=joints_association,
                                   compression="gzip")
            h5frame.create_dataset("joint_type", shape=joint_types.shape, data=joint_types, compression="gzip")
            # 6D transformation from npcs to camcs
            h5frame.create_dataset("npcs2cam_rt", shape=parts_npcs2cam_transformation.shape,
                                   data=parts_npcs2cam_transformation, compression="gzip")
            # scale from npcs to camcs
            h5frame.create_dataset("npcs2cam_scale", shape=parts_npcs2cam_scale.shape, data=parts_npcs2cam_scale,
                                   compression="gzip")
            h5frame.create_dataset("naocs2cam_rt", shape=naocs2cam_transformation.shape,
                                   data=naocs2cam_transformation, compression="gzip")
            h5frame.create_dataset("naocs2cam_scale", shape=(1,), data=naocs2cam_scale,
                                   compression="gzip")
            norm_factors = 1.0 / parts_npcs2cam_scale
            h5frame.create_dataset("norm_factors", shape=norm_factors.shape, data=norm_factors,
                                   compression="gzip")
            # part bounds at rest state
            norm_corners = np.stack((parts_min_bounds, parts_max_bounds), axis=1)
            h5frame.create_dataset("norm_corners", shape=norm_corners.shape, data=norm_corners,
                                   compression="gzip")
            bar.next()
        bar.finish()
        h5file.close()
        input_h5.close()
        return output_filepath


class ProcStage2:
    def __init__(self, cfg):
        self.cfg = cfg
        self.input_cfg = self.cfg.paths.preprocess.stage2.input
        self.input_h5_path = os.path.join(self.cfg.paths.preprocess.output_dir, self.input_cfg.pcd_data)
        self.output_dir = self.cfg.paths.preprocess.output_dir
        self.stag1_tmp_output = self.cfg.paths.preprocess.stage1.tmp_output
        self.tmp_output = self.cfg.paths.preprocess.stage2.tmp_output
        self.split_info = None
        self.debug = self.cfg.debug
        self.show = self.cfg.show
        self.export = self.cfg.export
        stage1_input = self.cfg.paths.preprocess.stage1.input
        self.part_orders = io.read_json(os.path.join(self.cfg.paths.preprocess.input_dir, stage1_input.part_order_file))
        self.heatmap_threshold = self.cfg.params.joint_association_threshold

    def split_data(self, train_percent=.6, split_on='objectId', seed=None):
        instances = []
        visit_groups = lambda name, node: instances.append(name) if isinstance(node, h5py.Group) else None
        input_h5 = h5py.File(self.input_h5_path, 'r')
        input_h5.visititems(visit_groups)
        df_dataset = pd.DataFrame([name.split('_') for name in instances],
                                  columns=['objectCat', 'objectId', 'articulationId', 'frameId'])
        df_dataset = df_dataset.drop_duplicates(ignore_index=True)
        # select data in config
        selected_categories = df_dataset['objectCat'].isin(self.cfg.settings.categories) \
            if len(self.cfg.settings.categories) > 0 else df_dataset['objectCat'].astype(bool)
        selected_object_ids = df_dataset['objectId'].isin(self.cfg.settings.object_ids) \
            if len(self.cfg.settings.object_ids) > 0 else df_dataset['objectId'].astype(bool)
        selected_articulation_ids = df_dataset['articulationId'].isin(self.cfg.settings.articulation_ids) \
            if len(self.cfg.settings.articulation_ids) > 0 else df_dataset['articulationId'].astype(bool)
        df_dataset = df_dataset[selected_categories & selected_object_ids & selected_articulation_ids]

        if io.file_exist(self.cfg.paths.preprocess.stage2.input.split_info, ext='.csv'):
            input_split_info = pd.read_csv(self.cfg.paths.preprocess.stage2.input.split_info)
            self.split_info = input_split_info.merge(df_dataset, how='inner',
                                                     on=['objectCat', 'objectId', 'articulationId', 'frameId'])
        else:
            # split to train, val, test
            log.info(f'Split on key {split_on}')
            if len(df_dataset):
                if split_on == 'objectId':
                    split_on_columns = ['objectCat', 'objectId']
                elif split_on == 'articulationId':
                    split_on_columns = ['objectCat', 'objectId', 'articulationId']
                elif split_on == 'frameId':
                    split_on_columns = ['objectCat', 'objectId', 'articulationId', 'frameId']
                else:
                    split_on_columns = ['objectCat', 'objectId']
                    log.warning(f'Cannot parse split_on {split_on}, split on objectId by default')

                val_end = train_percent + (1.0 - train_percent) / 2.0
                split_df = df_dataset[split_on_columns].drop_duplicates()
                set_size = len(split_df)
                train_set, val_set, test_set = np.split(
                    split_df.sample(frac=1.0, random_state=seed),
                    [int(train_percent * set_size), int(val_end * set_size)]
                )
                train = train_set.merge(df_dataset, how='left', on=split_on_columns)
                val = val_set.merge(df_dataset, how='left', on=split_on_columns)
                test = test_set.merge(df_dataset, how='left', on=split_on_columns)

                self.split_info = pd.concat([train, val, test], keys=["train", "val", "test"], names=['set', 'index'])
            else:
                log.error('No data to split!')
                return
        self.split_info.to_csv(os.path.join(self.output_dir, self.cfg.paths.preprocess.stage2.output.split_info))

    def process(self):
        io.ensure_dir_exists(self.output_dir)
        if self.split_info is None or self.split_info.empty:
            log.error('No data to process!')
            return
        train = self.split_info.loc['train']
        log.info(f'Stage2 Process Train Set {len(train)} instances')
        self.process_set(train, self.output_dir, self.cfg.paths.preprocess.stage2.output.train_data)
        val = self.split_info.loc['val']
        log.info(f'Stage2 Process Val Set {len(val)} instances')
        self.process_set(val, self.output_dir, self.cfg.paths.preprocess.stage2.output.val_data)
        test = self.split_info.loc['test']
        log.info(f'Stage2 Process Test Set {len(test)} instances')
        self.process_set(test, self.output_dir, self.cfg.paths.preprocess.stage2.output.test_data)

    def process_set(self, input_data, output_dir, output_filename):
        # process object info
        object_df = input_data[['objectCat', 'objectId']].drop_duplicates()
        object_infos = {}
        bar = Bar('Stage2 Parse Object Infos', max=len(object_df))
        for index, row in object_df.iterrows():
            stage1_tmp_data_dir = os.path.join(self.cfg.paths.preprocess.tmp_dir, row['objectCat'], row['objectId'],
                                               self.stag1_tmp_output.folder_name)
            rest_state_data_path = os.path.join(stage1_tmp_data_dir, self.stag1_tmp_output.rest_state_data)
            rest_state_data = io.read_json(rest_state_data_path)
            object_mesh_path = os.path.join(stage1_tmp_data_dir, self.stag1_tmp_output.rest_state_mesh)
            object_dict = utils.get_mesh_info(object_mesh_path)
            part_dict = {}
            part_order = None
            if self.part_orders:
                part_order = self.part_orders[row['objectCat']][row['objectId']]
            part_index = 0
            for link_index, link in enumerate(rest_state_data['links']):
                if link['virtual']:
                    continue
                part_mesh_path = os.path.join(stage1_tmp_data_dir,
                                              f'{link["name"]}_{self.stag1_tmp_output.rest_state_mesh}')
                part_dict[link_index] = utils.get_mesh_info(part_mesh_path)
                if part_order:
                    part_dict[link_index]['part_class'] = part_order.index(link['part_index'])
                else:
                    part_dict[link_index]['part_class'] = part_index
                    part_index += 1
            if row['objectCat'] in object_infos:
                object_infos[row['objectCat']][row['objectId']] = {'object': object_dict, 'part': part_dict}
            else:
                object_infos[row['objectCat']] = {row['objectId']: {'object': object_dict, 'part': part_dict}}
            bar.next()
        bar.finish()
        tmp_data_dir = os.path.join(self.cfg.paths.preprocess.tmp_dir, self.tmp_output.folder_name)
        io.ensure_dir_exists(tmp_data_dir)
        object_infos_path = os.path.join(tmp_data_dir, self.tmp_output.object_info)
        io.write_json(object_infos, object_infos_path)

        num_processes = min(cpu_count(), self.cfg.num_workers)
        # calculate the chunk size
        chunk_size = max(1, int(input_data.shape[0] / num_processes))
        chunks = [input_data.iloc[input_data.index[i:i + chunk_size]] for i in
                  range(0, input_data.shape[0], chunk_size)]
        log.info(f'Stage2 Processing Start with {num_processes} workers and {len(chunks)} chunks')

        config = OmegaConf.create()
        config.input_h5_path = self.input_h5_path
        config.stage1_tmp_dir = os.path.join(self.cfg.paths.preprocess.tmp_dir, self.stag1_tmp_output.folder_name)
        config.tmp_output_dir = os.path.join(self.cfg.paths.preprocess.tmp_dir, self.tmp_output.folder_name)
        config.output_path = os.path.join(config.tmp_output_dir, output_filename)
        config.rest_state_data_filename = self.stag1_tmp_output.rest_state_data
        config.object_infos_path = object_infos_path
        config.heatmap_threshold = self.heatmap_threshold
        config.export = self.cfg.export

        with Pool(processes=num_processes) as pool:
            proc_impl = ProcStage2Impl(config)
            output_filepath_list = pool.starmap(proc_impl, enumerate(chunks))

        h5_output_path = os.path.join(output_dir, output_filename)
        h5file = h5py.File(h5_output_path, 'w')
        for filepath in output_filepath_list:
            with h5py.File(filepath, 'r') as h5f:
                for key in h5f.keys():
                    h5f.copy(key, h5file)
        h5file.close()

        if self.debug:
            tmp_data_dir = os.path.join(self.cfg.paths.preprocess.tmp_dir, self.tmp_output.folder_name)
            io.ensure_dir_exists(tmp_data_dir)

            with h5py.File(h5_output_path, 'r') as h5file:
                visualizer = ANCSHVisualizer(h5file, NetworkType.ANCSH, gt=True, sampling=20)
                visualizer.point_size = 5
                visualizer.arrow_sampling = 10
                visualizer.prefix = ''
                visualizer.render(show=self.show, export=tmp_data_dir, export_mesh=self.export)
