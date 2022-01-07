import os
import logging
import trimesh
import pyrender

import numpy as np
from PIL import Image

from tools.utils import io
from matplotlib import cm

log = logging.getLogger('viewer')

from pyrender.constants import (DEFAULT_SCENE_SCALE,
                                DEFAULT_Z_FAR, DEFAULT_Z_NEAR)

os.environ['PYOPENGL_PLATFORM'] = 'egl'


class Viewer:
    def __init__(self, vertices=None, faces=None, colors=None, mask=None):
        self.vertices = vertices
        self.faces = faces
        self.colors = colors
        self.mask = mask
        self.trimesh = None
        self.point_cloud = None
        self.trimesh_list = []
        self.point_cloud_list = []
        self.scene = pyrender.Scene()
        self.scene.ambient_light = [1.0, 1.0, 1.0]
        if vertices is not None:
            self.add_geometry(vertices, faces, colors, mask)

    head_body_ratio = 1.0 / 4

    @staticmethod
    def rgba_by_index(index, cmap_name='Set1'):
        return list(cm.get_cmap(cmap_name)(index))

    @staticmethod
    def colors_from_mask(mask, empty_first=False):
        unique_val = np.sort(np.unique(mask))
        colors = np.empty([mask.shape[0], 4])
        for i, val in enumerate(unique_val):
            if empty_first and i == 0:
                rgba = [0.5, 0.5, 0.5, 0.5]
            else:
                rgba = Viewer.rgba_by_index(i)
            colors[mask == val] = rgba
        return colors

    def load(self, mesh_path):
        self.trimesh = trimesh.load(mesh_path, force='mesh')
        assert isinstance(self.trimesh, trimesh.base.Trimesh)

    def add_geometry(self, vertices, faces=None, colors=None, mask=None):
        if colors is None and mask is not None:
            colors = Viewer.colors_from_mask(mask)
        if faces is not None:
            geo = trimesh.base.Trimesh(vertices, faces=faces, vertex_colors=colors)
            self.add_trimesh(geo)
        else:
            geo = trimesh.points.PointCloud(vertices, vertex_colors=colors)
            self.add_point_cloud(geo)

    def add_trimesh(self, mesh):
        self.trimesh_list.append(mesh)

    def add_point_cloud(self, point_cloud):
        self.point_cloud_list.append(point_cloud)

    def merge_point_clouds(self):
        all_vertices = None
        all_colors = None
        for point_cloud in self.point_cloud_list:
            vertices = point_cloud.vertices
            colors = point_cloud.colors
            if colors.shape[0] == 1:
                colors = np.tile(colors, (vertices.shape[0], 1))
            all_vertices = vertices if all_vertices is None else np.vstack((all_vertices, vertices))
            all_colors = colors if all_colors is None else np.vstack((all_colors, colors))
        self.point_cloud = trimesh.points.PointCloud(vertices, colors=colors)

    # def transform_parts(self, part_mask, transformations, scales, mesh_index=0):
    #     if self.trimesh_list[mesh_index].vertices.shape[0] != part_mask.shape[0]:
    #         log.error('Number of points in mask does not match the points in mesh')
    #         return
    #     part_classes = np.sort(np.unique(part_mask))
    #     transformed_points = np.empty_like(self.mesh.vertices)
    #     for part_class in part_classes:
    #         part_points = self.mesh.vertices[part_mask == part_class]
    #         scale = scales[part_class]
    #         part_points_scaled = part_points * scale
    #         part_transformation = transformations[part_class].reshape((4, 4), order='F')
    #         part_points_scaled_p4 = np.column_stack((part_points_scaled, np.ones(part_points_scaled.shape[0])))
    #         part_points_scaled_p4 = part_points_scaled_p4.transpose()
    #         transformed_part = np.dot(part_transformation, part_points_scaled_p4)
    #         transformed_part = transformed_part.transpose()[:, :3]
    #         transformed_points[part_mask == part_class] = transformed_part
    #
    #     self.trimesh_list[mesh_index] = trimesh.base.Trimesh(transformed_points, colors=self.mesh.visual.vertex_colors)

    @staticmethod
    def draw_arrow(color=None, radius=0.01, length=0.5):
        if color is None:
            color = Viewer.rgba_by_index(0)
        head_length = length * Viewer.head_body_ratio
        body_length = length - head_length
        head_transformation = np.eye(4)
        head_transformation[:3, 3] += [0, 0, body_length / 2.0]
        head = trimesh.creation.cone(3 * radius, head_length, sections=10, transform=head_transformation)
        body = trimesh.creation.cylinder(radius, body_length, sections=10)
        arrow = head + body
        arrow.visual.vertex_colors = color
        return arrow

    def add_arrows(self, positions, axes, color=None, radius=0.01, length=0.5):
        transformations = []
        z_axis = [0, 0, 1]
        for i, pos in enumerate(positions):
            transformation = trimesh.geometry.align_vectors(z_axis, axes[i])
            transformation[:3, 3] += pos + axes[i] * (1 - Viewer.head_body_ratio) / 2 * length
            transformations.append(transformation)
        arrow = Viewer.draw_arrow(color, radius, length)
        arrows = pyrender.Mesh.from_trimesh(arrow, poses=transformations)
        self.scene.add(arrows)

    def add_trimesh_arrows(self, positions, axes, color=None, radius=0.01, length=0.5):
        arrows = []
        z_axis = [0, 0, 1]
        for i, pos in enumerate(positions):
            transformation = trimesh.geometry.align_vectors(z_axis, axes[i])
            transformation[:3, 3] += pos + axes[i] * (1 - Viewer.head_body_ratio) / 2 * length[i]
            if isinstance(length, float):
                arrow = Viewer.draw_arrow(color, radius, length)
            else:
                arrow = Viewer.draw_arrow(color, radius, length[i])
            arrow.apply_transform(transformation)
            arrows.append(arrow)
        self.trimesh_list += arrows

    def _merge_geometries(self):
        if len(self.trimesh_list) > 0:
            self.trimesh = trimesh.util.concatenate(self.trimesh_list)
        self.merge_point_clouds()

    def _add_geometries_to_scenen(self):
        if self.trimesh is None or self.point_cloud is None:
            self._merge_geometries()
        if self.trimesh is not None:
            mesh = pyrender.Mesh.from_trimesh(self.trimesh)
            self.scene.add(mesh)
        if self.point_cloud is not None:
            point_cloud = pyrender.Mesh.from_points(self.point_cloud.vertices, colors=self.point_cloud.colors)
            self.scene.add(point_cloud)

    def show(self, window_size=None, window_name='Default Viewer'):
        self._add_geometries_to_scenen()
        if window_size is None:
            window_size = [800, 600]
        pyrender.Viewer(self.scene, viewport_size=window_size, window_title=window_name, point_size=8.0)

    def _compute_initial_camera_pose(self):
        centroid = self.scene.centroid
        scale = self.scene.scale
        if scale == 0.0:
            scale = DEFAULT_SCENE_SCALE

        # s2 = 1.0 / np.sqrt(2.0)
        cp = np.eye(4)
        # cp[:3, :3] = np.array([
        #     [0.0, -s2, s2],
        #     [1.0, 0.0, 0.0],
        #     [0.0, s2, s2]
        # ])
        h_fov = np.pi / 6.0
        dist = scale / (1.5 * np.tan(h_fov))
        cp[:3, 3] = dist * np.array([0.0, 0.0, 1.0]) + centroid

        return cp

    def render(self, fig_path, fig_size=None):
        self._add_geometries_to_scenen()

        if fig_size is None:
            fig_size = [1024, 768]
        renderer = pyrender.OffscreenRenderer(viewport_width=fig_size[0], viewport_height=fig_size[1], point_size=10.0)
        z_far = max(self.scene.scale * 10.0, DEFAULT_Z_FAR)
        if self.scene.scale == 0:
            z_near = DEFAULT_Z_NEAR
        else:
            z_near = min(self.scene.scale / 10.0, DEFAULT_Z_NEAR)
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, znear=z_near, zfar=z_far)
        cam_pose = self._compute_initial_camera_pose()
        cam_node = pyrender.Node(camera=cam, matrix=cam_pose)
        self.scene.add_node(cam_node)
        color, depth = renderer.render(self.scene)
        image = Image.fromarray(color.astype('uint8'), 'RGB')
        image.save(fig_path)

        renderer.delete()
        del renderer
        return color, depth

    def export(self, mesh_path):
        if self.trimesh is None or self.point_cloud is None:
            self._merge_geometries()
        if self.point_cloud is not None:
            mesh = trimesh.base.Trimesh(self.point_cloud.vertices, vertex_colors=self.point_cloud.colors)
        if self.trimesh is not None and self.point_cloud is not None:
            mesh = trimesh.util.concatenate(self.trimesh, mesh)
        mesh.export(mesh_path)
