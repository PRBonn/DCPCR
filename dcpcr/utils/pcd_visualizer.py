import moviepy.video.io.ImageSequenceClip
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from dcpcr.models import loss
from dcpcr.utils import utils
from dcpcr.utils import fine_tuner
from matplotlib import cm
from torch.functional import Tensor
# import pytimer
from torch.utils.data import dataloader
import os
BACKGROUND = np.array([0, 0, 0])

def remove_ground(pcd, distance_threshold=0.3, num_iterations=2000, angle_threshold = 0.9):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,  # adjust maybe this threshold?
                                             ransac_n=3,
                                             num_iterations=num_iterations)

    ### Check if point normals fit to plane normal
    n = np.array([[plane_model[0], plane_model[1], plane_model[2]]])
    normals = np.asarray(pcd.normals)[inliers, :]
    # Check cosine distance between normals
    cos_dist = np.squeeze(n @ normals.T)
    inl = np.squeeze(np.argwhere(np.abs(cos_dist) > angle_threshold)
                     )  # adjust maybe this threshold?
    inliers = [inliers[i] for i in inl]
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return inlier_cloud, outlier_cloud


def torch2o3d(pcd, colors=None, estimate_normals=False):
    # print(pcd.shape)
    pcd = pcd.detach().cpu().squeeze().numpy() if isinstance(pcd, torch.Tensor) else pcd
    colors = colors.detach().cpu().squeeze().numpy() if isinstance(
        colors, torch.Tensor) else colors
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pcd)
    if estimate_normals:
        pcl.estimate_normals()
    if colors is not None:
        pcl.colors = o3d.utility.Vector3dVector(colors)
    return pcl


def visualize(pcd, estimate_normals=True, colors=None):
    pcd = pcd.detach().cpu().numpy() if isinstance(pcd, torch.Tensor) else pcd
    colors = colors.detach().cpu().numpy() if isinstance(
        colors, torch.Tensor) else colors

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pcd)
    if estimate_normals:
        pcl.estimate_normals()
    if colors is not None:
        pcl.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcl])


def corr2Lines(pt, ps_t, W: torch.Tensor):
    corr = [(i, torch.argmax(j).item()) for i, j in enumerate(W[0, :, :])]
    lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        pt, ps_t, corr)

    # colormap = cm.get_cmap('spring',)
    v, _ = (W[0, :, :]).detach().cpu().max(dim=-1)
    plt.hist(v.flatten().detach().cpu().numpy(), bins=1000)
    plt.show()
    # c = (1-v.unsqueeze(-1)).numpy()*colormap(v.numpy())

    # print(c)
    # print(v)
    # print(W.max())
    # lines.colors = o3d.utility.Vector3dVector(c[:,:3])
    # print(c.shape)
    # print(len(corr))
    # lines.colors =
    return lines


def transform(points, T):
    shape = list(points.shape)
    shape[-1] = 1
    ps_t = torch.cat([points[..., :, :3], torch.ones(shape)], -1)
    ps_t = (T@ps_t.transpose(-1, -2)).transpose(-1, -2)
    ps_t = ps_t[0, :, :3]
    return ps_t


class InputDataVis():
    def __init__(self, dataloader, draw_src=False):
        self.dataloader = dataloader
        self.draw_src = draw_src

    def getGeometries(self, i):
        batch = self.dataloader.next()
        # batch = self.dataloader.next()
        ps_t = transform(batch['source'], batch['pose'])
        mask = batch['mask_source'][0, :, 0]
        ps_t = ps_t[mask, :]
        ps_t = torch2o3d(ps_t)
        ps_t.paint_uniform_color(np.array([0, 0, 1]))
        print('Transformed Source: Blue')

        ##### Target ######
        mask = batch['mask_target'][0, :, 0]
        pt = batch['target'][0, mask, :3]
        pt = torch2o3d(pt)
        pt.paint_uniform_color(np.array([0, 1, 0]))
        print('Target: Green')
        geoms = [pt, ps_t]

        ##### Source ######
        if self.draw_src:
            ps = batch['source'][0, mask, :3]
            print('Source: Red')
            ps = torch2o3d(ps)
            ps.paint_uniform_color(np.array([1, 0, 0]))
            geoms.append(ps)
        print()
        return geoms


class RegVis():
    def __init__(self, dataloader, model, draw_corr_lines=False, draw_corr_points=False, show_gt=False, replace_pcd=None, ft=True):
        self.model = model
        self.dataloader = dataloader
        self.draw_corr_lines = draw_corr_lines
        self.draw_corr_points = draw_corr_points
        self.show_gt = show_gt
        self.replace_pcd = replace_pcd
        self.ft = ft

    def getGeometries(self, i):
        print(f"batch {i}")
        batch = next(self.dataloader)
        # batch = self.dataloader.next()
        target = batch['target']
        source = batch['source']
        T, W, ps_corr, ww = self.model.model(
            target, source, batch['mask_target'], batch['mask_source'])
        if self.ft:
            source = utils.torch2o3d(
                batch['source'][0, batch['mask_source'][0, :, 0]])
            target = utils.torch2o3d(
                batch['target'][0, batch['mask_target'][0, :, 0]])
            init_guess = T.detach().cpu().squeeze().numpy()
            pose = fine_tuner.refine_registration(source,
                                             target,
                                             init_guess,
                                             distance_threshold=1/40)
            T = torch.tensor(
                pose, device=T.device, dtype=T.dtype)
        # T = batch['pose']

        dt,dr = loss.pose_error(T, batch['pose'],scale=40)
        print(f'dt {dt.item()}, dr {dr.item()}')
        ##### Transformed Source ######
        mask = batch['mask_source'][0, :, 0]
        if self.replace_pcd is not None:
            file = batch['file_source'][0]
            for r1 in self.replace_pcd:
                file = file.replace(r1, self.replace_pcd[r1])
            ps_t = o3d.io.read_point_cloud(file)
            T2 = T.squeeze().detach().cpu().numpy()
            T2[:3, -1] *= 40
            ps_t.transform(T2)
        else:
            ps_t = transform(source, T)
            ps_t = ps_t[mask, :]
            ps_t = torch2o3d(ps_t)
        ps_t.paint_uniform_color(np.array([0, 0, 1]))
        print('Transformed Source: Blue')

        ##### original Source ######
        if self.replace_pcd is not None:
            ps = o3d.io.read_point_cloud(file)
            if self.show_gt:
                T2 = batch['pose'].squeeze().detach().cpu().numpy()
                T2[:3, -1] *= 40
                ps_t.transform(T2)
        else:
            ps = source[0, mask, :3]
            if self.show_gt:
                ps = transform(ps, batch['pose'])
            ps = torch2o3d(ps)
        print('GT Transformed Source: Red' if self.show_gt else 'Source: Red')
        ps.paint_uniform_color(np.array([1, 0, 0]))

        ##### Target Source ######
        mask = batch['mask_target'][0, :, 0]
        if self.replace_pcd is not None:
            file = batch['file_target'][0]
            for r1 in self.replace_pcd:
                file = file.replace(r1, self.replace_pcd[r1])
            pt = o3d.io.read_point_cloud(file)
        else:
            pt = target[0, mask, :3]
            pt = torch2o3d(pt)
        pt.paint_uniform_color(np.array([0, 1, 0]))
        print('Target: Green')
        print()
        geoms = [ps, pt, ps_t]
        pt.estimate_normals()
        ps_t.estimate_normals()
        geoms = [pt, ps_t]
        # print(T)
        # print(len(corr))
        if self.draw_corr_lines:
            lines = corr2Lines(pt, ps_t, W)
            geoms.append(lines)

        if self.draw_corr_points:
            shape = list(ps_corr.shape)
            shape[-1] = 1
            ps_corr_t = torch.cat([ps_corr[..., :, :3], torch.ones(shape)], -1)
            ps_corr_t = torch2o3d(ps_corr_t[0, :, :3])
            ps_corr_t = ps_corr_t.transform(T[0, :, :].cpu().detach().numpy())
            ps_corr_t.paint_uniform_color(np.array([0, 0, 0]))
            geoms.append(ps_corr_t)
            # ps_corr_t = (T@ps_corr_t.transpose(-1, -2)).transpose(-1, -2)
            corr = [(i, i) for i in range(target.shape[-2])]
            lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
                pt, ps_corr_t, corr)
            # print(pt,ps_corr_t.shape)
            # print(lines)
            # geoms.append(lines)
        # o3d.visualization.draw_geometries([ps, pt, ps_t])
        return geoms


class Visualizer():
    def __init__(self, point_cloud_provider, width=1920, height=1080):
        self.i = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        # key_to_callback = {ord("N"): self.next,
        #                    #    ord("B"): self.prev,
        #                    ord("K"): self.change_background_to_black,
        #                    ord("S"): self.start_prev,
        #                    ord("X"): self.stop_prev, }
        self.vis.register_key_callback(ord("N"), self.next)
        self.vis.register_key_callback(
            ord("K"),  self.change_background_to_black)
        self.vis.register_key_callback(ord("S"), self.start_prev)
        self.vis.register_key_callback(ord("X"), self.stop_prev)
        self.vis.register_key_callback(ord("Z"), self.save)
        self.vis.register_key_callback(ord("R"), self.render_video)

        self.stop = False
        self.clouds = point_cloud_provider
        self.second = True
        self.vis.create_window(width = width, height = height)
        
        
        self.render = False
        self.image_files=[]


    def save(self,vis):
        for i,g in enumerate(self.geoms):
            o3d.io.write_point_cloud(f"{i}.ply",g)
        
    def updatePoints():
        pass

    def change_background_to_black(self, vis):
        opt = vis.get_render_option()
        # opt.background_color = np.asarray([0, 0, 0])
        opt.background_color = BACKGROUND
        return False
        # self.

    def next(self, vis):
        if self.second:
            # self.second = False
            # t = pytimer.Timer()
            # print('bla')
            self.vis.clear_geometries()
            self.geoms = self.clouds.getGeometries(self.i)
            for g in self.geoms:
                self.vis.add_geometry(g, reset_bounding_box=(self.i==0))
            # self.updatePoints()
            # t.toc('get data: ')
            # vis.update_geometry(self.cloud)
            # vis.poll_events()
            # vis.update_renderer()
            self.i += 1
        else:
            self.second=True
            
    def prev(self, vis):
        self.i -= self.step
        self.i = self.i % len(self.files)
        cloud = o3d.io.read_point_cloud(self.files[self.i])
        self.cloud.points = cloud.points
        vis.update_geometry(self.cloud)
        vis.poll_events()
        vis.update_renderer()

    def start_prev(self, vis):
        self.stop = False
        while not self.stop:
            self.next(vis)
            time.sleep(0.1)

    def stop_prev(self, vis):
        self.stop = True

    def render_video(self, vis):
        if not self.render:
            self.render = True
            n = len(self.clouds)
            print(f'{n} scans')
            fname = self.clouds.name()
            print('file',fname)
            for i in range(n):
                ## draw loop
                self.vis.clear_geometries()
                self.geoms = self.clouds.getGeometries(i)
                for g in self.geoms:
                    self.vis.add_geometry(g, reset_bounding_box=(i == 0))
                vis.poll_events()
                vis.update_renderer()

                image_file = f"tmp/{i}.png"
                self.vis.capture_screen_image(image_file)
                self.image_files.append(image_file)
                time.sleep(0.01)
                
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
                self.image_files, fps=30)
            clip.write_videofile(f"{fname}.mp4")
            
            ##Pad
            os.system(f'ffmpeg -i {fname}.mp4 -vf tpad=start_mode=clone:start_duration=0.5:stop_mode=clone:stop_duration=1 {fname}_pad.mp4 -y')
            # time.sleep(0.01)
            # self.vis.destroy_window()
            print('done')
        self.render = False
        

    def run(self):
        # self.cloud = self.dataloader[0]
        # self.cloud.points = o3d.utility.Vector3dVector(
        #     self.dataset[self.i]['points'])
        # self.updatePoints()

        # o3d.visualization.draw_geometries_with_key_callbacks(
        #     [self.cloud], key_to_callback)
        print('N: next')
        # print('B: previous')
        print('S: start')
        print('X: stop')
        print('R: render video')
        self.next(self.vis)
        self.vis.run()
        # print('X


# class Visualizer():
#     def __init__(self, dataloader):
#         self.i = 1
#         self.vis = o3d.visualization.VisualizerWithKeyCallback()
#         self.stop = False
#         self.cloud = o3d.geometry.PointCloud()
#         self.dataloader = dataloader

#     def updatePoints():
#         pass

#     def change_background_to_black(self, vis):
#         opt = vis.get_render_option()
#         # opt.background_color = np.asarray([0, 0, 0])
#         opt.background_color = BACKGROUND
#         return False

#     def next(self, vis):
#         self.i += 1
#         t = pytimer.Timer()
#         self.cloud.points = self.dataloader[self.i].points
#         # self.updatePoints()
#         t.toc('get data: ')
#         vis.update_geometry(self.cloud)
#         vis.poll_events()
#         vis.update_renderer()

#     def prev(self, vis):
#         self.i -= self.step
#         self.i = self.i % len(self.files)
#         cloud = o3d.io.read_point_cloud(self.files[self.i])
#         self.cloud.points = cloud.points
#         vis.update_geometry(self.cloud)
#         vis.poll_events()
#         vis.update_renderer()

#     def start_prev(self, vis):
#         self.stop = False
#         while not self.stop:
#             self.next(vis)
#             time.sleep(0.1)

#     def stop_prev(self, vis):
#         self.stop = True

#     def run(self):
#         self.cloud = self.dataloader[0]
#         # self.cloud.points = o3d.utility.Vector3dVector(
#         #     self.dataset[self.i]['points'])
#         # self.updatePoints()
#         key_to_callback = {ord("N"): self.next,
#                            #    ord("B"): self.prev,
#                            ord("K"): self.change_background_to_black,
#                            ord("S"): self.start_prev,
#                            ord("X"): self.stop_prev, }
#         o3d.visualization.draw_geometries_with_key_callbacks(
#             [self.cloud], key_to_callback)
#         print('N: next')
#         # print('B: previous')
#         print('S: start')
#         print('X: stop')
