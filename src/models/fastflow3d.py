
"""
This file is directly copied from: 
https://github.com/kylevedder/zeroflow/blob/master/models/fast_flow_3d.py

with slightly modification to have unified format with all benchmark.
"""

import dztimer, torch
import torch.nn as nn
import torch.nn.functional as F

from torch import knn
from .basic.unet import FastFlow3DUNet
from .basic.encoder import DynamicEmbedder #编码器
from .basic.decoder import LinearDecoder,ConvGRUDecoder #解码器
from .basic import cal_pose0to1,MiniPointNet

class FastFlow3D(nn.Module):
    def __init__(self, voxel_size=[0.2, 0.2, 6],
                 point_cloud_range=[-51.2, -51.2, -3, 51.2, 51.2, 3],
                 grid_feature_size=[512, 512],
                 decoder_option="gru",
                 num_iters=4,
                 search_radius=1.0):
        super().__init__()
        self.embedder = DynamicEmbedder(voxel_size=voxel_size,
                                        pseudo_image_dims=grid_feature_size,
                                        point_cloud_range=point_cloud_range,
                                        feat_channels=32)

        self.backbone = FastFlow3DUNet()
        if decoder_option == "gru":
            self.coarse_head = ConvGRUDecoder(num_iters=num_iters)
        elif decoder_option == "linear":
            self.coarse_head = LinearDecoder()
        self.refine_head = MiniPointNet(in_channels=6)

        self.search_radius = search_radius

        self.timer = dztimer.Timing()
        self.timer.start("Total")
        
    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("model.") :]: v for k, v in ckpt.items() if k.startswith("model.")
        }
        print("\nLoading... model weight from: ", ckpt_path, "\n")
        return self.load_state_dict(state_dict=state_dict, strict=False)
    
    def _model_forward(self, pc0s, pc1s):

        pc0_before_pseudoimages, pc0_voxel_infos_lst = self.embedder(pc0s)
        pc1_before_pseudoimages, pc1_voxel_infos_lst = self.embedder(pc1s)

        grid_flow_pseudoimage = self.backbone(pc0_before_pseudoimages,
                                            pc1_before_pseudoimages)
        coarse_flows = self.coarse_head(
            torch.cat((pc0_before_pseudoimages, pc1_before_pseudoimages),
                    dim=1), grid_flow_pseudoimage, pc0_voxel_infos_lst)

        # 准备特征：假设embedder输出点级特征（若无，需在DynamicEmbedder中添加point_feats输出）
        pc0_point_feats = [info.get('point_feats',info['points']) for info in pc0_voxel_infos_lst]
        pc1_point_feats = [info.get('point_feats', info['points']) for info in pc1_voxel_infos_lst]

        # 阶段二：基于点的光流精调
        fine_flows = []
        for batch_idx, (coarse_flows,pc0_points,pc1_points,pc0_feats,pc1_feats) in enumerate(
            zip(coarse_flows,[e["points"] for e in pc0_voxel_infos_lst],
                [e["points"] for e in pc1_voxel_infos_lst],pc0_point_feats,pc1_point_feats)
        ):
            # 计算大致目标位置
            target_pos = pc0_points + coarse_flows
            # 建立关联窗口：使用kNN查找pc1中距离target_pos < radius的点（高效替代全扫描）
            # knn返回索引：graph = knn(pc1_points, target_pos, k=32)  # 限制最大k以控制计算
            # 但为精确窗口，使用距离过滤
            dists = torch.cdist(target_pos,pc1_points)
            mask = dists < self.search_radius
            residual_flow = torch.zeros_like(coarse_flows)

            for i in range(len(pc0_points)):
                local_idx = mask[i].nonzero().squeeze(-1)
                if len(local_idx) == 0:
                    continue
                local_points = pc1_points[local_idx] - target_pos[i]
                local_feats = pc1_feats[local_idx]
                point_feats = pc0_feats[i].unsqueeze(0)
                residual = self.refine_head(local_points, point_feats, local_feats)
                residual_flow[i] = residual

            fine_flows.append(coarse_flows + residual_flow)

        pc0_points_lst = [e["points"] for e in pc0_voxel_infos_lst]
        pc0_valid_point_idxes = [e["point_idxes"] for e in pc0_voxel_infos_lst]
        pc1_points_lst = [e["points"] for e in pc1_voxel_infos_lst]
        pc1_valid_point_idxes = [e["point_idxes"] for e in pc1_voxel_infos_lst]

        pc0_warped_pc1_points_lst = [
            pc0_points + fine_flow
            for pc0_points, fine_flow in zip(pc0_points_lst, fine_flows)
        ]

        return {
            "flow": fine_flows,
            "coarse_flow":coarse_flows,
            "pc0_points_lst": pc0_points_lst,
            "pc0_warped_pc1_points_lst": pc0_warped_pc1_points_lst,
            "pc0_valid_point_idxes": pc0_valid_point_idxes,
            "pc1_points_lst": pc1_points_lst,
            "pc1_valid_point_idxes": pc1_valid_point_idxes,
            "num_occupied_voxels": [grid_flow_pseudoimage.size(-1)*grid_flow_pseudoimage.size(-2)]
        }

    def forward(self, batch,
                compute_cycle=False,
                compute_symmetry_x=False,
                compute_symmetry_y=False):

        self.timer[0].start("Data Preprocess")
        batch_sizes = len(batch["pose0"])

        pose_flows = []
        transform_pc0s = []
        for batch_id in range(batch_sizes):
            selected_pc0 = batch["pc0"][batch_id]
            self.timer[0][0].start("pose")
            pose_0to1 = cal_pose0to1(batch["pose0"][batch_id], batch["pose1"][batch_id])
            self.timer[0][0].stop()
            
            self.timer[0][1].start("transform")
            # transform selected_pc0 to pc1
            transform_pc0 = selected_pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
            self.timer[0][1].stop()
            pose_flows.append(transform_pc0 - selected_pc0)
            transform_pc0s.append(transform_pc0)

        pc0s = torch.stack(transform_pc0s, dim=0)
        pc1s = batch["pc1"]
        self.timer[0].stop()

        model_res = self._model_forward(pc0s, pc1s)

        ret_dict = model_res
        ret_dict["pose_flow"] = pose_flows
        if compute_cycle:
            # The warped pointcloud, original pointcloud should be the input to the model
            pc0_warped_pc1_points_lst = model_res["pc0_warped_pc1_points_lst"]
            pc0_points_lst = model_res["pc0_points_lst"]
            # Some of the warped points may be outside the pseudoimage range, causing them to be clipped.
            # When we compute this reverse flow, we need to solve for the original points that were warped to the clipped points.
            backward_model_res = self._model_forward(pc0_warped_pc1_points_lst,
                                                     pc0_points_lst)
            # model_res["reverse_flow"] = backward_model_res["flow"]
            # model_res[
            #     "flow_valid_point_idxes_for_reverse_flow"] = backward_model_res[
            #         "pc0_valid_point_idxes"]
            ret_dict["backward"] = backward_model_res

        if compute_symmetry_x:
            pc0s_sym = pc0s.clone()
            pc0s_sym[:, :, 0] *= -1
            pc1s_sym = pc1s.clone()
            pc1s_sym[:, :, 0] *= -1
            model_res_sym = self._model_forward(pc0s_sym, pc1s_sym)
            ret_dict["symmetry_x"] = model_res_sym

        if compute_symmetry_y:
            pc0s_sym = pc0s.clone()
            pc0s_sym[:, :, 1] *= -1
            pc1s_sym = pc1s.clone()
            pc1s_sym[:, :, 1] *= -1
            model_res_sym = self._model_forward(pc0s_sym, pc1s_sym)
            ret_dict["symmetry_y"] = model_res_sym

        return ret_dict