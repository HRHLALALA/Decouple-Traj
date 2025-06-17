from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from src.data_src.dataset_src.dataset_create import create_dataset
from src.models.base_model import Base_Model
from src.models.model_utils.U_net_CNN import UNet, Decoder
from src.losses import MSE_loss, Goal_BCE_loss
from src.metrics import ADE_best_of, KDE_negative_log_likelihood, \
    FDE_best_of_goal, FDE_best_of_goal_world
from src.models.model_utils.cnn_big_images_utils import create_CNN_inputs_loop
from src.models.model_utils.conv_lib import CoordConv2d, RelativeCoordConv2d
from src.models.model_utils.sampling_2D_map import conditional_waypoints_sampling, sampling, argmax_over_map, \
    TTST_test_time_sampling_trick, softargmax_over_map, test_time_sampling_trick, softmax_over_map
from src.models.ynet import YNet
from src.utils import get_gaussian_heatmap_patch, get_dist_patch, str2bool
from src.models.modules import YNetEncoder, YNetDecoder

import torch.nn.functional as F

class DynamicHead(nn.Module):
    def __init__(self, in_dim, out_dim, coord_conv=True, dynamic=False):
        super(DynamicHead, self).__init__()
        self.coord_conv = coord_conv
        if coord_conv == 1:
            self.middle = CoordConv2d(
                in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=1,
                padding=1
            )
        elif coord_conv == 0:
            self.middle= nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=1,
                                  padding=1)
        elif coord_conv == 4:
            self.middle= nn.Identity()
        elif coord_conv == 2:
            self.middle = RelativeCoordConv2d(
                in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=1,
                padding=1
            )
        else:
            raise RuntimeError("Run Mode")

        if not dynamic:
            self.predictor = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=0)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = nn.LeakyReLU(0.02, True) if coord_conv != 4 else nn.Identity()

    def predict_with_kernel(self, x, kernel):
        """
        x: [B, C, H,W] or [B , C, H, W]
        kernel: (K, B, pred_len, C)

        return:
        [B, pred_len, H,W] if kernel is None
        """

        B, C, H, W = x.shape
        K = kernel.shape[0]
        kernel = kernel.permute(1,0, 2,3).reshape(B, -1, C)
        return torch.matmul(kernel, x.view(B, C, -1)).view(B, K, -1, H, W).permute(1,0,2,3,4)

    def forward(self, x, coord_map=None, kernel=None):
        if self.coord_conv == 1 and coord_map is not None:
            x = self.middle(x,coord_map)
        else:
            x = self.middle(x)
        x = self.activation(x)
        if kernel is None:
            indv_out = self.predictor(x)
        else:
            # assert kernel is not None and kernel.shape[-2] == self.out_dim, f'kernel {kernel.shape} must match {self.out_dim} at dim=-2'
            indv_out = self.predict_with_kernel(x, kernel)  # last predictor layer

        return indv_out

# torch.use_deterministic_algorithms(True)
class KernalHead(DynamicHead):
    def __init__(self, in_dim, out_dim, waypoints = 1, coord_conv=1, dynamic=False, longterm_adapter=False):
        super(KernalHead, self).__init__(in_dim, out_dim, coord_conv, dynamic)
        self.predictor = nn.Linear(in_dim * waypoints,out_dim)
        nn.init.kaiming_normal_(self.predictor.weight)
        nn.init.constant_(self.predictor.bias,0)
        self.waypoints = waypoints
        if waypoints > 1 and longterm_adapter:
            self.longterm_adaptor = nn.Sequential(
                nn.Linear(in_dim * waypoints, 128),
                nn.ReLU(),
                nn.Linear(128, in_dim * waypoints),
                nn.ReLU(),
            )
        else:
            self.longterm_adaptor = nn.Identity()
    def forward(self, x,coord_map=None, pos=None):
        """
        x: [B, C, H, W]
        pos: (K, B, L,  2)
        return: kernel (B, out_C, H, W) if pos is None else (k, B L * out_C)
        """
        B = x.shape[0]
        if self.coord_conv == 1 and coord_map is not None:
            x = self.middle(x, coord_map)
        else:
            x = self.middle(x)
        x = self.activation(x)
        H, W = x.shape[-2:]
        if pos is None:
            # return a kernel map
            kernel_map = self.predictor(x.permute(0, 2,3,1)).permute(0,3,1,2)
            pred_traj_map = kernel_map[:,-2:]
            return kernel_map[:,:-2],pred_traj_map
        else:
            try:
                K = pos.shape[0]

                pos[...,1] = pos[..., 1].clamp(0, H-1)
                pos[...,0] = pos[..., 0].clamp(0, W-1)
                indice_flatten = (pos[..., 1].long() * W + pos[..., 0].long()) #(K, B, num_kernel)
                indice_flatten = indice_flatten.permute(1,0,2) # (B, K, L)
                L = indice_flatten.shape[-1]
                x_flatten = x[:, None, None].expand(-1, K,  L, -1,-1,-1).flatten(-2,-1) #(B, K, num_kernel, C, L)
            except:
                import pdb; pdb.set_trace()
                raise
            if indice_flatten.lt(0).any() or indice_flatten.ge(x_flatten.shape[-1]).any():
                import pdb; pdb.set_trace()
            try:
                kernel_seq = torch.take_along_dim(
                    x_flatten,indice_flatten[:,:, :, None, None],dim=-1
                ).squeeze(-1)
                kernel_seq = kernel_seq.flatten(start_dim = -2, end_dim=-1)
            except:
                import pdb; pdb.set_trace()
                raise
            kernel_seq = self.longterm_adaptor(kernel_seq)
            return  self.predictor(kernel_seq).permute(1,0,2)


class HyperTraj(YNet):
    def __init__(self, args, device):
        super(HyperTraj, self).__init__(args, device)
        self.args.coord_conv = [int(i) for i in self.args.coord_conv.split(",")]

        self.goal_head = DynamicHead(
            self.dec_chs[-1], self.args.pred_length, coord_conv=self.args.coord_conv[2]
        )
        self.kernel_head = KernalHead(
            self.dec_chs[-1] , (self.dec_chs[-1] + 2) * self.args.pred_length,
            longterm_adapter=args.longterm_adaptor,
            waypoints=len(args.waypoints), coord_conv=self.args.coord_conv[1]
        )

        self.traj_decoder = YNetDecoder(self.enc_chs[1:],  self.dec_chs, output_len=self.args.pred_length)
        self.traj_head = DynamicHead(
            self.dec_chs[-1], self.args.pred_length, coord_conv=self.args.coord_conv[2], dynamic=True
        )

    def init_train_metrics(self):
        train_metrics = {
            "ADE": [],
            "FDE": [],
            "EC": [],
        }
        return train_metrics

    def init_test_metrics(self):
        test_metrics = {
            "ADE": [],
            "FDE": [],
            "ADE_world": [],
            "FDE_world": [],
            # "NLL": [],
            "EC": [],
        }
        return test_metrics

    def init_best_metrics(self):
        best_metrics = {
            "ADE": 1e9,
            "FDE": 1e9,
            "ADE_world": 1e9,
            "FDE_world": 1e9,
            "goal_BCE_loss": 1e9,
            "EC": 1e9,
        }
        return best_metrics
    def forward(self, inputs, num_samples=1, mode="train"):
        if_test = (mode in ["valid", "test"])
        batch_coords = inputs["abs_pixel_coord"]
        # Number of agent in current batch_abs_world
        seq_length, num_agents, _ = batch_coords.shape

        ##################
        # PREDICT GOAL
        ##################
        # extract precomputed map for goal goal_idx
        if not self.args.use_seg_mask:
            if inputs.get('epoch', 1) < self.args.unfreeze_segmentor_after_epoch:
                self.semantic_segmentation.eval()
                training_segmentor = False
            else:
                training_segmentor = True

            tensor_image = self.segmentation(inputs['tensor_image_preprocessed'][None], training_segmentor)
            self.semantic_segmentation.train()
        else:
            tensor_image = inputs['tensor_image_preprocessed'][None]
        tensor_image = tensor_image. \
            repeat(num_agents, 1, 1, 1)
        obs_traj_maps = inputs["input_traj_maps"][:, 0:self.args.obs_length]

        input_goal_module = torch.cat((tensor_image, obs_traj_maps), dim=1)

        enc_ftrs = self.encoder(input_goal_module)
        # compute goal maps

        dec_feats = self.goal_decoder(enc_ftrs, return_mid_feats=True)
        goal_logit_map_start = self.goal_head(dec_feats[-1])
        goal_prob_map = torch.sigmoid(
            goal_logit_map_start[:, -1:] / self.args.sampler_temperature)
        # import pdb; pdb.set_trace()
        if self.args.use_ttst and num_samples > 3:
            goal_point_start = test_time_sampling_trick(
                goal_prob_map,
                num_goals=num_samples,
                rel_thresh=self.args.rel_thresh,
                device=self.device)
            goal_point_start = goal_point_start.squeeze(2).permute(1, 0, 2)
        elif num_samples == 1:
            goal_point_start = argmax_over_map(goal_prob_map).float()
        else:
            goal_point_start = sampling(goal_prob_map, num_samples=num_samples, rel_threshold=self.args.rel_thresh)
            goal_point_start = goal_point_start.squeeze(1)  # [batch, sample, 2]

        if self.args.use_cws and len(self.args.waypoints) > 1:
            goal_point_start = conditional_waypoints_sampling(
                last_observed=batch_coords[self.args.obs_length - 1],
                goal_samples=goal_point_start.permute(1, 0, 2)[..., None, :],
                pred_waypoint_map_sigmoid=torch.sigmoid(
                    goal_logit_map_start[:, self.args.waypoints] / self.args.sampler_temperature),
            )  # (K_g` , B, L, 2)
            goal_point_start = goal_point_start.permute(1, 0, 2, 3)  # (L ,K_g, B,  2)
        else:
            goal_point_start = goal_point_start[:, :, None]

        # START SAMPLES LOOP
        all_outputs = []
        # import pdb; pdb.set_trace()
        topk = min(1 if mode == "valid" else num_samples, num_samples)
        all_aux_outputs = {
            "goal_logit_map": goal_logit_map_start,
            "goal_point": [goal_point_start[:, i, -1] for i in range(num_samples)],
            "pred_traj_map": torch.empty((topk,) + goal_logit_map_start.shape).to(goal_logit_map_start)
        }

        if mode in ["valid", "test"]:
            # choose the best endpoint. To speed up the test
            gt_goals = batch_coords[-1][:, None, None]
            _, best_index = torch.topk(torch.linalg.norm(goal_point_start[:, :, -1:] - gt_goals, dim=-1), topk,
                                       largest=False, dim=1)
            goal_point_start = torch.cat([
                goal_point_start[torch.arange(goal_point_start.shape[0]), best_index[:, i].flatten()][:, None] for i in
                range(topk)
            ], dim=1)

        if self.args.share_goal_decoder:
            pred_traj_feat = dec_feats[-1]
        else:
            pred_traj_feat = self.traj_decoder(enc_ftrs, return_mid_feats=True)[-1]

        # if if_test:
        #     goal_point = goal_point_start.permute(1,0,2,3)
        # else:
        if not if_test or self.args.test_ec:
            goal_point_start = batch_coords[self.args.obs_length:][self.args.waypoints].permute(1, 0, 2)[:, None]
        goal_point = goal_point_start.permute(1,0,2,3)

        if not self.args.decouple_kernel_decoder:
            kernel_seq = self.kernel_head(dec_feats[-1],
                                           pos = goal_point)  # (1, B, pred_len * C + pred_len * 2)
        else:
            pred_traj_feat = self.traj_decoder(enc_ftrs, return_mid_feats=True)[-1]
            kernel_seq = self.kernel_head(pred_traj_feat,
                                          pos=goal_point)  # (1, B, pred_len * C + pred_len * 2)

        B = pred_traj_feat.shape[0]
        kernel_seq = kernel_seq.reshape(kernel_seq.shape[0], B, self.args.pred_length, -1)
        pred_traj = kernel_seq[0, ..., -2:]  # (1,B,pred_len, 2)
        kernel_seq = kernel_seq[..., :-2]
        pred_traj_map = self.traj_head(dec_feats[-1], None, kernel_seq)
        pred_traj = softargmax_over_map(pred_traj_map.reshape(-1, *pred_traj_map.shape[2:]))
        pred_traj = pred_traj.reshape(-1, B,  self.args.pred_length, 2).permute(0,2,1,3)

        # import pdb; pdb.set_trace()
        all_outputs = torch.cat([
            batch_coords[0:self.args.obs_length][None].expand(pred_traj.shape[0], -1, -1 ,-1), pred_traj], dim=1)
        all_aux_outputs['pred_traj_map'] = pred_traj_map
        all_aux_outputs['goal_logit_map'] = goal_logit_map_start[None]
        all_aux_outputs['goal_point'] = goal_point_start[:,:,-1].permute(1,0,2)
      
        return all_outputs, all_aux_outputs

    def best_valid_metric(self):
        return "ADE"

    @staticmethod
    def extra_args(parser):
        parser = YNet.extra_args(parser)
        parser.add_argument('--share_goal_decoder', default=True, const=True, nargs="?", type=str2bool)
        parser.add_argument('--decouple_kernel_decoder', default=False, const=True, nargs="?", type=str2bool)
        parser.add_argument('--coord_conv', default= "1,1,1", type=str )
        parser.add_argument('--longterm_adaptor', default=False, const=True, nargs="?", type=str2bool)
        parser.add_argument('--test_ec', default=False, const=True, nargs="?", type=str2bool)
        return parser