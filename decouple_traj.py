import os
from collections import OrderedDict
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from src.models.model_utils.segment_anything.modeling.common import LayerNorm2d
from src.data_pre_process import Trajectory_Data_Pre_Process

from src.data_src.dataset_src.dataset_create import create_dataset
from src.models.base_model import Base_Model
from src.models.model_utils.U_net_CNN import UNet
from src.losses import MSE_loss, Goal_BCE_loss
from src.metrics import ADE_best_of, KDE_negative_log_likelihood, \
    FDE_best_of_goal, FDE_best_of_goal_world
from src.models.model_utils.sampling_2D_map import conditional_waypoints_sampling, softargmax_over_map, sampling, argmax_over_map, \
    TTST_test_time_sampling_trick, test_time_sampling_trick
from src.utils import get_dist_patch, get_gaussian_heatmap_patch, str2list
from src.models.model_utils.segment_anything import sam_model_registry

class WaypointsDecoder(nn.Module):
    def __init__(self, embed_dim, output_len, num_waypoints=1):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.ConvTranspose2d(embed_dim + num_waypoints, embed_dim//4, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 4),
            nn.GELU(), 
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 8),
            nn.GELU(),
            nn.Conv2d(
                embed_dim // 8,  output_len, kernel_size=1,stride=1 
            )
        )
        

    def forward(self, x): 
        """
        x: (B, C, H/64, W/64) output feature 
        endpoint_map: (B, L, H, W)
        """
        return self.predictor(x)   
            

class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: B, L, 2
        """
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x), None


class Goal_SAR_SAM(Base_Model):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.args = args
        self.device = device

        ##################
        # MODEL PARAMETERS
        ##################

        # set parameters for network architecture
        self.input_size = 2  # size of the input 2: (x,y)
        self.embedding_size = 32  # embedding dimension
        self.nhead = 8  # number of heads in multi-head attentions TF
        self.d_hidden = 2048  # hidden dimension in the TF encoder layer
        self.n_layers_temporal = 1  # number of TransformerEncoderLayers
        self.dropout_prob = 0  # the dropout probability value
        self.add_noise_traj = self.args.add_noise_traj
        self.noise_size = 16  # size of random noise vector
        self.output_size = 2  # output size

        # GOAL MODULE PARAMETERS
        self.num_image_channels = 6

        # U-net encoder channels
        self.enc_chs = (self.num_image_channels + self.args.obs_length,
                        32, 32, 64, 64, 64)
        # U-net decoder channels
        self.dec_chs = (64, 64, 64, 32, 32)

        self.extra_features = 4  # extra information to concat goals: time,
        # last positions, predicted final positions, distance to predicted goals

        ##################
        # MODEL LAYERS
        ##################
        sam = self._build_sam()
        self.image_encoder = sam.image_encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.mask_decoder = sam.mask_decoder

        self.prompt_encoder = sam.prompt_encoder
        # for param in self.prompt_encoder.parameters():
        #     param.requires_grad = False

        prompt_temporal_encoding = args.prompt_temporal_encoding
        if prompt_temporal_encoding == "gru":
            self.prompt_lstm = nn.GRU(self.prompt_encoder.embed_dim, self.prompt_encoder.embed_dim, batch_first=True)
        elif prompt_temporal_encoding == "pe":
            self.prompt_lstm = PositionalEncoding(
                self.prompt_encoder.embed_dim,
            )
        else:
            raise NotImplementedError("No prompt temporal encoding " + prompt_temporal_encoding)

        self.waypoints_decoder = WaypointsDecoder(
            embed_dim = self.prompt_encoder.embed_dim ,
            output_len=self.args.pred_length, num_waypoints = len(self.args.waypoints)
        )

        if self.args.extra_scale_up > 0:
            self.mask_decoder.output_upscaling = nn.Sequential(
                *[m for m in self.mask_decoder.output_upscaling],
                *[nn.Sequential(nn.ConvTranspose2d(self.mask_decoder.transformer_dim//8,self.mask_decoder.transformer_dim//8, kernel_size=2, stride=2),
                    LayerNorm2d(self.mask_decoder.transformer_dim//8),
                    nn.GELU()) for i in range(self.args.extra_scale_up-1)
                  ],
                nn.ConvTranspose2d(self.mask_decoder.transformer_dim // 8, self.mask_decoder.transformer_dim // 8,
                                   kernel_size=2, stride=2),
                nn.GELU()
            )

        

        size = int(4200)
        middle = size // 2
        dist_template = np.indices([size, size]) - np.array([middle, middle])[:, None, None]
        self.dist_template = torch.Tensor(dist_template).permute(1, 2, 0).to(device)

        candidate_idx_list = list(range( self.args.pred_length))
        optimise_waypoints_idx = [candidate_idx_list[i] for i in self.args.waypoints]
        candidate_idx_list = [ i for i in candidate_idx_list if i not in optimise_waypoints_idx]

        if self.args.optimise_aux_waypoints:
            optimise_waypoints_idx = [self.args.obs_length - 1] + optimise_waypoints_idx
            if len(optimise_waypoints_idx) == 2:
                optimise_waypoints_idx =  [candidate_idx_list[-5], candidate_idx_list[-7]] + optimise_waypoints_idx
            else:
                optimise_waypoints_idx = [candidate_idx_list[-5]] + optimise_waypoints_idx

        self.optimise_waypoints_idx = optimise_waypoints_idx

    def _build_sam(self):
        return sam_model_registry[self.args.sam_type](
            image_encoder=True,
            prompt_encoder=True,
            mask_decoder=True,
        ).to(self.device)


    @torch.no_grad()
    def prepare_inputs(self, batch_data, batch_id):
        """
        Prepare inputs to be fed to a generic model.
        """
        # we need to remove first dimension which is added by torch.DataLoader
        # float is needed to convert to 32bit float
        selected_inputs = {k: v.squeeze(0).float().to(self.device) if \
            torch.is_tensor(v) else v for k, v in batch_data.items()}
        # extract seq_list
        seq_list = selected_inputs["seq_list"]
        # decide which is ground truth
        if not self.args.multi_agent_mode:
            # decide which is ground truth
            selected_inputs["abs_pixel_coord"] = selected_inputs["abs_pixel_coord"][:, seq_list.all(0)]

            ground_truth = selected_inputs["abs_pixel_coord"]
            seq_start_end = batch_id['seq_start_end'][0]
            new_start = 0
            new_seq_start_end = []
            for start, end in seq_start_end:
                offset = seq_list[:, start:end].all(0).sum()
                new_seq_start_end.append((new_start, new_start + offset))
                new_start = new_start + offset
            selected_inputs['seq_start_end'] = new_seq_start_end
            seq_list = seq_list[:, seq_list.all(0)]


        ground_truth = selected_inputs["abs_pixel_coord"]

        scene_name = batch_id["scene_name"][0]
        scene = self.dataset.scenes[scene_name]
        selected_inputs["scene"] = scene
        H, W = selected_inputs['tensor_image'].shape[-2:]

        selected_inputs["input_traj_maps"] = get_gaussian_heatmap_patch(
            # create_CNN_inputs_loop(selected_inputs["abs_pixel_coord"], semantic_image[0]).to(self.device)
            self.dist_template, ground_truth[self.args.obs_length:].permute(1, 0, 2).reshape(-1, 2),
            H, W, nsig= 8 #min(W, H) / 64
        ).reshape([-1, self.args.pred_length, H, W])

        self.image_encoder.eval()

        tensor_image = selected_inputs['tensor_image'].flip(0).clone()

        pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(tensor_image)
        pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(tensor_image)
        tensor_image = (tensor_image - pixel_mean) / pixel_std
        # tensor_image = F.pad(tensor_image, (0,padw, 0, padh))
        h, w = tensor_image.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        tensor_image = F.pad(tensor_image, (0, padw, 0, padh))

        selected_inputs['scene_features'] = list(self.image_encoder(tensor_image[None]))
        if self.args.img_zeros:
            selected_inputs['scene_features'][0] = torch.zeros_like(selected_inputs['scene_features'][0])
            selected_inputs['scene_features'][1] = [torch.zeros_like(i) for i in selected_inputs['scene_features'][1] ]
        selected_inputs['seq_start_end'] = batch_id['seq_start_end'][0]
        selected_inputs['down_factor'] = batch_id['down_factor'].item()
        selected_inputs['waypoints_index'] = [-1, -2]
        return selected_inputs, ground_truth, seq_list

    def init_losses(self):
        losses = {
            "traj_BCE_loss": 0,
            "goal_BCE_loss": 0,
        }
        return losses

    def set_losses_coeffs(self):
        losses_coeffs = {
            "traj_BCE_loss": 1e3,
            "goal_BCE_loss": 1e3,
        }
        return losses_coeffs

    def init_train_metrics(self):
        train_metrics = {
            "ADE": [],
            "FDE": [],
        }
        return train_metrics

    def init_test_metrics(self):
        test_metrics = {
            "ADE": [],
            "FDE": [],
            "ADE_world": [],
            "FDE_world": [],
            # "NLL": [],
        }
        return test_metrics

    def init_best_metrics(self):
        best_metrics = {
            "ADE": 1e9,
            "FDE": 1e9,
            "ADE_world": 1e9,
            "FDE_world": 1e9,
            "goal_BCE_loss": 1e9,
        }
        return best_metrics

    def best_valid_metric(self):
        return "FDE"

    def compute_model_losses(self,
                             outputs,
                             ground_truth,
                             loss_mask,
                             inputs,
                             aux_outputs):
        """
        Compute loss for a generic model.
        """
        out_maps_GT_goal = inputs["input_traj_maps"]
        H, W = inputs['tensor_image'].shape[-2:]
        goal_logit_map = aux_outputs["goal_logit_map"]
        pred_traj_map = aux_outputs["pred_traj_map"][:1]
        pred_traj_map = pred_traj_map[:, loss_mask.sum(0) > 0]
        G, B,L= pred_traj_map.shape[:3]
        # pred_traj_map = F.interpolate(pred_traj_map.reshape(-1, L, 256,256), (1024, 1024),  mode="bilinear", align_corners=False).reshape(G, B,L,1024,1024)
        # pred_traj_map = pred_traj_map[..., :H, :W]
        
        goal_logit_map = goal_logit_map[:, loss_mask.sum(0) > 0]
        out_maps_GT_goal = out_maps_GT_goal[loss_mask.sum(0) > 0]
        
        # candidate_idx_list = list(range( self.args.pred_length))
        # optimise_waypoints_idx = [candidate_idx_list[i] for i in self.args.waypoints]
        # candidate_idx_list = [ i for i in candidate_idx_list if i not in optimise_waypoints_idx]
        #
        # if self.args.optimise_aux_waypoints:
        #     optimise_waypoints_idx = [self.args.obs_length - 1] + optimise_waypoints_idx
        #     if len(optimise_waypoints_idx) == 2:
        #         optimise_waypoints_idx =  [candidate_idx_list[-5], candidate_idx_list[-7]] + optimise_waypoints_idx
        #     else:
        #         optimise_waypoints_idx = [candidate_idx_list[-5]] + optimise_waypoints_idx

        
        out_maps_GT_goal_resized = get_gaussian_heatmap_patch(
            # create_CNN_inputs_loop(selected_inputs["abs_pixel_coord"], semantic_image[0]).to(self.device)
            self.dist_template, ground_truth[self.args.obs_length:, loss_mask.sum(0) > 0].permute(1, 0, 2).reshape(-1, 2) // 4,
            256, 256, nsig=8
        ).reshape([-1, self.args.pred_length, 256,256])

        loss_mask = loss_mask[:, loss_mask.sum(0) > 0]
        # import matplotlib.pyplot as plt
        # plt.subplot(1,2,1)
        # plt.imshow(torch.sigmoid(pred_traj_map[0,0,-1]).detach().cpu().numpy())
        # plt.subplot(1,2,2)
        # plt.imshow( out_maps_GT_goal_resized[0,].amax(0).detach().cpu().numpy())
        # plt.plot(ground_truth.cpu().numpy()[:, 0,0] * 0.25, ground_truth[:,0,1].cpu().numpy()* 0.25, "r*" )
        # plt.show()
        traj_BCE_loss = Goal_BCE_loss(
            pred_traj_map, out_maps_GT_goal_resized, loss_mask
        )

        goal_BCE_loss = Goal_BCE_loss(
            goal_logit_map, out_maps_GT_goal[:, self.optimise_waypoints_idx], loss_mask)

        losses = {
            # "traj_MSE_loss": MSE_loss(outputs, ground_truth, loss_mask),
            "goal_BCE_loss": goal_BCE_loss,
            "traj_BCE_loss": traj_BCE_loss
        }

        return losses

    def compute_model_metrics(self,
                              metric_name,
                              phase,
                              predictions,
                              ground_truth,
                              metric_mask,
                              all_aux_outputs,
                              inputs,
                              obs_length=8):
        """
        Compute model metrics for a generic model.
        Return a list of floats (the given metric values computed on the batch)
        """
        if phase == 'test':
            compute_nll = self.args.compute_test_nll
            num_samples = self.args.num_test_samples
        elif phase == 'valid':
            compute_nll = self.args.compute_valid_nll
            num_samples = self.args.num_valid_samples
        else:
            compute_nll = False
            num_samples = 1
        down_factor = inputs['down_factor']
        # scale back to original dimension
        predictions = predictions.detach() * down_factor
        ground_truth = ground_truth.detach() * down_factor

        # convert to world coordinates
        scene = inputs["scene"]
        pred_world = []
        for i in range(predictions.shape[0]):
            pred_world.append(scene.make_world_coord_torch(predictions[i]))
        pred_world = torch.stack(pred_world)

        GT_world = scene.make_world_coord_torch(ground_truth)
        if metric_name == 'ADE':
            return ADE_best_of(
                predictions, ground_truth, metric_mask, obs_length)
        elif metric_name == 'FDE':
            return FDE_best_of_goal(all_aux_outputs, ground_truth,
                                    metric_mask, down_factor)
        if metric_name == 'ADE_world':
            return ADE_best_of(
                pred_world, GT_world, metric_mask, obs_length)
        elif metric_name == 'FDE_world':
            return FDE_best_of_goal_world(all_aux_outputs, scene,
                                          GT_world, metric_mask, down_factor)
        elif metric_name == 'NLL':
            if compute_nll and num_samples > 1:
                return KDE_negative_log_likelihood(
                    predictions, ground_truth, metric_mask, obs_length)
            else:
                return [0, 0, 0]
        else:
            raise ValueError("This metric has not been implemented yet!")

    def predict_goal(self, scene_features, sparse_embeddings, dense_embeddings, **kwargs):
        scene_features, _ = scene_features
        low_res_masks, low_res_aux_masks, iou_predictions, hs, src = self.mask_decoder(
            image_embeddings=scene_features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            **kwargs
        )

        return low_res_masks, hs, src

    def forward(self, inputs, num_samples=1, mode="train"):
        if_test = (mode in ["valid", "test"])
        batch_coords = inputs["abs_pixel_coord"]
        # Number of agent in current batch_abs_world
        seq_length, num_agents, _ = batch_coords.shape

        ##################
        # PREDICT GOAL
        ##################
        # extract precomputed map for goal goal_idx
        # tensor_image = inputs["tensor_image"].unsqueeze(0).\
        #     repeat(num_agents, 1, 1, 1)

        # scene encoding
        features = inputs['scene_features']
        del inputs['scene_features']
        observed = batch_coords[:self.args.obs_length].permute(1, 0, 2)
        # prompt_encoding:
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(observed, torch.ones_like(observed[:, :, 0])),
            masks=None,
            boxes=None
        )  

        temporal_sparse_embeddings = torch.cat(
            [self.prompt_lstm(sparse_embeddings[:, :-1])[0],  # [:,required_indexes], # TODO: add last hidden [:,-1:],
             sparse_embeddings[:, -1:]], dim=1
        )  # N,L+1,D∗H, out
        # heatmap decoder
        pred_goal_args = dict(
            seq_start_end= inputs['seq_start_end'] if self.args.multi_agent_mode else None,
            multimask_output=True,  # params['auxillary_waypoints_optimisation'],
            scale_up_on_raw=False,
        )
        pred_goal_map, hs, src = self.predict_goal(features, temporal_sparse_embeddings, dense_embeddings, **pred_goal_args)

        if self.args.scene_cf == "zeros":
            pred_goal_map = pred_goal_map - self.predict_goal((torch.zeros_like(features[0]), features[1]), temporal_sparse_embeddings,
                                                              dense_embeddings, **pred_goal_args)[0]
                                                              
        elif self.args.scene_cf == "random":
            pred_goal_map = pred_goal_map - self.predict_goal(((torch.rand_like(features[0]) - 0.5) / 5,features[1]), temporal_sparse_embeddings,
                                                           dense_embeddings, **pred_goal_args)[0]
        elif self.args.scene_cf == "empty":
            with torch.no_grad():
                empty_features, _ = list(self.image_encoder(torch.zeros(1, 3, 1024, 1024).to(features[0])))

            pred_goal_map = pred_goal_map - self.predict_goal((empty_features,features[1]), temporal_sparse_embeddings,
                                                           dense_embeddings, **pred_goal_args)[0]

        pred_goal_map = F.interpolate(pred_goal_map, (1024, 1024), mode="bilinear", align_corners=False)
        H, W = inputs['tensor_image'].shape[-2:]
        pred_goal_map = pred_goal_map[..., :H, :W]

        goal_logit_map_start = pred_goal_map
        
        goal_prob_map = torch.sigmoid(
            goal_logit_map_start[:, -1:] / self.args.sampler_temperature)
        
        if self.args.use_ttst and num_samples > 3:
            goal_point_start = test_time_sampling_trick(
                goal_prob_map,
                num_goals=num_samples,
                rel_thresh= self.args.rel_thresh,
                device=self.device)
            goal_point_start = goal_point_start.squeeze(2).permute(1, 0, 2)
        elif num_samples == 1:
            goal_point_start = argmax_over_map(goal_prob_map)
        else:
            goal_point_start = sampling(goal_prob_map, num_samples=num_samples,
                    rel_threshold= self.args.rel_thresh)
            goal_point_start = goal_point_start.squeeze(1)# [batch, sample, 2]

        if len(self.args.waypoints) > 1:
            goal_point_start = conditional_waypoints_sampling(
                    last_observed =  batch_coords[self.args.obs_length - 1],
                    goal_samples = goal_point_start.permute(1,0,2)[..., None,:],
                    pred_waypoint_map_sigmoid = torch.sigmoid(goal_logit_map_start[:, inputs['waypoints_index']] / self.args.sampler_temperature),
                ) # (K_g` , B, L, 2)
            goal_point_start = goal_point_start.permute(1,0,2, 3) # (L ,K_g, B,  2)
        else:
            goal_point_start = goal_point_start[:,:, None]
        bs = pred_goal_map.shape[0]
        all_outputs = []
        topk = min(1 if mode == "valid" else num_samples, num_samples)
        all_aux_outputs = {
            "goal_logit_map": pred_goal_map[None],
            "goal_point": [goal_point_start[:, i,-1] for i in range(num_samples)],
            "pred_traj_map": torch.empty((topk, bs, self.args.pred_length, 256,256)).to(pred_goal_map)
        }

        if if_test:
            # choose the best endpoint. To speed up the test
            gt_goals = batch_coords[-1][:,None, None]
            _, best_index = torch.topk(torch.linalg.norm( goal_point_start[:,:,-1:] - gt_goals, dim=-1),topk, dim=1, largest=False)
            goal_point_start = torch.cat([
                goal_point_start[torch.arange(goal_point_start.shape[0]), best_index[:,i].flatten()][:,None] for i in range(topk)
                ], dim=1)

        for sample_idx in range(topk):

            if if_test:
                goal_point = goal_point_start[:, sample_idx]
            else:
                # teacher forcing

                goal_point = batch_coords[self.args.obs_length:][self.args.waypoints].permute(1,0,2)
            waypoint_map = get_dist_patch(
                # create_CNN_inputs_loop(selected_inputs["abs_pixel_coord"], semantic_image[0]).to(self.device)
                self.dist_template, goal_point.reshape(-1,2) / 16,
                64, 64, #nsig=min(W, H) / 64
            ).reshape([-1, len(self.args.waypoints), 64, 64])
            pred_traj_map = self.waypoints_decoder(
                torch.cat(
                    [waypoint_map,  src], dim=1
                )
            )
            # pred_traj_map = F.interpolate(pred_traj_map, (1024, 1024), mode="bilinear", align_corners=False)
            # pred_traj_map = pred_traj_map[..., :H, :W]
            #if sample_idx == 0:
            all_aux_outputs['pred_traj_map'][sample_idx] = pred_traj_map
            pred_traj = softargmax_over_map(pred_traj_map).permute(1,0,2).float() * 4
            
            all_outputs.append(torch.cat([batch_coords[0:self.args.obs_length], pred_traj], dim=0))


            

        # stack predictions
        all_outputs = torch.stack(all_outputs)
        # from list of dict to dict of list (and then tensors)
        all_aux_outputs = {k: torch.stack(v) if isinstance(v, list) else v
                           for k,v in all_aux_outputs.items()}
        return all_outputs, all_aux_outputs

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict = OrderedDict(
            {k: v for k, v in state_dict.items() if "image_encoder" not in k}
        )
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):

        # state_dict = OrderedDict({k:v for k,v in state_dict.items() if "waypoints_decoder" not in k})
        if not any(["image_encoder" in k for k in state_dict.keys()]):
            state_dict.update({
                "image_encoder." + k: v for k, v in self.image_encoder.state_dict().items()
            })
        # goal_sar_ckpt = torch.load("checkpoints/Goal_SAR_best_model.pt")['model_state_dict']
        # for k, v in goal_sar_ckpt.items():
        #     if "goal_module" not in k:
        #         state_dict[k] = v
        return super().load_state_dict(state_dict, strict=False)

    @staticmethod
    def extra_args(parser):
        from src.utils import str2bool
        parser.add_argument('--prompt_temporal_encoding',choices= ["pe", "gru"], default="gru")
        parser.add_argument(
            '--scene_cf', default=None, type=str, choices=['zeros', 'random', "empty"]
        )

        parser.add_argument(
            '--multi_agent_mode',default=False, type=str2bool, const=True, nargs='?'
        )
        parser.add_argument(
            '--sam_type',default='vit_b', choices=['default', 'vit_h', 'vit_l', 'vit_b', 'vit_tiny']
        )

        parser.add_argument(
            '--resize_mode', const="longside", default="longside", nargs='?', type=lambda x:"longside"
        )

        parser.add_argument(
            '--down_factor', const=1024, default=1024,nargs='?', type=lambda x: 1024
        )

        parser.add_argument(
            '--use_seg_mask', const=False, default=False, nargs='?', type=lambda x: False
        )

        parser.add_argument(
            '--extra_scale_up', default=0, type=int
        )
        
        parser.add_argument(
            '--optimise_aux_waypoints', default=True,type=str2bool, nargs="?", const=True
        )
        return parser

    @staticmethod
    def set_fixed_args(args):
        args.use_seg_mask = False
        args.use_gt_map = False
        args.resize_mode = "longside"
        args.down_factor = 1024
        return args

