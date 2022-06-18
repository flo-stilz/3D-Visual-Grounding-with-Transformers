# 3DETR Architecture use for object proposal module
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import math
from functools import partial

from lib.pointnet2.pointnet2_utils import furthest_point_sample
from utils.box_util import get_3d_box
from lib.eval_helper import construct_bbox_corners
# not implemented yet
from DETR.utils.pc_util import scale_points, shift_scale_points

from DETR.datasets import build_dataset
from DETR.models.helpers import GenericMLP
from DETR.models.position_embedding import PositionEmbeddingCoordsSine
from DETR.models.transformer import (MaskedTransformerEncoder, TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer)

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

class BoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        if angle_logits.shape[-1] == 1:
            # special case for datasets with no rotation angle
            # we still use the predictions so that model outputs are used
            # in the backwards pass (DDP may complain otherwise)
            angle = angle_logits * 0 + angle_residual * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        return self.dataset_config.box_parametrization_to_corners(
            box_center_unnorm, box_size_unnorm, box_angle
        )
    def box_to_corners(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        for i in range(box_center_unnorm.shape[0]):
            print(box_size_unnorm.shape)
            if i==0:
                box_corners = construct_bbox_corners(box_center_unnorm[i].detach().cpu().numpy(), box_size_unnorm[i].detach().cpu().numpy())
            else:
                box_corners = box_corners.vstack((box_corners, construct_bbox_corners(box_center_unnorm[i].detach().cpu().numpy(), box_size_unnorm[i].detach().cpu().numpy())))
                
        return box_corners


class Object_Detection(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.input_feature_dim = input_feature_dim
        self.enc_dim=256
        self.enc_nhead=4
        self.enc_nlayers=3
        self.enc_dropout=0.1
        self.enc_activation="relu"
        self.enc_ffn_dim=128
        self.enc_pos_emeb=None
        
        self.dec_nlayers=8
        self.dec_dim=256
        self.dec_ffn_dim=256
        self.dec_dropout=0.1
        self.dec_nhead=4

        self.position_embedding="fourier",
        self.mlp_dropout=0.3,
        self.num_queries=256,
        self.encoder_only = False
        dataset_config = build_dataset("scannet")
        # --------- Architecture -----------
        
        self.pre_encoder = self.build_preencoder()
        self.encoder = self.build_encoder("masked")
        self.decoder = self.build_decoder()
        self.build_mlp_heads(dataset_config, self.dec_dim, self.mlp_dropout)
        self.box_processor = BoxProcessor(dataset_config)
        # there might be no need for this linear layer
        # definitely try also without it
        #self.feature_processor = nn.Linear(256,128)
        #self.feature_processor.cuda()
        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [self.enc_dim]
        else:
            hidden_dims = [self.enc_dim, self.enc_dim]
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=self.enc_dim,
            hidden_dims=hidden_dims,
            output_dim=self.dec_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=self.dec_dim, pos_type=self.position_embedding[0], normalize=True
        )
        self.query_projection = GenericMLP(
            input_dim=self.dec_dim,
            hidden_dims=[self.dec_dim],
            output_dim=self.dec_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features
    
    def build_preencoder(self):
        mlp_dims = [self.input_feature_dim, 64, 128, self.enc_dim]
        #mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
        preencoder = PointnetSAModuleVotes(
            radius=0.2,
            nsample=64,
            #npoint=args.preenc_npoints,
            npoint=2048,
            mlp=mlp_dims,
            normalize_xyz=True,
        )
        return preencoder

    def build_encoder(self, enc_type):
        if enc_type == "vanilla":
            encoder_layer = TransformerEncoderLayer(
                d_model=self.enc_dim,
                nhead=self.enc_nhead,
                dim_feedforward=self.enc_ffn_dim,
                dropout=self.enc_dropout,
                activation=self.enc_activation,
            )
            encoder = TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=self.enc_nlayers
            )
        elif enc_type in ["masked"]:
            encoder_layer = TransformerEncoderLayer(
                d_model=self.enc_dim,
                nhead=self.enc_nhead,
                dim_feedforward=self.enc_ffn_dim,
                dropout=self.enc_dropout,
                activation=self.enc_activation,
            )
            interim_downsampling = PointnetSAModuleVotes(
                radius=0.4,
                nsample=32,
                npoint=2048 // 2,
                mlp=[self.enc_dim, 256, 256, self.enc_dim],
                normalize_xyz=True,
            )
            
            masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
            encoder = MaskedTransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=3,
                interim_downsampling=interim_downsampling,
                masking_radius=masking_radius,
            )
        else:
            raise ValueError(f"Unknown encoder type {enc_type}")
        return encoder
    
    def build_decoder(self):
        decoder_layer = TransformerDecoderLayer(
            d_model=self.dec_dim,
            nhead=self.dec_nhead,
            dim_feedforward=self.dec_ffn_dim,
            dropout=self.dec_dropout,
        )
        decoder = TransformerDecoder(
            decoder_layer, num_layers=self.dec_nlayers, return_intermediate=True
        )
        return decoder
    
    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # add 1 for background/not-an-object class
        semcls_head = mlp_func(output_dim=dataset_config.num_semcls + 1)

        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=dataset_config.num_angle_bin)
        angle_reg_head = mlp_func(output_dim=dataset_config.num_angle_bin)

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = furthest_point_sample(encoder_xyz, self.num_queries[0])
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed
    
    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)
        center_offset = (
            self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
        )
        size_normalized = (
            self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
        )
        angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](
            box_features
        ).transpose(1, 2)

        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
        size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(
            num_layers, batch, num_queries, -1
        )
        angle_residual = angle_residual_normalized * (
            np.pi / angle_residual_normalized.shape[-1]
        )

        outputs = []
        for l in range(num_layers):
            # box processor converts outputs so we can get a 3D bounding box
            (
                center_normalized,
                center_unnormalized,
            ) = self.box_processor.compute_predicted_center(
                center_offset[l], query_xyz, point_cloud_dims
            )
            angle_continuous = self.box_processor.compute_predicted_angle(
                angle_logits[l], angle_residual[l]
            )
            size_unnormalized = self.box_processor.compute_predicted_size(
                size_normalized[l], point_cloud_dims
            )
            
            box_corners = self.box_processor.box_parametrization_to_corners(
                center_unnormalized, size_unnormalized, angle_continuous
            )
            '''
            box_corners = self.box_processor.box_to_corners(center_unnormalized, size_unnormalized, angle_continuous)
            print(box_corners.shape)
            '''
            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                (
                    semcls_prob,
                    objectness_prob,
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])

            box_prediction = {
                "sem_cls_logits": cls_logits[l],
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "size_normalized": size_normalized[l],
                "size_unnormalized": size_unnormalized,
                "angle_logits": angle_logits[l],
                "angle_residual": angle_residual[l],
                "angle_residual_normalized": angle_residual_normalized[l],
                "angle_continuous": angle_continuous,
                "objectness_prob": objectness_prob,
                "sem_cls_prob": semcls_prob,
                "box_corners": box_corners,
                "center_offset": center_offset[l],
            }
            
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }


    def forward(self, data_dict):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            data_dict: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        
        pointcloud = data_dict["point_clouds"]

        batch_size = pointcloud.shape[0]
        
        xyz, features = self._break_up_pc(pointcloud)
        
        #########################
        #####Encoding Step#######
        #########################
        
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)
            # xyz: batch x npoints x 3
            # features: batch x channel x npoints
            # inds: batch x npoints
            
        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )
        if enc_inds is None:
            # encoder does not perform any downsampling
            enc_inds = pre_enc_inds
        else:
            # use gather here to ensure that it works for both FPS and random sampling
            enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.to(torch.int64))
            
        #######################   
        #####Decoding Step#####
        #######################
        
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3

        if self.encoder_only:
            # return: batch x npoints x channels
            return enc_xyz, enc_features.transpose(0, 1)

        point_cloud_dims = [
            data_dict["point_cloud_dims_min"],
            data_dict["point_cloud_dims_max"],
        ]
        
        query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)
        # query_embed: batch x channel x npoint
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # decoder expects: npoints x batch x channel
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        box_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos
        )[0]
        
        box_predictions = self.get_box_predictions(
            query_xyz, point_cloud_dims, box_features
        )
        
        '''
        print(box_pred["sem_cls_logits"].size())
        print(box_pred["center_normalized"].size())
        print(box_pred["center_unnormalized"].size())
        print(box_pred["size_normalized"].size())
        print(box_pred["size_unnormalized"].size())
        print(box_pred["angle_logits"].size())
        print(box_pred["angle_residual"].size())
        print(box_pred["angle_residual_normalized"].size())
        print(box_pred["angle_continuous"].size())
        print(box_pred["objectness_prob"].size())
        print(box_pred["sem_cls_prob"].size())
        print(box_pred["box_corners"].size())
        '''
        # try to adapt output:
        '''
        data_dict['objectness_scores'] = box_pred["objectness_prob"]
        data_dict['center'] = box_pred["center_normalized"]
        data_dict['heading_scores'] = box_pred["angle_logits"]
        data_dict['heading_residuals_normalized'] = box_pred["angle_residual_normalized"]
        data_dict['heading_residuals'] = box_pred["angle_residual"]
        #data_dict['size_scores'] = box_pred[]
        data_dict['size_residuals_normalized'] = box_pred["size_normalized"]
        data_dict['size_residuals'] = box_pred["size_unnormalized"]
        data_dict['sem_cls_scores'] = box_pred["sem_cls_logits"] # maybe use sem_cls_logits
        data_dict['aggregated_vote_xyz'] = box_pred["center_offset"]
        '''
        data_dict['outputs'] = box_predictions['outputs']
        data_dict['aux_outputs'] = box_predictions['aux_outputs']
        
        
        # final decoder layer output
        box_final_features = box_features[-1].clone()
        #box_final_features = torch.zeros(256,2,256).cuda()
        features = box_final_features.transpose(0, 1)
        #data_dict["aggregated_vote_features"] = self.feature_processor(features)
        data_dict["aggregated_features"] = features
        
        # look at box_predictions
        #return box_predictions

        '''
        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        data_dict['sa1_inds'] = fps_inds
        data_dict['sa1_xyz'] = xyz
        data_dict['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        data_dict['sa2_inds'] = fps_inds
        data_dict['sa2_xyz'] = xyz
        data_dict['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        data_dict['sa3_xyz'] = xyz
        data_dict['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        data_dict['sa4_xyz'] = xyz
        data_dict['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(data_dict['sa3_xyz'], data_dict['sa4_xyz'], data_dict['sa3_features'], data_dict['sa4_features'])
        features = self.fp2(data_dict['sa2_xyz'], data_dict['sa3_xyz'], data_dict['sa2_features'], features)
        data_dict['fp2_features'] = features
        data_dict['fp2_xyz'] = data_dict['sa2_xyz']
        num_seed = data_dict['fp2_xyz'].shape[1]
        data_dict['fp2_inds'] = data_dict['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds
        '''
        return data_dict
    

if __name__=='__main__':
    backbone_net = Object_Detection(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(16,20000,6).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
