import torch
import torch.nn as nn
import random
import numpy as np
from utils.box_util import box3d_iou_batch

class VTransMatchModule(nn.Module):
    def __init__(self, args, num_proposals=256, lang_size=256, hidden_size=128):
        super().__init__() 

        self.args = args
        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size
        self.num_encoder_layers = args.m_enc_layers
        
        #self.vt_fuse = nn.Transformer(d_model=self.lang_size+128, nhead=8, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=256, dropout=0.1, custom_encoder=None, custom_decoder=self.decoder)
        self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.lang_size+128,
                nhead=8,
                dim_feedforward=1028,
                dropout=0.1,
                #activation=self.enc_activation,
            )
        self.vt_fuse = nn.TransformerEncoder(self.encoder_layer, self.num_encoder_layers)
        # try without dim reduction and increase hidden size to self.lang_size + 128
        self.reduce = nn.Sequential(
            nn.Conv1d(self.lang_size + 128, hidden_size, 1),
            nn.ReLU()
        )
        # self.match = nn.Conv1d(hidden_size, 1, 1)
        self.match = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, 1)
        )

    def forward(self, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        # unpack outputs from detection branch
        features = data_dict['aggregated_vote_features'] # batch_size, num_proposal, 128
        if self.args.detection_module == "3detr":
            objectness_masks = torch.as_tensor((data_dict['outputs']["objectness_prob"].unsqueeze(-1))>0.5,dtype=torch.float32)
        elif self.args.detection_module == "votenet":
            objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2) # batch_size, num_proposals, 1

        # unpack outputs from language branch
        lang_feat = data_dict["lang_emb"] # batch_size * len_nun_max, lang_size
        
        #for key in data_dict:
        #    print(f'{key}: {data_dict[key]}')

        if self.args.use_chunking:
            data_dict["random"] = random.random()
            batchsize, len_nun_max = data_dict['ref_center_label_list'].shape[:2]
            # print(f'batchsize, len_nun_max: {batchsize}, {len_nun_max}')
            features = features.unsqueeze(1).repeat(1, len_nun_max, 1, 1)
            v1, v2, v3, v4 = features.shape[:4]
            #print(f'v1: {v1}, v2: {v2}, v3: {v3}, v4: {v4}') # batchsize, len_nun
            features = features.reshape(batchsize * len_nun_max, v3, v4)
            #print(f'feature shape after unsquezze: {features.shape}')
            #print(f'objectness_masks shape before unsquezze: {objectness_masks.shape}')
            objectness_masks = objectness_masks.unsqueeze(1).repeat(1, len_nun_max, 1, 1).reshape(batchsize * len_nun_max, v3, 1)
            #print(f'objectness_masks shape after unsquezze: {objectness_masks.shape}')
        else:
            batchsize = data_dict['ref_center_label'].shape[0]

        #print(f'lang_feat shape: {lang_feat.shape}')
        lang_feat = lang_feat.unsqueeze(1).repeat(1, self.num_proposals, 1) # batch_size, num_proposals, lang_size
        #print(f'lang_feat shape after unsquezze: {lang_feat.shape}')

        # fuse
        features = torch.cat([features, lang_feat], dim=-1) # batch_size, num_proposals, 128 + lang_size
        features = features.permute(1, 0, 2).contiguous() # num_proposals, batch_size, 128 + lang_size
        # fuse features
        # big question what to use as tgt? -> maybe find gt bbox and best matching proposal bbox as tgt
        # requires to change TransformerDecoderLayer
        '''
        if 'lang_feat_list' in data_dict or 'lang_inputs_list' in data_dict:
            gt_ref = torch.argmax(data_dict["ref_box_label_list"], -1)
        else:
            gt_ref = torch.argmax(data_dict["ref_box_label"], 1)
        gt_labels = np.zeros((batchsize, len_nun_max, self.num_proposals))
        lang_num = data_dict["lang_num"]
        for i in range(batchsize):
            for j in range(len_nun_max):
                if j < lang_num[i]:
                    labels = np.zeros((len_nun_max, self.num_proposals))
                    gt = data_dict['gt_box_corners'][i][gt_ref[i,j]].detach().cpu().numpy()
                    #gt_bbox_batch.detach().cpu().numpy()
                    pred_bbox_batch = data_dict['outputs']['box_corners'][i]
                    pred_bbox_batch = pred_bbox_batch.detach().cpu().numpy()
                    ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt, (self.num_proposals, 1, 1)))
                    # clustering ious should match normal ious
                    labels[j, ious.argmax()] = 1
            gt_labels[i] = labels
        print(gt_labels.shape)
        '''
        features = self.vt_fuse(features) # num_proposals, batch_size, lang_size + 128
        features = features.permute(1, 2, 0).contiguous() # batch_size, lang_size +128, num_proposals
        features = self.reduce(features) # batch_size, hidden_size, num_proposals
        # mask out invalid proposals
        objectness_masks = objectness_masks.permute(0, 2, 1).contiguous() # batch_size, 1, num_proposals
        features = features * objectness_masks

        # match
        confidences = self.match(features).squeeze(1) # batch_size, num_proposals
                
        data_dict["cluster_ref"] = confidences

        return data_dict
