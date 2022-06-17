import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.lang_module import LangModule
from models.match_module import MatchModule
from models.dvg_match_module import DVGMatchModule
from models.Object_Detection import Object_Detection
from models.BERT_module import BERTModule

class RefNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, args,
    input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
    use_lang_classifier=True, use_bidir=False, no_reference=False, chunking = False,
    emb_size=300, hidden_size=256, lang_module='gru'):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir      
        self.no_reference = no_reference
        self.chunking = chunking
        self.lang_module = lang_module
        self.args = args
        
        # --------- Object Detection ------------
        # self.Object_Detection = Object_Detection(input_feature_dim=self.input_feature_dim)
        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)
        
        if not no_reference:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)

            if self.args.lang_module == 'gru':
                self.lang_encoder = LangModule(num_class, use_lang_classifier, use_bidir, emb_size, hidden_size, self.chunking)
            elif self.args.lang_module == 'bert':
                self.lang_encoder = BERTModule(num_class, use_lang_classifier, hidden_size, self.chunking)
            else:
                AssertionError

            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            if self.args.match_module == 'scanrefer':
                self.match = MatchModule(args=self.args, num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * hidden_size)
            elif self.args.match_module == 'dvg':
                self.lang_encoder = DVGMatchModule(args=self.args,  num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * hidden_size)
            else:
                AssertionError
            # self.match = MatchModule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * hidden_size)

    def forward(self, data_dict):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################
        
        # --------- 3DETR ----------------
        #data_dict = self.Object_Detection(data_dict)
        
        # --------- HOUGH VOTING ---------
        data_dict = self.backbone_net(data_dict)
                
        # --------- HOUGH VOTING ---------
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"]
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz
        data_dict["seed_features"] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features

        # --------- PROPOSAL GENERATION ---------
        data_dict = self.proposal(xyz, features, data_dict)
        
        
        if not self.no_reference:
            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################

            # --------- LANGUAGE ENCODING ---------
            data_dict = self.lang_encoder(data_dict)

            #######################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################

            # --------- PROPOSAL MATCHING ---------
            data_dict = self.match(data_dict)

        return data_dict
