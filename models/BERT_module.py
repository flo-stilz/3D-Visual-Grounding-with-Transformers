import os
import sys
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, DistilBertModel, AutoModel, AutoConfig, AutoModelForSequenceClassification, DistilBertConfig
# mvt
from transformers import BertTokenizer, BertModel, BertConfig
#from transformers import AutoTokenizerFast

class BERTModule(nn.Module):
    def __init__(self, num_text_classes, use_lang_classifier=True, hidden_size=256, chunking=False):
        super().__init__() 

        self.num_text_classes = num_text_classes
        self.use_lang_classifier = use_lang_classifier
        self.chunking = chunking
        
        '''
        #set Bert
        configuration = AutoConfig.from_pretrained('distilbert-base-cased')

        # configuration = DistilBertConfig()
        # For BERT
        #configuration.hidden_dropout_prob = hparams["b_drop"]
        #configuration.attention_probs_dropout_prob = hparams["b_drop"]
        # For DistilBERT
        configuration.dropout = 0.1
        configuration.attention_dropout = 0.1
        
        DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.bert = AutoModel.from_pretrained('distilbert-base-cased', config = configuration)
        '''

        self.language_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.language_encoder.encoder.layer = BertModel(BertConfig()).encoder.layer[:3]

        
        
        '''
        # freeze some of the BERT weights:
        modules = [self.bert.embeddings, *self.bert.transformer.layer[:4]] 
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        
        self.MLP = nn.Linear(768,hidden_size)
        '''

        self.MLP = nn.Sequential(
            nn.Linear(768,512),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, hidden_size)
        )

        #lang_size = hidden_size * 2 if self.use_bidir else hidden_size
        lang_size = hidden_size

        # language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(lang_size, num_text_classes),
                nn.Dropout()
            )


    def forward(self, data_dict):
        """
        encode the input descriptions
        """

        if self.chunking:
            lang_inputs_list = data_dict["lang_inputs_list"]
            lang_mask_list = data_dict["lang_mask_list"]

            batch_size, len_nun_max, max_des_len = lang_inputs_list.shape[:3]

            lang_inputs_list = lang_inputs_list.reshape(batch_size * len_nun_max, max_des_len)
            lang_mask_list = lang_mask_list.reshape(batch_size * len_nun_max, max_des_len)

            # pooled_output = self.bert(input_ids=lang_inputs_list, attention_mask=lang_mask_list,return_dict=False)
            pooled_output = self.language_encoder(input_ids=lang_inputs_list, attention_mask=lang_mask_list,return_dict=False)
            lang_last = pooled_output[0]
            # output of CLS Token
            lang_last = lang_last[:,0]
            lang_last = self.MLP(lang_last)

            # store the encoded language features
            data_dict["lang_emb"] = lang_last # B, hidden_size

            # classify
            if self.use_lang_classifier:
                data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"])


        else:
            #lang_feat = data_dict['lang_feat']
            lang_input = data_dict['lang_inputs']
            lang_mask = data_dict['lang_mask']

            # feed x into encoder!
            # For BERT
            #_, pooled_output = self.bert(input_ids=input_ids, attention_mask=mask,return_dict=False)
            # For DistilBERT

            # pooled_output = self.bert(input_ids=lang_input, attention_mask=lang_mask,return_dict=False)
            pooled_output = self.language_encoder(input_ids=lang_input, attention_mask=lang_mask,return_dict=False)
            lang_last = pooled_output[0]
            # output of CLS Token
            lang_last = lang_last[:,0]
            lang_last = self.MLP(lang_last)

            # store the encoded language features
            data_dict["lang_emb"] = lang_last # B, hidden_size
            
            # classify
            if self.use_lang_classifier:
                data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"])

        return data_dict

