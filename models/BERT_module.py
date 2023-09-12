import torch.nn as nn
from argparse import Namespace

from transformers import (
    BertModel, 
    BertConfig,
    DistilBertModel, 
    DistilBertConfig,
)

class BERTModule(nn.Module):
    def __init__(
            self, 
            args: Namespace, 
            num_text_classes: int, 
            use_lang_classifier: bool = True, 
            hidden_size: int = 256, 
            chunking: bool = False
        ) -> None:
        """
        Args:
            - args: config file
            - num_text_classes: number of text classes
            - use_lang_classifier: whether the language classifier is getting trained
            - hidden_size: size of hidden layer 
            - chunking: whether the language input is chunked
        """
        super().__init__() 

        self.args = args
        self.num_text_classes = num_text_classes
        self.use_lang_classifier = use_lang_classifier
        self.chunking = chunking
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.encoder.layer = BertModel(BertConfig()).encoder.layer[:self.args.num_bert_layers]

        if self.args.match_module == 'dvg':
            # dvg uses a specific hidden size
            hidden_size = 128
        
        # take CLS token of Bert and create language embeddings with MLP 
        self.MLP = nn.Sequential(
            nn.Linear(768,512),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, hidden_size)
        )
        
        # language classifier
        if self.use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(hidden_size, num_text_classes),
                nn.Dropout()
            )


    def forward(self, data_dict: dict) -> dict:
        """
        Encode language input descriptions and return the input dictionary with the added language embeddings.
        Additionally, if language classification is enabled, the dictionary contains language scores.

        Args:
        - data_dict (dict): A dictionary containing input data, including language inputs and masks.

        Returns:
        - dict: Modified input dictionary containing the following keys:
            - 'lang_emb' (tensor): Encoded language embeddings.
            - 'lang_scores' (tensor, optional): Language classification scores if enabled.
            - 'attention_mask' (None or tensor): Attention mask for DVG fusion module, if applicable.
        """

        if self.args.use_chunking:
            # if we use chunking each batch element is a list of descriptions
            lang_inputs_list = data_dict["lang_inputs_list"]
            lang_mask_list = data_dict["lang_mask_list"]

            # reshape to batch_size * len_nun_max, max_des_len
            # can be applied since all chunks have the same size
            batch_size, len_nun_max, max_des_len = lang_inputs_list.shape[:3]
            lang_input = lang_inputs_list.reshape(batch_size * len_nun_max, max_des_len) 
            lang_mask = lang_mask_list.reshape(batch_size * len_nun_max, max_des_len) 
        else:
            lang_input = data_dict['lang_inputs']
            lang_mask = data_dict['lang_mask']
     
        # encode the language features with Bert
        bert_output = self.bert(
            input_ids=lang_input, 
            attention_mask=lang_mask,
            return_dict=False
        )
        lang_last = bert_output[0] # last_hidden_state, pooler_output
        # output of CLS Token
        lang_last = lang_last[:,0]
        #
        lang_emb = self.MLP(lang_last)

        # store the encoded language features
        data_dict["lang_emb"] = lang_emb # B (* chunk_size), hidden_size
            
        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"])

        # DVG fusion module
        if self.args.match_module == 'dvg':
            data_dict["attention_mask"] = None

        return data_dict

