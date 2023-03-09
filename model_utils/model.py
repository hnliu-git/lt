
import torch

from torchcrf import CRF
from torch import nn 
from transformers import AutoModelForTokenClassification

class BertTkModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels
        )
    
    def forward(self, input_dict):
        if torch.cuda.is_available():
            input_dict = {k: v.cuda() for k, v in input_dict.items()}

        output = self.model(**input_dict)
        return output['loss'], output['logits']

class BertCRFTkModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels
        )
        self.crf = CRF(config.num_labels, batch_first=True) 

    def forward(self, batch):
        input_dict, crf_mask = batch
        if torch.cuda.is_available():
            input_dict = {k: v.cuda() for k, v in input_dict.items()}

        output = self.model(**input_dict)

        # Step forward to pass [CLS] labels * masks can remove all -100
        emissions = output['logits'][:, 1:, :]
        labels = (input_dict['labels'] * crf_mask)[:, 1:]
        masks = crf_mask[:, 1:]

        crf_loss = self.crf(emissions, labels, masks)
        tags = [[0] + tag for tag in self.crf.decode(emissions)]

        return -crf_loss, tags