import tqdm
import torch

from torch import nn, optim
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification

class BertTkModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels
        )
    
    def forward(self, input_dict):
        output = self.model(**input_dict)
        return output['loss'], output['logits']