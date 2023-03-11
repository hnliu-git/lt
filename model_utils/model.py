
import torch

from torchcrf import CRF
from torch import nn 
from transformers import AutoModelForTokenClassification
from transformers import XLMRobertaModel
class BertTkModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels
        )
        if torch.cuda.is_available():
            self.cuda()
    
    def forward(self, input_dict):
        if torch.cuda.is_available():
            input_dict = {k: v.cuda() for k, v in input_dict.items()}

        output = self.model(**input_dict)
        tags = torch.max(output['logits'].data, 2)[1]

        if torch.cuda.is_available():
            tags = tags.cuda()

        return output['loss'], tags.numpy().tolist()

class BertCRFTkModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels
        )
        self.crf = CRF(config.num_labels, batch_first=True) 

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, batch):
        input_dict, crf_mask = batch
        if torch.cuda.is_available():
            input_dict = {k: v.cuda() for k, v in input_dict.items()}
            crf_mask = crf_mask.cuda()

        output = self.model(**input_dict)

        # Step forward to pass [CLS] labels * masks can remove all -100
        emissions = output['logits'][:, 1:, :]
        labels = (input_dict['labels'] * crf_mask)[:, 1:]
        masks = crf_mask[:, 1:]

        crf_loss = self.crf(emissions, labels, masks)
        tags = [[0] + tag for tag in self.crf.decode(emissions)]

        return -crf_loss, tags


class BertChunkTkModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = XLMRobertaModel.from_pretrained(
            config.model_name
        )
        self.embedding = self.model.get_input_embeddings()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, config.num_labels)

        self.classifier.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
    
    def forward(self, input_dict, ctx_embed=None):
        if torch.cuda.is_available():
            input_dict = {k: v.cuda() for k, v in input_dict.items()}

        if ctx_embed is None:
            inputs_embeds = self.embedding(input_dict['input_ids'])
        else:
             # It might be better to detach the CLS embedding
            inputs_embeds = torch.cat([
                ctx_embed,
                self.embedding(input_dict['input_ids'][:, 1:])
            ], dim=1)

        outputs = self.model(
            attention_mask=input_dict['attention_mask'],
            inputs_embeds=inputs_embeds
        )

        last_encoder_layer = outputs[0]
        logits = self.classifier(self.dropout(last_encoder_layer))

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.num_labels), input_dict['labels'].view(-1))
        tags = torch.max(logits, 2)[1]

        if torch.cuda.is_available():
            tags = tags.cuda()

        return loss, tags.numpy().tolist(), outputs[0][:, 0:1, :]