import torch

from tqdm import tqdm
from torch import nn, optim
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

class BertTkModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels
        )

    def get_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 1e-3
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.
            }
        ]
        optimizer = optim.AdamW(optimizer_parameters, lr=self.config.lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.config.warmup_steps * self.config.train_steps),
            num_training_steps=self.config.train_steps
        )

        return optimizer, scheduler
    
    def forward(self, input_dict):
        output = self.model(**input_dict)
        return output['loss'], output['logits']

    def evaluation(self, val_loader, func):
        self.eval()
        y_trues =[]
        y_preds = []
        losses = []
        for batch in tqdm(val_loader):
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                loss, logits = self(batch)
            y_pred = torch.max(logits.data, 2)[1]
            if torch.cuda.is_available():
                y_pred = y_pred.cpu()
                labels = labels.cpu()
            y_preds.extend(y_pred.numpy().tolist())
            y_trues.extend(batch['labels'].numpy().tolist())
            losses.append(loss)
            
        performance = func(y_preds, y_trues)
        loss = sum(losses) / len(losses)

        self.train()
        return loss, performance


class BertClfModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = AutoConfig.from_pretrained(config.model_name)
        model_config.output_hidden_states = True
        self.model = AutoModel.from_pretrained(config.model_name, config=model_config)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(model_config.hidden_size, config.num_labels)
       
        torch.nn.init.normal_(self.classifier.weight, std=0.02)
        
        if torch.cuda.is_available():
            self.cuda()
            
    def get_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 1e-3
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.
            }
        ]
        optimizer = optim.AdamW(optimizer_parameters, lr=self.config.lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.config.warmup_steps * self.config.train_steps),
            num_training_steps=self.config.train_steps
        )

        return optimizer, scheduler

    def evaluation(self, val_loader):
        self.eval()
        y_trues =[]
        y_preds = []
        for batch in val_loader:
            input_dict, labels = batch
            if torch.cuda.is_available():
                input_dict = {k: v.cuda() for k, v in input_dict.items()}
                labels = labels.cuda()
            with torch.no_grad():
                logits, _ = self(input_dict)
            y_pred = torch.max(logits.data, 1)[1].cpu()
            if torch.cuda.is_available():
                labels = labels.cpu()
            y_preds.extend(y_pred)
            y_trues.extend(labels)
            
        f1 = f1_score(y_trues, y_preds, average='macro')
        acc = accuracy_score(y_trues, y_preds)
        rec = recall_score(y_trues, y_preds, average='macro')

        self.train()
        return f1, acc, rec
    
    def forward(self, input_dict):
        pooled_output = self.model(**input_dict)['last_hidden_state']
        pooled_output = self.dropout(pooled_output[:,0,:])

        start_logits = self.classifier(pooled_output)
        return start_logits, pooled_output