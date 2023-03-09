
import wandb
import torch

from transformers import get_linear_schedule_with_warmup
from torch import optim
from tqdm import tqdm

class Trainer:

    def __init__(
        self,
        config,
        train_loader,
        val_loader,
        model,
        metric_func,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.metric_func = metric_func
    
    def train(self):

        if torch.cuda.is_available():
            self.model.cuda()
        
        self.model.train()
        optimizer, scheduler = self.get_optimizer()
        pbar = tqdm(total=self.config.train_steps)
        pbar.set_description('Training steps:')
        step = 0

        for _ in range(self.config.epochs):
            for batch in self.train_loader:
                
                loss, _ = self.model(batch)
                wandb.log({'train/loss': loss}, commit=False)

                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pbar.update(n=1) 

                if step != 0 and step % self.config.steps_show == 0:
                    loss, performance = self.evaluation(self.val_loader, self.metric_func)
                    wandb.log({'eval/loss': loss})
                    wandb.log({'eval/' + k: v for k, v in performance.items()})
                
                step += 1
       # if dev_f1 > best_f1:
    #     best_f1 = dev_f1
    #     torch.save(model, f'{config.saved_model_path}/test.pth')
    #     print('save best model   f1:%.6f'%best_f1) 

    def evaluation(self, val_loader, func):

        pbar = tqdm(total=len(val_loader))
        pbar.set_description('Validation step:')

        self.model.eval()

        y_trues =[]
        y_preds = []
        losses = []

        for batch in val_loader:
            with torch.no_grad():
                loss, y_pred = self.model(batch)
            # y_pred = torch.max(logits.data, 2)[1]
            
            if torch.cuda.is_available():
                y_pred = y_pred.cpu()
                if self.config.use_crf:
                    labels = batch[0]['labels'].cpu()
                else:
                    labels = batch['labels'].cpu()
            else:
                if self.config.use_crf:
                    labels = batch[0]['labels'].cpu()
                else:
                    labels = batch['labels']

            # TODO unify it
            # if not self.config.use_crf:
                # y_preds.extend(y_pred.numpy().tolist())
            y_preds.extend(y_pred)
            y_trues.extend(labels.numpy().tolist())
            losses.append(loss)
            pbar.update(n=1)
            
        performance = func(y_preds, y_trues)
        loss = sum(losses) / len(losses)

        self.model.train()
        return loss, performance


    def get_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias']
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
        optimizer = optim.AdamW(
            optimizer_parameters,
            lr=self.config.lr,

        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.config.warmup_steps * self.config.train_steps),
            num_training_steps=self.config.train_steps
        )

        return optimizer, scheduler