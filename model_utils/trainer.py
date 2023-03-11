
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

        self.model.train()
        if self.config.use_crf:
            optimizer, optimizer_crf, scheduler, scheduler_crf = self.get_crf_optimizer()
        else:
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
                
                if self.config.use_crf: 
                    optimizer_crf.step()
                    scheduler_crf.step()
                    optimizer_crf.zero_grad()

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
            
            if torch.cuda.is_available():
                if self.config.use_crf:
                    labels = batch[0]['labels'].cpu()
                else:
                    labels = batch['labels'].cpu()
            else:
                if self.config.use_crf:
                    labels = batch[0]['labels']
                else:
                    labels = batch['labels']

            labels = labels.numpy().tolist()

            y_preds.extend(y_pred)
            y_trues.extend(labels)
            losses.append(loss)
            pbar.update(n=1)
            
        performance = func(y_preds, y_trues)
        loss = sum(losses) / len(losses)

        self.model.train()
        return loss, performance

    def get_crf_optimizer(self):
        model_params = list(self.model.model.named_parameters())
        no_decay = ['bias']
        optimizer_parameters = [
            {
                'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
                'weight_decay': 1e-3
            },
            {
                'params': [p for n, p in model_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.
            }
        ]
        optimizer_model = optim.AdamW(
            optimizer_parameters,
            lr=self.config.lr,
        )
        crf_params = list(self.model.crf.named_parameters())
        crf_optimizer_parameters = [
            {
                'params': [p for n, p in crf_params],
                'weight_decay': 1e-3
            }
        ] 
        optimizer_crf = optim.AdamW(
            crf_optimizer_parameters,
            lr=self.config.lr*100,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer_model,
            num_warmup_steps=int(self.config.warmup_steps * self.config.train_steps),
            num_training_steps=self.config.train_steps
        )
        scheduler_crf = get_linear_schedule_with_warmup(
            optimizer_crf,
            num_warmup_steps=int(self.config.warmup_steps * self.config.train_steps),
            num_training_steps=self.config.train_steps
        )
        
        return optimizer_model, optimizer_crf, scheduler, scheduler_crf 


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
    

class ChunkTrainer:

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
        self.config.train_steps = sum([len(loader)*num for num, loader in self.train_loader])*self.config.epochs
        self.model.train()
        optimizer, scheduler = self.get_optimizer()

        pbar = tqdm(total=self.config.train_steps)
        pbar.set_description('Training steps:')

        for _ in range(self.config.epochs):

            for num_chunk, loader in self.train_loader:
                for batches in loader:
                    for i in range(num_chunk):
                        if i == 0:
                            loss, _, ctx = self.model(batches[i])
                        else:
                            loss, _, ctx = self.model(batches[i], ctx)

                        wandb.log({'train/loss': loss}, commit=False)

                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        if pbar.n != 0 and pbar.n % self.config.steps_show == 0:
                            loss, performance = self.evaluation(self.val_loader, self.metric_func)
                            wandb.log({'eval/loss': loss})
                            wandb.log({'eval/' + k: v for k, v in performance.items()})
                        pbar.update(n=1) 

       # if dev_f1 > best_f1:
    #     best_f1 = dev_f1
    #     torch.save(model, f'{config.saved_model_path}/test.pth')
    #     print('save best model   f1:%.6f'%best_f1) 

    def evaluation(self, val_loader, func):

        pbar = tqdm(total=sum([len(loader)*num for num, loader in val_loader]))
        pbar.set_description('Validation step:')

        self.model.eval()

        y_trues =[]
        y_preds = []
        losses = []

        for num_chunk, loader in val_loader:
            for batches in loader:
                for i in range(num_chunk):
                    if i == 0:
                        with torch.no_grad():
                            loss, y_pred, ctx = self.model(batches[i])
                    else:
                        with torch.no_grad():
                            loss, y_pred, ctx = self.model(batches[i], ctx)
                    
                    if torch.cuda.is_available():
                        labels = batches[i]['labels'].cpu().numpy().tolist()
                    else:
                        labels = batches[i]['labels'].numpy().tolist()

                    y_preds.extend(y_pred)
                    y_trues.extend(labels)
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