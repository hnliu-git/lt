
from tqdm import tqdm
from transformers import AutoTokenizer
from model_utils.model import BertTkModel
from model_utils.dataset import TKDataset
from metric_utils.data_utils import TagDict
from metric_utils.metrics import initialize_metrics
from model_utils.utils import vac_tags_dict, vac_main_dict, tokenize_and_align_labels, tokenize_and_align_labels_and_chunk

import torch
import wandb
import datasets

def compute_metrics(predictions, labels):
    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metrics.compute(predictions=true_predictions, gold=true_labels)

    return results


class GlobalConfig:   
    def __init__(self):
        # general setting
        self.seed = 2022
        # model setting
        self.model_name = 'xlm-roberta-base'
        self.num_labels = len(vac_tags_dict)
        # data setting
        self.max_length = 512
        self.batch_size = 32
        # training setting
        self.epochs = 10
        self.steps_show = 100
        self.warmup_steps = 0.1
        self.lr = 5e-6
        self.saved_model_path = 'saved_models'

data_folder = 'data/'
config = GlobalConfig()
exp_name = '%s-nl-epoch%d'%(config.model_name, config.epochs)

data_files = {
    'train': data_folder + 'train.jsonl',
    'validation': data_folder + 'devel.jsonl',
    'test': data_folder + 'test.jsonl'
}

dataset = datasets.load_dataset('json', data_files=data_files)

print(f'train: {len(dataset["train"])}')
print(f'eval: {len(dataset["validation"])}')
print(f'test: {len(dataset["test"])}')

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
# tokenized_dataset.cleanup_cache_files()

metrics = initialize_metrics(
    metric_names=['accuracy', 'entity_score', 'entity_overlap'],
    tags=TagDict.from_file('metric_utils/vac-phrases-full-tags.txt'),
    main_entities=open('metric_utils/main_ents.txt').read().splitlines()
)

train_loader = TKDataset(tokenizer, tokenized_dataset, 'train', config).build_dataloader()
val_loader = TKDataset(tokenizer, tokenized_dataset, 'validation', config).build_dataloader()
test_loader = TKDataset(tokenizer, tokenized_dataset, 'test', config).build_dataloader()

config.train_steps = len(train_loader) * config.epochs
# config.steps_show = int(len(train_loader) * 0.3)

model = BertTkModel(config)
optimizer, scheduler = model.get_optimizer()

steps = 0
best_f1 = 0
model.train()

wandb.init(project='bert_vac_ner', name=exp_name)

for epoch in range(1, config.epochs + 1):
    for batch in tqdm(train_loader):
        # if torch.cuda.is_available():
        #     input_dict = {k: v.cuda() for k, v in input_dict.items()}
        #     labels = labels.cuda()
            
        loss, logits = model(batch)

        wandb.log({'train/loss': loss}, )

        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        preds = torch.max(logits.data, 1)[1].cpu()
        if torch.cuda.is_available():
            labels = labels.cpu()
            
        if steps % config.steps_show == 0:
            loss, performance = model.evaluation(val_loader, compute_metrics)
            wandb.log({'eval/loss': loss})
            wandb.log({'eval/' + k: v for k, v in performance.items()})

        steps += 1
            
    # if dev_f1 > best_f1:
    #     best_f1 = dev_f1
    #     torch.save(model, f'{config.saved_model_path}/test.pth')
    #     print('save best model   f1:%.6f'%best_f1)

