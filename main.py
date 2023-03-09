# TODO
# - Trainer Class DONE!
# - Add CRF Layer 
# - The idea


from tqdm import tqdm
from transformers import AutoTokenizer
from model_utils.model import BertTkModel, BertCRFTkModel
from model_utils.trainer import Trainer
from model_utils.dataset import TKDataset
from metric_utils.data_utils import TagDict
from metric_utils.metrics import initialize_metrics
from model_utils.utils import vac_tags_dict, vac_main_dict, tokenize_and_align_labels, tokenize_and_align_labels_and_chunk

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
        self.model_name = 'xlm-roberta-large'
        self.num_labels = len(vac_tags_dict)
        # data setting
        self.max_length = 512
        self.batch_size = 16
        # training setting
        self.epochs = 20
        self.steps_show = 100
        self.warmup_steps = 0.003
        self.lr = 2e-5
        self.saved_model_path = 'saved_models'
        self.use_crf = True

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

model = BertCRFTkModel(config)

wandb.init(project='bert_vac_ner', name=exp_name)

trainer = Trainer(
    config,
    train_loader,
    val_loader,
    model,
    compute_metrics
)

trainer.train()

