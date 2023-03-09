import torch
import pandas as pd

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, IterableDataset

class TKDataset(Dataset):

    def __init__(self, tokenizer, dataset, split, config) -> None:
        """
        dataset: Huggingface Dataset
        """
        self.label_pad_token_id = -100
        self.dataset = dataset[split]
        self.tokenizer = tokenizer
        self.split = split
        self.config = config
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]

    def collate_fn(self, features):

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        features = [{k: v for k, v in feature.items() if k in ['input_ids', 'attention_mask']} for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=True,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        batch[label_name] = [
            list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
        ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}

        if self.config.use_crf:
            crf_mask = []
            for feature in features:
                mask = feature['attention_mask']
                mask[0] = 0
                mask[-1] = 0
                crf_mask.append(mask + [0] * (sequence_length - len(mask)))
            crf_mask = torch.tensor(crf_mask, dtype=torch.bool)

            return batch, crf_mask

        return batch

    def build_dataloader(self):
        dataloader = DataLoader(
            self,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            shuffle=self.split == 'train'
        )
        
        return dataloader


class TKChunkDataset(IterableDataset):

    def __init__(self, dataset, split) -> None:
        self.dataset = dataset[split]

    def __len__(self):
        return sum(self.dataset['chunk_num'])

    def __iter__(self):
        for doc in self.dataset:
            for i in range(doc['chunk_num']):
                yield {
                    k: doc[k][i]
                    for k in ['input_ids', 'attention_mask', 'labels']
                }

class DFDataset(Dataset):
    
    def __init__(self, df, config, is_test):

        if type(df) == str:
            self.df = pd.read_csv(df, index_col=0)
        else:
            self.df = df
        
        self.config = config
        self.is_test = is_test
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def collate_fn(self, batch):
        texts = [item[self.config.train_col] for item in batch]
        encoded_text = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            max_length=self.config.max_length
        )
        
        if self.is_test:
            return encoded_text
        else:
            type2label = {2:1, 1:0}
            labels = [type2label[item[self.config.label_col]] for item in batch]
            labels = torch.LongTensor(labels)
            return encoded_text, labels
    
    def build_dataloader(self):
        dataloader = DataLoader(
            self,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            shuffle=not self.is_test
        )
        
        return dataloader
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, ix):
        return self.df.iloc[ix]
