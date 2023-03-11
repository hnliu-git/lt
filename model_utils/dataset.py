import torch

from datasets import Dataset as HgDataset
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

class TKChunkData(Dataset):

    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]


class TKChunkDataset:

    def __init__(self, tokenizer, dataset, split, config) -> None:
        self.dataset = dataset[split]
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.label_pad_token_id = -100
    
    def build_dataloaders(self):
        """
        return a list of dataloaders, each loader has docs with the same chunk num
        """
        num2sets = {}

        for doc in self.dataset:
            if doc['chunk_num'] not in num2sets:
                num2sets[doc['chunk_num']] = []
            # TODO Limit the length of doc to avoid short loader
            num2sets[doc['chunk_num']].append(doc)

        max_num_chunk = max(num2sets.keys())
        print("Max chunk num is", max_num_chunk)

        dataloaders = []
        for num in num2sets.keys():
            dataset = HgDataset.from_list(num2sets[num])
            dataloaders.append((num, DataLoader(
                TKChunkData(dataset),
                batch_size=self.config.batch_size,
                collate_fn=self.collate_fn,
                shuffle=self.split == 'train'
            )))
        
        return dataloaders
    
    def collate_fn(self, features):
        
        num_chunk = features[0]['chunk_num']
        batches = []

        for i in range(num_chunk):
            labels_chunk = [feature['labels'][i] for feature in features]
            features_chunk = [{k: v[i] for k, v in feature.items() if k in ['input_ids', 'attention_mask']} for feature in features] 

            batch = self.tokenizer.pad(
                features_chunk,
                padding=True,
                # Conversion to tensors will fail if we have labels as they are not of the same length yet.
                return_tensors=None,
            )

            sequence_length = torch.tensor(batch["input_ids"]).shape[1]
            batch['labels'] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels_chunk
            ]

            batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
            batches.append(batch)

        return batches 
    
# class TKChunkDataset(IterableDataset):

#     def __init__(self, dataset, split) -> None:
#         self.dataset = dataset[split]

#     def __len__(self):
#         return sum(self.dataset['chunk_num'])

#     def __iter__(self):
#         for doc in self.dataset:
#             for i in range(doc['chunk_num']):
#                 yield {
#                     k: doc[k][i]
#                     for k in ['input_ids', 'attention_mask', 'labels']
#                 }

#     def build_dataloaders(self):
        
#         pass

