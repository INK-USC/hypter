import os
import json
import re
import string
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

class MyQADataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 in_metadata=None, out_metadata=None,
                 is_training=False):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]

class MyDataLoader(DataLoader):

    def __init__(self, args, dataset, is_training):
        if is_training:
            sampler=RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler=SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(MyDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)

class MyGroupedQADataset(Dataset):
    def __init__(self,
                 relation_ids, relation_mask, input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 metadata_rel, metadata_questions,
                 is_training=False):
        self.relation_ids = torch.LongTensor(relation_ids)
        self.relation_mask = torch.LongTensor(relation_mask)
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        self.metadata_rel = metadata_rel
        self.metadata_questions = metadata_questions
        self.is_training = is_training

        assert len(self.input_ids)==len(self.attention_mask)==self.metadata_rel[-1][-1]
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.metadata_questions[-1][-1]

    def __len__(self):
        return len(self.metadata_rel)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx]

        rel_ids = self.relation_ids[idx]
        rel_masks = self.relation_mask[idx]

        in_indices = np.random.choice(range(*self.metadata_rel[idx]), 4, replace=False)
        # print(in_indices)

        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = [], [], [], []
        for in_index in in_indices:
            input_ids.append(self.input_ids[in_index])
            attention_mask.append(self.attention_mask[in_index])

            out_idx = np.random.choice(range(*self.metadata_questions[in_index]))
            # print(out_idx)
            # print(self.decoder_input_ids[out_idx])

            # assert out_idx < len(self.decoder_input_ids)
            decoder_input_ids.append(self.decoder_input_ids[out_idx])
            decoder_attention_mask.append(self.decoder_attention_mask[out_idx])

        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        decoder_input_ids = torch.stack(decoder_input_ids)
        decoder_attention_mask = torch.stack(decoder_attention_mask)

        return rel_ids, rel_masks, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask

class MyGroupedDataLoader(DataLoader):

    def __init__(self, args, dataset, is_training):
        if is_training:
            sampler=RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler=SequentialSampler(dataset)
            batch_size = args.predict_batch_size

        super(MyGroupedDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)
        self.collate_fn = self.dummy_collate

    def dummy_collate(self, input_data):
        return input_data