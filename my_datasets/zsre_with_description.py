import os
import json
import re
import string
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from .utils import MyQADataset, MyDataLoader
from .zsre import ZSREData
from .zsre_relations import ZSRE_RELATIONS

class ZSREWithDescriptionData(ZSREData):

    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer
        postfix = 'Withdescription-' + tokenizer.__class__.__name__.replace("zer", "zed")
        
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(".json", "-{}.json".format(postfix)))
        
        if self.load and os.path.exists(preprocessed_path):
            # load preprocessed input
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
                    metadata = json.load(f)

        else:
            print("Start tokenizing ... {} instances".format(len(self.data)))

            questions = [add_description(d["input"]) for d in self.data]
            if self.data_type != "test":
                answers = [[item["answer"] for item in d["output"]] for d in self.data]
            else:
                answers = [['TEST_NO_ANSWER'] for d in self.data]
                
            answers, metadata = self.flatten(answers)

            if self.args.do_lowercase:
                questions = [question.lower() for question in questions]
                answers = [answer.lower() for answer in answers]
            if self.args.append_another_bos:
                questions = ["<s> "+question for question in questions]
                answers = ["<s> " +answer for answer in answers]

            print(questions[:10])
            print(answers[:10])
            
            print("Tokenizing Input ...")
            question_input = tokenizer.batch_encode_plus(questions,
                                                         pad_to_max_length=True,
                                                         max_length=self.args.max_input_length)
            print("Tokenizing Output ...")
            answer_input = tokenizer.batch_encode_plus(answers,
                                                       pad_to_max_length=True)

            input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            if self.load:
                preprocessed_data = [input_ids, attention_mask,
                                     decoder_input_ids, decoder_attention_mask,
                                     metadata]
                with open(preprocessed_path, "w") as f:
                    json.dump([input_ids, attention_mask,
                               decoder_input_ids, decoder_attention_mask,
                               metadata], f)

        self.dataset = MyQADataset(input_ids, attention_mask,
                                        decoder_input_ids, decoder_attention_mask,
                                        in_metadata=None, out_metadata=metadata,
                                        is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

def add_description(input_str):
    split_idx = input_str.index('[SEP]')
    rel_name = input_str[split_idx+6:]
    description = ZSRE_RELATIONS[rel_name]["description"]
    return "{} [SEP] description: {}".format(input_str, description)

def get_accuracy(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([int(prediction==gt) for gt in groundtruth])
    return int(prediction==gt)


def get_exact_match(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([get_exact_match(prediction, gt) for gt in groundtruth])
    return (normalize_answer(prediction) == normalize_answer(groundtruth))

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))
