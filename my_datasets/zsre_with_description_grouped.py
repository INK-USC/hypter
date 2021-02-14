import os
import json
import re
import string
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from .utils import MyGroupedQADataset, MyGroupedDataLoader
from .zsre_grouped import ZSREGroupedData
from .zsre_relations import ZSRE_RELATIONS

class ZSREWithDescriptionGroupedData(ZSREGroupedData):

    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer
        postfix = 'Withdescription-Grouped-' + tokenizer.__class__.__name__.replace("zer", "zed")
        
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(".json", "-{}.json".format(postfix)))
        
        if self.load and os.path.exists(preprocessed_path):
            # load preprocessed input
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                relation_ids, relation_mask, input_ids, attention_mask, \
                    decoder_input_ids, decoder_attention_mask, \
                    metadata_rel, metadata_questions, self.raw_questions, self.raw_answers, self.raw_ids = json.load(f)


        else:
            print("Start tokenizing ... {} instances".format(len(self.data)))

            relations = sorted(list(set([d['input'][d['input'].index("[SEP]")+6:] for d in self.data])))
            raw_data = [[] for _ in range(len(relations))]
            id2relation = {k: v for k,v in enumerate(relations)}
            relation2id = {v: k for k,v in enumerate(relations)}
            relations = [add_description(item, raw=True) for item in relations]

            print("relation2id: {}".format(relation2id))

            self.raw_questions = []
            self.raw_answers = []
            self.raw_ids = []

            for d in self.data:
                rel = d['input'][d['input'].index("[SEP]")+6:]
                rel_id = relation2id[rel]
                if self.data_type != 'test':
                    raw_data[rel_id].append((add_description(d["input"]), [item["answer"] for item in d["output"]], d["id"]))
                else:
                    raw_data[rel_id].append((add_description(d["input"]), ['TEST_NO_ANSWER'], d["id"]))

            # qas are sorted according to relations
            for one_rel_data in raw_data:
                self.raw_questions += [item[0] for item in one_rel_data]
                self.raw_answers += [item[1] for item in one_rel_data]
                self.raw_ids += [item[2] for item in one_rel_data]

            print(relations[:10])
            print(self.raw_questions[:10])
            print(self.raw_answers[:10])
            print(self.raw_ids[:10])
            
            # metadata_rel[idx] = (st, ed); it means questions[st: ed] are examples of relation idx;
            # metadata_questions[idx] = (st, ed); it means answers[st: ed] are examples of quesiton idx.
            questions, answers, metadata_rel, metadata_questions = self.flatten(raw_data)

            # if self.args.do_lowercase:
            #     questions = [question.lower() for question in questions]
            #     answers = [answer.lower() for answer in answers]
            # if self.args.append_another_bos:
            #     questions = ["<s> "+question for question in questions]
            #     answers = ["<s> " +answer for answer in answers]
            print("Tokenizing Relations ...")
            relation_input = tokenizer.batch_encode_plus(relations,
                                                         pad_to_max_length=True)
            
            print("Tokenizing Questions ...")
            question_input = tokenizer.batch_encode_plus(questions,
                                                         pad_to_max_length=True,
                                                         max_length=self.args.max_input_length)
            print("Tokenizing Answers ...")
            answer_input = tokenizer.batch_encode_plus(answers,
                                                       pad_to_max_length=True)

            relation_ids, relation_mask = relation_input["input_ids"], relation_input["attention_mask"]
            input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            if self.load:
                # preprocessed_data = [relation_ids, relation_mask, input_ids, attention_mask,
                #                      decoder_input_ids, decoder_attention_mask,
                #                      metadata_rel, metadata_questions]

                with open(preprocessed_path, "w") as f:
                    json.dump([relation_ids, relation_mask, input_ids, attention_mask,
                               decoder_input_ids, decoder_attention_mask,
                               metadata_rel, metadata_questions, self.raw_questions, self.raw_answers, self.raw_ids], f)

        self.dataset = MyGroupedQADataset(relation_ids, relation_mask, input_ids, attention_mask,
                                        decoder_input_ids, decoder_attention_mask,
                                        metadata_rel, metadata_questions, self.args.inner_bsz,
                                        is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def save_predictions(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))
        prediction_dict = [{"id": id0, "output": [{"answer": prediction.strip()}]} for id0, prediction in zip(self.raw_ids, predictions)]
        save_path = os.path.join(self.args.output_dir, "{}_predictions.jsonl".format(self.args.prefix))
        with open(save_path, "w") as f:
            f.writelines([json.dumps(dp)+'\n' for dp in prediction_dict])
        self.logger.info("Saved prediction in {}".format(save_path))

def add_description(input_str, raw=False):
    if raw:
        rel_name=input_str
    else:
        split_idx = input_str.index('[SEP]')
        rel_name = input_str[split_idx+6:]

    description = ZSRE_RELATIONS[rel_name]["description"]
    return "{} [SEP] description: {}".format(input_str, description)

def get_accuracy(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([int(prediction==gt) for gt in groundtruth])
    return int(prediction==groundtruth)


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
