import os
import json
import re
import string
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from .utils import MyGroupedQADataset, MyGroupedDataLoader

class ZSREGroupedData(object):

    def __init__(self, logger, args, data_path, is_training):
        self.data_path = data_path
        if args.debug:
            self.data_path = data_path.replace("train", "dev")
        with open(self.data_path, "r") as f:
            json_list = list(f)
        self.data = [json.loads(json_str) for json_str in json_list]

        if args.debug:
            self.data = self.data[:40]

        assert type(self.data)==list
        assert all(["id" in d for d in self.data]), self.data[0].keys()

        if type(self.data[0]["id"])==int:
            for i in range(len(self.data)):
                self.data[i]["id"] = str(self.data[i]["id"])

        self.index2id = {i:d["id"] for i, d in enumerate(self.data)}
        self.id2index = {d["id"]:i for i, d in enumerate(self.data)}

        self.is_training = is_training
        self.load = not args.debug
        self.logger = logger
        self.args = args

        if "test" in self.data_path:
            self.data_type = "test"
        elif "dev" in self.data_path:
            self.data_type = "dev"
        elif "train" in self.data_path:
            self.data_type = "train"
        else:
            raise NotImplementedError()

        self.metric = "EM"
        self.max_input_length = self.args.max_input_length
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None

    def __len__(self):
        return len(self.data)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def flatten(self, raw_data):
        questions, answers, metadata_rel, metadata_questions = [], [], [], []
        new_questions = []
        new_answers = []
        for relation in raw_data:
            metadata_rel.append((len(new_questions), len(new_questions)+len(relation)))
            new_questions += [qa[0] for qa in relation]
            for qa in relation:
                metadata_questions.append((len(new_answers), len(new_answers)+len(qa[1])))
                new_answers += qa[1]

        return new_questions, new_answers, metadata_rel, metadata_questions

    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer
        postfix = 'Grouped-' + tokenizer.__class__.__name__.replace("zer", "zed")
        
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(".json", "-{}.json".format(postfix)))
        
        if self.load and os.path.exists(preprocessed_path):
            # load preprocessed input
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                relation_ids, relation_mask, input_ids, attention_mask, \
                    decoder_input_ids, decoder_attention_mask, \
                    metadata_rel, metadata_questions = json.load(f)

        else:
            print("Start tokenizing ... {} instances".format(len(self.data)))

            relations = sorted(list(set([d['input'][d['input'].index("[SEP]")+6:] for d in self.data])))
            raw_data = [[] for _ in range(len(relations))]
            id2relation = {k: v for k,v in enumerate(relations)}
            relation2id = {v: k for k,v in enumerate(relations)}

            print("relation2id: {}".format(relation2id))

            for d in self.data:
                rel = d['input'][d['input'].index("[SEP]")+6:]
                rel_id = relation2id[rel]
                raw_data[rel_id].append((d["input"], [item["answer"] for item in d["output"]]))
            
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
                               metadata_rel, metadata_questions], f)

        self.dataset = MyGroupedQADataset(relation_ids, relation_mask, input_ids, attention_mask,
                                        decoder_input_ids, decoder_attention_mask,
                                        metadata_rel, metadata_questions,
                                        is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = MyGroupedDataLoader(self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))
        ems = []
        for (prediction, dp) in zip(predictions, self.data):
            ems.append(get_exact_match(prediction, [item["answer"] for item in dp["output"]]))
        # for i in range(5):
        #     print(predictions[i])
        #     print([item["answer"] for item in self.data[i]["output"]])
        
        return ems

    def save_predictions(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))
        prediction_dict = {dp["id"]:prediction for dp, prediction in zip(self.data, predictions)}
        save_path = os.path.join(self.args.output_dir, "{}predictions.json".format(self.args.prefix))
        with open(save_path, "w") as f:
            json.dump(prediction_dict, f)
        self.logger.info("Saved prediction in {}".format(save_path))

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
