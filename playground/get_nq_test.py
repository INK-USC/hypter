import json
import re
import pprint
import os
import argparse
import pandas as pd
import random

import numpy as np
from tqdm import tqdm

SQUAD_TEST_HEADS = ['where were', 'what political', 'what religion', 'why did', 'what type', 'what language', 'who had', 'what percentage', 'what can', 'how much']

def strip(sent):
    return sent.strip(" ").rstrip('.').rstrip('?').rstrip('!').rstrip('"')

blacklist = ["of the", "is a", "is the", "did the"]

wh = {
    "(what|what's)": 0,
    "(who|who's)": 0,
    "where": 0,
    "when": 0,
    "which": 0,
    "whose": 0,
    "whom": 0,
    "how": 0,
    "why": 0,
    "(can|could|may|might|should)": 0,
    "(is|are|were|was)": 0,
    "(will|would)": 0,
    "(do|does|did)": 0,
    "(has|have|had)": 0,
    "(name|identify|describe|define)": 0
}

wh2 = {}
wh3 = {}

keys = wh.keys()
isascii = lambda s: len(s) == len(s.encode())


def find_match(query, keys):
    for key in keys:
        if re.search('^' + key + '$', query):
            return key
    return None


def dict_add(entry, example, dict):
    if entry in blacklist:
        return
    if entry in dict:
        dict[entry].append(example)
    else:
        dict[entry] = [example]

def find_top_q_head(examples, topn):
    for example in examples:
        question_text = strip(example["question"])

        # simple tokenization
        t = question_text.split(" ")
        t = [strip(item.lower()) for item in t]

        # search if the any key is in the first three words
        flag = False
        for i in range(3):
            if i >= len(t):
                break
            key = find_match(t[i], keys)
            if key:
                wh[key] += 1
                try:
                    if key == "which" and "in which" in question_text:
                        st2 = " ".join(t[i - 1:i + 1])
                        st3 = " ".join(t[i - 1:i + 2])
                    elif key == "whom" and "by whom" in question_text:
                        st2 = " ".join(t[i - 1:i + 1])
                        st3 = " ".join(t[i - 1:i + 2])
                    else:
                        st2 = " ".join(t[i:i + 2])
                        st3 = " ".join(t[i:i + 3])
                    dict_add(st2, example, wh2)
                    # dict_add(st3, example, wh3)
                except Exception as e:
                    print(e.args)
                flag = True
                break

        if not flag:
            for i in range(len(t)):
                key = find_match(t[len(t) - i - 1], keys)
                if key:
                    wh[key] += 1
                    flag = True
                    idx = len(t) - i - 1
                    try:
                        if key == "which" and "in which" in question_text:
                            st2 = " ".join(t[i - 1:i + 1])
                            st3 = " ".join(t[i - 1:i + 2])
                        elif key == "whom" and "by whom" in question_text:
                            st2 = " ".join(t[i - 1:i + 1])
                            st3 = " ".join(t[i - 1:i + 2])
                        else:
                            st2 = " ".join(t[i:i + 2])
                            st3 = " ".join(t[i:i + 3])
                        dict_add(st2, example, wh2)
                        # dict_add(st3, wh3)
                    except Exception as e:
                        print(e.args)
                    break
        # if not flag:
            # print("No question word found: ", question_text)

    sorted_x = sorted(wh2.items(), key=lambda kv: len(kv[1]), reverse=True)
    print('#Question Head:', len(sorted_x))
    # for i in range(topn):
    #     print(sorted_x[i][0], len(sorted_x[i][1]))
    # pp = pprint.PrettyPrinter(indent=4)
    # print(sorted_x[:topn])
    # print('#Hits in Top {}:'.format(topn), sum(item[1] for item in sorted_x[:40]))
    # print('#Examples', len(examples))
    # return [kv[0] for kv in sorted_x[:topn]]
    return sorted_x[:topn]


def get_questions(examples, head, num):
    random.shuffle(examples)
    ret = []
    count = 0
    for example in examples:
        if head in example.question_text.lower() and len(example.orig_answer_text) > 0 \
                and isascii(example.orig_answer_text) and isascii(" ".join(example.doc_tokens)):
            ret.append(example)
            count += 1
            if count == num:
                break
    if count != num:
        print(head)
        print(ret)
    return ret

def read_nq_examples(input_file):
    with open(input_file, 'r') as fin:
        lines = fin.readlines()
        lines = lines[1:] # exclude the header
        source = [json.loads(line.strip()) for line in lines]

    total = 0
    examples = []
    for para in tqdm(source):
        context = para["context"]
        for qa in para["qas"]:
            total += 1
            ques = qa["question"]
            ans = qa["answers"]
            examples.append({'context': context, 'question': ques, 'answer': [{'text': ans[0]}]})

    print(examples[:5])
    return examples

def down_sample_and_split(heads, n_per_head):
    random.shuffle(heads)
    new_heads = {}

    for head in heads:
        if len(head[1]) < n_per_head:
            continue
        new_heads[head[0]] = random.sample(head[1], n_per_head)
    
    print(new_heads.keys())

    test = {head: new_heads[head] for head in SQUAD_TEST_HEADS if head in new_heads}

    return test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", default='../data/newsqa/NewsQA.jsonl', type=str, required=False)
    parser.add_argument("--out_dir", default='../data/newsqa/', type=str, required=False,
                        help="Output directory")
    parser.add_argument("--out_train", default='zs_train.json', type=str, required=False)
    parser.add_argument("--out_dev", default='zs_dev.json', type=str, required=False)
    parser.add_argument("--out_test", default='zs_test.json', type=str, required=False)
    parser.add_argument('--seed', type=int, default=55, help="random seed")

    args = parser.parse_args()
    opt = vars(args)

    random.seed(opt['seed'])

    examples = read_nq_examples(opt['in_file'])
    top_heads = find_top_q_head(examples, topn=300)
    test = down_sample_and_split(top_heads, n_per_head=64)

    print('Test heads: {}'.format(test.keys()))

    # if not os.path.exists(opt['out_dir']):
    #     os.makedirs(opt['out_dir'])
    # with open(os.path.join(opt['out_dir'], opt['out_train']), 'w') as fout:
    #     json.dump(train, fout)
    # with open(os.path.join(opt['out_dir'], opt['out_dev']), 'w') as fout:
    #     json.dump(dev, fout)
    with open(os.path.join(opt['out_dir'], opt['out_test']), 'w') as fout:
        json.dump(test, fout)


if __name__ == "__main__":
    main()