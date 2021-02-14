import json
import re
import pprint
import os
import argparse
import pandas as pd
import random

import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default='../data/squad/', type=str, required=False,
                        help="Input directory")
    parser.add_argument("--in_train", default='zs_train.json', type=str, required=False)
    parser.add_argument("--out_dir", default='../data/squad/', type=str, required=False,
                        help="Output directory")
    parser.add_argument("--out_train", default='zs_train-80-8.json', type=str, required=False)
    parser.add_argument("--n_tasks", default=80, type=int, required=False)
    parser.add_argument("--n_per_task", default=8, type=int, required=False)
    parser.add_argument('--seed', type=int, default=55, help="random seed")

    args = parser.parse_args()
    opt = vars(args)

    with open(os.path.join(opt['in_dir'], opt['in_train'])) as fin:
        d = json.load(fin)
        new_d = {}

        for k, v in list(d.items())[:opt['n_tasks']]:
            # print(k)
            new_d[k] = v[:opt['n_per_task']]

    with open(os.path.join(opt['out_dir'], opt['out_train']), "w") as fout:
        json.dump(new_d, fout)

    # examples = read_squad_examples(opt['in_file'])
    # top_heads = find_top_q_head(examples, topn=100)
    # train, dev, test = down_sample_and_split(top_heads, n_per_head=64)
    # # print(list(dev.values())[0])

    # print('Train heads: {}'.format(train.keys()))
    # print('Dev heads: {}'.format(dev.keys()))
    # print('Test heads: {}'.format(test.keys()))




if __name__ == "__main__":
    main()