## Code for "Zero-shot Learning by Generating Task-specific Adapters"

This is the repository containing code for "Zero-shot Learning by Generating Task-specific Adapters" ([arXiv](https://arxiv.org/abs/2101.00420)). This is a beta version and we will add more details in the future.


### Environment
We modified the code in [shmsw25/bart-closed-book-qa](https://github.com/shmsw25/bart-closed-book-qa) (Thanks to the authors!). 

Following their instructions, please install the environment with these commands:

```
pip install torch==1.1.0
pip install git+https://github.com/huggingface/transformers.git@7b75aa9fa55bee577e2c7403301ed31103125a35
```

### Data

Download ZEST dataset from [here](https://allenai.org/data/zest) and place (`zest_{train|dev|test_unanswered}.jsonl`) in `./data`.

### Run
See `./scripts/zest_bart_large.sh` and `./scripts/zest_grouped_bart_large_from_trained.sh`

### Cite Us
```
@article{Ye2021ZeroshotLB,
  title={Zero-shot Learning by Generating Task-specific Adapters},
  author={Qinyuan Ye and Xiang Ren},
  journal={ArXiv},
  year={2021},
  volume={abs/2101.00420}
}
```