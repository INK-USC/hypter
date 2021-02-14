from transformers import BartTokenizer, RobertaTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

b = [0, 2060, 1322, 1916, 1029, 611, 459, 2186, 15747, 4, 6154, 718, 38263, 354, 646, 3388, 510, 742, 38, 12945, 487, 8360, 2194, 2]

ids = tokenizer.convert_ids_to_tokens(b)
print(ids)

# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# b = [    0,   646, 49643, 23454,  1215,   560,  1215,  3698,  1215, 34015,                                                                                               
#          7862,    22,   282,    73,   102,  1297,    22, 10111,  1215,   560,                                                                                               
#          1215,  9854,  7862,    22,   282,    73,   102, 48805,   742,     2,                                                                                               
#             1]

# ids = tokenizer.convert_ids_to_tokens(b)
# print(ids)