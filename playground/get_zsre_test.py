import json

def main():
    lines = []
    for i in range(10):
        with open("/home/qinyuan/zs/data/zsre/relation_splits/train.{}".format(i)) as fin:
            lines += fin.readlines()
        with open("/home/qinyuan/zs/data/zsre/relation_splits/dev.{}".format(i)) as fin:
            lines += fin.readlines() 
        with open("/home/qinyuan/zs/data/zsre/relation_splits/test.{}".format(i)) as fin:
            lines += fin.readlines() 
    
    d = {}

    for line in lines:
        elements = line.strip().split("\t")
        if len(elements) < 5:
            continue
        rel = elements[0]
        subj = elements[2]
        obj = elements[4]

        input0 = "{} [SEP] {}".format(subj, rel)
        if input0 not in d:
            d[input0] = [obj]
        else:
            d[input0].append(obj)

    output_lines = []
    with open("/home/qinyuan/zs/data/structured_zeroshot-test_without_answers-kilt.jsonl") as fin:
        for line in fin:
            p = json.loads(line)
            if p["input"] in d:
                if len(set(d[p["input"]])) > 1:
                    # print(p["input"])
                    print(set(d[p["input"]]))
                p["output"] = [{"answer": ans.strip()} for ans in set(d[p["input"]])]
                output_lines += json.dumps(p)+'\n'
            else:
                print(p["input"])
    
    with open("/home/qinyuan/zs/data/structured_zeroshot-test.jsonl", "w") as fout:
        fout.writelines(output_lines)

if __name__ == "__main__":
    main()