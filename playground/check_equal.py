import json

d1 = {}
with open("/home/qinyuan/zs/out/bart-large-with-description-grouped-1e-5-outerbsz4-innerbsz32-adapterdim4-unfreeze-dec29/test_predictions.jsonl") as fin:
    for line in fin:
        d = json.loads(line)
        d1[d["id"]] = d["output"][0]["answer"]

d2 = {}
dq = {}
with open("/home/qinyuan/zs/out/bart-large-zsre-with-description-LR2e-5-FREQ32-dec27/test_predictions_submitted.jsonl") as fin:
    for line in fin:
        d = json.loads(line)
        d2[d["id"]] = d["output"][0]["answer"]
        dq[d["id"]] = d["input"]

d3 = {}
with open("/home/qinyuan/zs/data/structured_zeroshot-test.jsonl") as fin:
    for line in fin:
        d = json.loads(line)
        d3[d["id"]] = [item["answer"] for item in d["output"]]

count = 0
win1 = 0
win2 = 0
for key in d1.keys():
    if d1[key]!= d2[key]:
        print("{}. {}. {}. {}. {}".format(key, dq[key], d1[key], d2[key], d3[key]))
        count += 1

        if d1[key] in d3[key] and d2[key] not in d3[key]:
            win1 += 1
            print(d1[key])
            print(d2[key])
        if d2[key] in d3[key] and d1[key] not in d3[key]:
            win2 += 1
            print(d1[key])
            print(d2[key])
            

print(count)
print(win1)
print(win2)


