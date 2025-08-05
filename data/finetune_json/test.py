import json

with open('data/finetune_json/rsitmd_train.json', 'r') as f:
    data = json.load(f)

print(data[0].keys())

print(data[0]['label_name'])

tot_label = set()
for item in data:
    tot_label.add(item['label_name'])

print(tot_label)