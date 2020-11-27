import pandas as pd 
from uer.utils.constants import *


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split('\t')
            tgt = int(line[columns["label"]])
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            if "text_b" not in columns: # Sentence classification.
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
                seg = [1] * len(src)
            else: # Sentence-pair classification.
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[:args.seq_length]
                seg = seg[:args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, tgt, seg, soft_tgt))
            else:
                dataset.append((src, tgt, seg))

    return dataset

def multi_cls_loader(args, path):
    label2id = {'时政': 0, '房产': 1, '财经': 2, '科技': 3, '时尚': 4, '教育': 5, '家居': 6, '游戏':7, '娱乐':8, '体育':9}
    id2label = {0:'时政', 1:'房产', 2:'财经', 3:'科技', 4:'时尚', 5:'教育', 6:'家居', 7:'游戏', 8:'娱乐', 9:'体育'}

    dataset, labels, groups = [], [], []
    df = pd.read_csv(path)
    for idx, label, text in zip(df['id'], df['class_label'], df['content']):
        tgt = label2id[label]
        src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text))
        seg = [1] * len(src)
        if len(src) > args.seq_length:
            src = src[:args.seq_length]
            seg = seg[:args.seq_length]
        while len(src) < args.seq_length:
            src.append(0)
            seg.append(0)
        dataset.append((src, tgt, seg))
        labels.append(tgt)
        groups.append(idx)
    if args.test_run:
        print("Test run")
        return dataset[:1000], labels[:1000], groups[:1000], len(label2id)
    return dataset, labels, groups, len(label2id)
