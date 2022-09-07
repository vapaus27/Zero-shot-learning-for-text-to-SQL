import json

import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer


class Text2SQLDataset(Dataset):
    def __init__(self, table_json_path, data_json_path, tokenizer: T5Tokenizer, src_length=512, tgt_length=128):
        super().__init__()
        self.table_header = self.load_table(table_json_path)
        self.data = self.load_data(data_json_path)
        self.tokenizer = tokenizer
        self.prefix = "translate English to SQL: "
        self.src_length = src_length
        self.tgt_length = tgt_length

    def load_table(self, table_json_path):
        with open(table_json_path, "r", encoding="utf-8") as f:
            table_data = json.load(f)
        table_header = {}
        for data in table_data:
            table_header[data["db_id"]] = [d[1] for d in data["column_names_original"]]
        return table_header

    def load_data(self, data_json_path):
        with open(data_json_path, "r", encoding="utf-8") as f:
            data_json = json.load(f)
        data = []
        for d in data_json:
            example = {}
            example["columns"] = self.table_header[d["db_id"]]
            example["query"] = d["query"]
            example["question"] = d["question"]
            data.append(example)
        return data

    def concat_source(self, question, columns, type_sep_token="<extra_id_0>", column_sep_token="<extra_id_1>"):
        source = self.prefix
        columns_with_sep = []
        for c in columns:
            columns_with_sep.append(c)
            columns_with_sep.append(column_sep_token)
        columns_with_sep = " ".join(columns_with_sep)
        source = source + question
        source = " {} ".format(type_sep_token).join([source, columns_with_sep])
        return source

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        concat_src = self.concat_source(data["question"], data["columns"])
        print(concat_src)
        print(data["query"])
        src = self.tokenizer(concat_src,
                             max_length=self.src_length,
                             padding="max_length",
                             truncation=True,
                             # return_tensors="pt",
                             )
        with self.tokenizer.as_target_tokenizer():
            tgt = self.tokenizer(data["query"],
                                 max_length=self.tgt_length,
                                 padding="max_length",
                                 truncation=True,
                                 # return_tensors="pt",
                                 )
        input_ids = src.input_ids
        attention_mask = src.attention_mask
        labels = tgt.input_ids
        labels = [token if token != self.tokenizer.pad_token_id else -100 for token in labels]
        decoder_attention_mask = tgt.attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_attention_mask": decoder_attention_mask,
        }

tokenizer = T5Tokenizer.from_pretrained("/Users/daning/Project/RTEWorkShop/RTECode/GenerativeRTE/pretrained/t5-small")
dataset = Text2SQLDataset("spider/tables.json", "spider/train_spider.json", tokenizer)
# dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=3)
src_length = []
tgt_length = []
for i in dataset:
    src_length.append(len(i["input_ids"]))
    tgt_length.append(len(i["labels"]))

print(max(src_length))
print(max(tgt_length))

print(sorted(src_length, reverse=True)[:200])
print(sorted(tgt_length, reverse=True)[:200])
