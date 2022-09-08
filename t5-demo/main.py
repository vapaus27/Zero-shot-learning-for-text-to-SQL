import torch
import numpy as np
import random
import json
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    get_linear_schedule_with_warmup,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from dataset import Text2SQLDataset

from datasets import load_metric

rouge = load_metric("./rouge.py")
import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
    res = []
    for pred, label in zip(pred_str, label_str):
        res.append({
            "pred": pred,
            "label": label,
        })

    with open("output.json", "w", encoding='utf-8') as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=4))
    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

epochs = 100
batch_size = 3
weight_decay = 0.0
lr = 1e-4

config = T5Config.from_pretrained("pretrained/t5-base-finetuned-wikiSQL/")
tokenizer = T5Tokenizer.from_pretrained("pretrained/t5-base-finetuned-wikiSQL/")
model = T5ForConditionalGeneration.from_pretrained(
    "pretrained/t5-base-finetuned-wikiSQL/")
train_set = Text2SQLDataset("spider/tables.json", "spider/train_spider.json", tokenizer)
test_set = Text2SQLDataset("spider/tables.json", "spider/dev.json", tokenizer)
args = Seq2SeqTrainingArguments(
    output_dir="output/",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    weight_decay=weight_decay,
    learning_rate=lr,
    logging_dir='logs/',
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="rouge2_fmeasure",
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=test_set,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
