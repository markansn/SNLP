#!/usr/bin/env python3
# src https://huggingface.co/patrickvonplaten/roberta2roberta-share-cnn_dailymail-fp16
import nlp
import logging
import IPython
from transformers import RobertaTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
from datasets import load_dataset
logging.basicConfig(level=logging.INFO)

import torch
torch.cuda.empty_cache()

print(torch.cuda.is_available())


model = EncoderDecoderModel.from_encoder_decoder_pretrained("roberta-base", "roberta-base", tie_encoder_decoder=True)
tokenizer =  RobertaTokenizer.from_pretrained("roberta-base")

# load train and validation data
train_dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", split="train[:5%]")
val_dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", split="validation[:5%]")

print(train_dataset)
print(val_dataset)
# load rouge for validation
rouge = nlp.load_metric("rouge", experiment_id=0)

# set decoding params
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 142
model.config.min_length = 56
model.config.no_repeat_ngram_size = 3
model.early_stopping = True
model.length_penalty = 2.0
model.num_beams = 4

encoder_length = 512
decoder_length = 128
batch_size = 4


# map data correctly
def map_to_encoder_decoder_inputs(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at Longformer at 2048
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_length)
    # force summarization <= 256
    outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=decoder_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["labels"] = outputs.input_ids.copy()
    # mask loss for padding
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    ]
    batch["decoder_attention_mask"] = outputs.attention_mask

    assert all([len(x) == encoder_length for x in inputs.input_ids])
    assert all([len(x) == decoder_length for x in outputs.input_ids])

    return batch


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.eos_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


# make train dataset ready
train_dataset = train_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
)
train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_attention_mask", "decoder_input_ids", "labels"],
)

# same for validation dataset
val_dataset = val_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
)
val_dataset.set_format(
    type="torch", columns=["input_ids", "decoder_attention_mask", "attention_mask", "decoder_input_ids", "labels"],
)

# set training arguments - these params are not really tuned, feel free to change
training_args = TrainingArguments(
    output_dir="./",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    do_train=True,
    do_eval=True,
    logging_steps=1000,
    save_steps=1000,
    eval_steps=1000,
    overwrite_output_dir=True,
    warmup_steps=2000,
    save_total_limit=3,
    fp16=True,
)

# instantiate trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
# IPython.embed()
# start training
trainer.train()