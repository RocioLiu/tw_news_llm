import gc
import os
import random
import re
from os.path import abspath, dirname, split, join, isfile

from accelerate import Accelerator
import argparse
import bitsandbytes as bnb
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, Value, ClassLabel, Features
from datetime import datetime
from distutils.util import strtobool
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    default_data_collator,
    get_scheduler,
)

from peft import (
    get_peft_config,
    get_peft_model,
    PeftModel,
    PeftConfig,
    LoraConfig,
    prepare_model_for_int8_training,
    TaskType,
    PeftType,
)

ROOT_PATH = join(*split(abspath(dirname("__file__"))))
DATA_PATH = join(ROOT_PATH, "data")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='seed of the experimnet')
    parser.add_argument('--task-name', type=str, default='tw-llm')
    parser.add_argument('--dataset-name-or-path', type=str,
                        default="./data/corpus.json")
    parser.add_argument('--model-name-or-path', type=str,
                        default="bigscience/bloomz-560m")
    parser.add_argument('--output-dir', type=str, default="outputs")
    parser.add_argument('--fp16', default=True)
    parser.add_argument('--bf16', default=False)
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=int, default=0.05)
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--min-length', type=int, default=32)
    parser.add_argument('--lr', type=int, default=3e-5)
    parser.add_argument('--weight-decay', type=int, default=0.02)
    parser.add_argument('--scheduler-name', type=str, default="cosine")
    parser.add_argument('--num-warmup-steps', type=int, default=2000)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--print-train-every-n-steps', type=int, default=25)
    parser.add_argument('--eval-steps', type=int, default=100)

    # to add weight and bias, we set
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default="tw-llm",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    args = parser.parse_args()
    return args


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) if device == "cuda" else None


def tokenize(examples):
    outputs = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    mask_batch = []
    length_batch = []
    for i, (length, input_ids) in enumerate(zip(outputs['length'], outputs['input_ids'])):
        if length >= min_length:
            padding_length = (max_length - length)
            padded_input = [tokenizer.pad_token_id]*padding_length + input_ids
            attention_mask = [0]*padding_length + [1]*length
            input_batch.append(padded_input)
            mask_batch.append(attention_mask)
            length_batch.append(length)

    return {'input_ids': input_batch, 'attention_mask': mask_batch, 'num_tokens': length_batch}


def calculate_model_size(model, model_name_or_path):
    num_params = sum(t.numel() for t in model.parameters())

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2

    print(f"{model_name_or_path} size: {num_params/1000**2:.1f}M parameters")
    print(f"model stored size: {size_all_mb:.3f}MB")
            

def eval_fn(data_loader):
    model.eval()
    losses = []
    for step, batch in enumerate(data_loader):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
        loss = outputs.loss

        del batch
        gc.collect()
        torch.cuda.empty_cache()    

        losses.append(accelerator.gather(torch.unsqueeze(loss, -1))) 

    avg_loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")

    return avg_loss.item(), perplexity.item()


if __name__ == "__main__":
    accelerator = Accelerator()
    device = accelerator.device

    args = parse_args()

    seed = args.seed
    task_name = args.task_name
    dataset_name_or_path = args.dataset_name_or_path
    model_name_or_path = args.model_name_or_path
    output_dir = args.output_dir
    fp16 = args.fp16
    bf16 = args.bf16
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    max_length = args.max_length
    min_length = args.min_length
    lr = args.lr
    weight_decay = args.weight_decay
    scheduler_name = args.scheduler_name
    num_warmup_steps = args.num_warmup_steps
    gradient_accumulation_steps = args.gradient_accumulation_steps
    print_train_every_n_steps = args.print_train_every_n_steps
    eval_steps = args.eval_steps
    track = args.track
    wandb_project_name = args.wandb_project_name
    wandb_entity = args.wandb_entity

    set_seed(seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if fp16:
        MODEL_DTYPE = torch.float16
    elif bf16:
        MODEL_DTYPE = torch.bfloat16
    else:
        MODEL_DTYPE = torch.float

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
        )
    
    peft_model_id = f"{task_name}_{model_name_or_path.split('/')[-1]}_{peft_config.peft_type}_{peft_config.task_type}"
    # wandb run name
    run_name = f"{peft_model_id}_{datetime.now().strftime('%Y%m%d%H%M')}"
    lora_run_name = f"{peft_model_id}_r-{lora_rank}_alpha-{lora_alpha}_{datetime.now().strftime('%Y%m%d%H%M')}"
    wandb_run_name = lora_run_name if peft_config.task_type == "LORA" else run_name

    # Experiment tracking with weights & Bias
    if track:
        run = wandb.init(project=wandb_project_name,
                         entity=wandb_entity,
                         name=wandb_run_name,
                         group=model_name_or_path.split("/")[-1],
                         config=vars(args),
                         sync_tensorboard=True,
                         save_code=True,
                         job_type="train")
        
    
    # 1. Data preprocess
    # 1-1. Import the preprocessed data
    ds = load_dataset("json", data_files=dataset_name_or_path, split="train")
    features = ds.features.copy()
    class_features = list(set(ds['class']))
    features["class"] = ClassLabel(names=class_features)
    ds = ds.cast_column("class", features["class"])
    

    # 1-2. Train / validation split
    ds_split = ds.train_test_split(test_size=0.05, stratify_by_column="class", seed=seed) # 12266:646
    ds_train, ds_valid = ds_split['train'], ds_split['test']
    ds_split['valid'] = ds_valid
    del ds_split['test']
    
    
    # 1-3. Import model tokenizer and tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tknz_ds = ds_split.map(tokenize, batched=True, remove_columns=ds_split["train"].column_names)

    # 2. Create DataLoader
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_dataloader = DataLoader(tknz_ds["train"],
                                  collate_fn=data_collator,
                                  batch_size=batch_size,
                                  shuffle=True)
    eval_dataloader = DataLoader(tknz_ds["valid"],
                                 collate_fn=data_collator,
                                 batch_size=batch_size)

    # 3. Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=MODEL_DTYPE,
        # load_in_8bit=True,
        device_map='auto',
    )
    model = get_peft_model(model, peft_config)
    print(f"Is model on cuda? {next(model.parameters()).is_cuda}")
    model.print_trainable_parameters()
    calculate_model_size(model, model_name_or_path)

    # 4. Training hyperparameters
    # optimizer and lr scheduler
    no_decay = ["bias", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # 5. Training loop
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    gc.collect()
    torch.cuda.empty_cache()

    completed_steps = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        model.train()
        for step, batch in tqdm(
            enumerate(train_dataloader, start=1), total=num_training_steps
        ):
            with torch.cuda.amp.autocast(dtype=MODEL_DTYPE):
                outputs = model(**batch)
            logits = outputs.logits
            loss = outputs.loss
            total_loss += loss.detach().float()

            del batch
            gc.collect()
            torch.cuda.empty_cache()    

            if step % print_train_every_n_steps == 0:
                accelerator.print(f"steps: [{epoch+1}]{step}/{len(train_dataloader)} |",
                                  f"updated_steps: {completed_steps} |",
                                  f"lr: {lr_scheduler.get_last_lr()[0]:.7f} |",
                                  f"train/loss: {loss.item():.4f}"
                                  )
            if args.track:
                wandb.log({f"train/loss": loss.item(),
                           f"lr": lr_scheduler.get_last_lr()[0]})
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

            if (step % (eval_steps * gradient_accumulation_steps)) == 0 or (step == (len(train_dataloader)-1)):
                eval_loss, eval_ppl = eval_fn(eval_dataloader)
                accelerator.print(f"eval/loss: {eval_loss:.4f} | eval/ppl: {eval_ppl:.4f}")
                model.train()

        avg_train_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(avg_train_loss)
        accelerator.print(f"epoch {epoch+1} result:\n" 
                          f"avg_train_loss={avg_train_loss:.4f} | train_ppl={train_ppl:.4f} | "
                          f"avg_eval_loss={eval_loss:.4f} | eval_ppl={eval_ppl:.4f}")
        if args.track:
            wandb.log(
                {
                    f"epoch": epoch+1, 
                    f"train/avg_loss": avg_train_loss,
                    f"train/perplexity": train_ppl,
                    f"eval/avg_loss": eval_loss,
                    f"eval/perplexity": eval_ppl,
                })

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(f"{output_dir}/{peft_model_id}_epoch{epoch+1}_{datetime.now().strftime('%Y%m%d%H%M')}", save_function=accelerator.save)
    #     if accelerator.is_main_process:
    #         tokenizer.save_pretrained(peft_model_id)

    if args.track:
        run.finish()
