from os.path import abspath, dirname, split, join
import random
import re

import argparse
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
import torch
import gc

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
    PeftModel, #
    PeftConfig, #
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
    PeftType,
)

ROOT_PATH = join(*split(abspath(dirname("__file__"))))
DATA_PATH = join(ROOT_PATH, "data")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='seed of the experimnet')
    parser.add_argument('--dataset-name-or-path', type=str, default="./data/corpus.json")
    parser.add_argument('--model-name-or-path', type=str, default="bigscience/bloomz-1b7")
    parser.add_argument('--peft-model-name', type=str, default="tw-llm_bloomz-1b7_LORA_CAUSAL_LM_epoch19_202305090455")
    parser.add_argument('--output-dir', type=str, default="outputs")
    parser.add_argument('--input-prompt', type=str, default="")
    parser.add_argument('--fp16', default=True)
    parser.add_argument('--bf16', default=False)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--min-length', type=int, default=32)
    parser.add_argument('--uid', type=str, default="27-ES2003b")
    
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



def inference_fn(model, text, max_new_tokens=300, repetition_penalty=1.1):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    model, inputs = accelerator.prepare(model, inputs)
    model.eval()
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}       
            outputs = accelerator.unwrap_model(model).generate(**inputs, 
                                                               max_new_tokens=max_new_tokens, 
                                                               eos_token_id=tokenizer.eos_token_id, 
                                                               repetition_penalty=repetition_penalty,
                                                               )
        outputs = accelerator.gather(outputs).cpu().numpy()
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # find the last period "。"
        try:
            output_end = re.search(r'(.*?)(.*)\。', output_text).span()[1]
        except:
            # if there's no period
            output_end = len(output_text)
        pred = output_text[:output_end]
        print(f"prompt: {text}\noutputs: {pred}")
    del outputs, inputs
    gc.collect()


if __name__ == "__main__":
    accelerator = Accelerator()
    device = accelerator.device

    args = parse_args()

    seed = args.seed
    dataset_name_or_path_1 = args.dataset_name_or_path_1
    dataset_name_or_path_2 = args.dataset_name_or_path_2
    model_name_or_path = args.model_name_or_path
    peft_model_name = args.peft_model_name
    output_dir = args.output_dir
    input_prompt = args.input_prompt
    fp16 = args.fp16
    bf16 = args.bf16
    batch_size = args.batch_size
    max_length = args.max_length
    min_length = args.min_length
    uid = args.uid

    set_seed(seed)

    if fp16:
        MODEL_DTYPE = torch.float16
    elif bf16:
        MODEL_DTYPE = torch.bfloat16
    else:
        MODEL_DTYPE = torch.float


    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # creating model
    peft_model_path = f"{output_dir}/checkpoint/{peft_model_name}"
    peft_config = PeftConfig.from_pretrained(peft_model_path)

    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=MODEL_DTYPE,
        device_map="auto"
        )

    model = PeftModel.from_pretrained(
        model, 
        peft_model_path, 
        torch_dtype=MODEL_DTYPE,
        device_map="auto")
    
    model.print_trainable_parameters()
    calculate_model_size(model, model_name_or_path)

    accelerator = Accelerator()

    
    # inference
    inference_fn(model, input_prompt, max_new_tokens=400, repetition_penalty=1.1)