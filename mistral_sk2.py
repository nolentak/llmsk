import os
import datasets
import psutil
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, \
    AutoModelForCausalLM, AutoTokenizer, logging

from util import pack

#WORK_DIR = os.getenv('WORK', os.getcwd())
OUTPUT_DIR = os.path.join('/leonardo_work/L-SLK_001_24/output_mistral_sk2')
DATASET_DIR = os.path.join('/leonardo_work/L-SLK_001_24/aranea+hplt+fineweb-hf')
BASE_MODEL_DIR = os.path.join('/leonardo_work/L-SLK_001_24/Mistral-7B-v0.1')

def train(base_model, context_length, dataset, new_model_name):
    model = AutoModelForCausalLM.from_pretrained(base_model,
                                                 torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True,
                                                 attn_implementation='flash_attention_2',
                                                 use_safetensors=True,
                                                 device_map='auto')

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

    # fix padding (mostly for inference, later for finetuning changed to unk_token_id)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # it is customary to train LLMs by fully "packing" the context length with
    # fragments of one or more documents
    packed_train_dataset = datasets.IterableDataset.from_generator(
        generator=pack,
        gen_kwargs={'dataset': dataset['train'],
                    'tokenizer': tokenizer,
                    'context_length': context_length})

    packed_validation_dataset = datasets.IterableDataset.from_generator(
        generator=pack,
        gen_kwargs={'dataset': dataset['validation'],
                    'tokenizer': tokenizer,
                    'context_length': context_length})

    per_device_train_batch_size = 2
    gradient_accumulation_steps = 8
    training_steps = 10_000_000_000 // (torch.cuda.device_count() * per_device_train_batch_size *
                                       gradient_accumulation_steps * context_length)

    save_steps = 512
    eval_steps = 256

    print(f'training steps: {training_steps}')
    print(f'save steps: {save_steps}')
    print(f'eval steps: {eval_steps}')

    training_args = TrainingArguments(
        max_steps=training_steps,
        optim='adamw_bnb_8bit',
        learning_rate=2e-5,
        lr_scheduler_type='cosine',
        warmup_steps=int(training_steps * 0.1),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,
        eval_strategy='steps',
        eval_steps=eval_steps,
        per_device_eval_batch_size=per_device_train_batch_size,
        eval_accumulation_steps=gradient_accumulation_steps,
        save_strategy='steps',
        save_steps=save_steps,
        bf16=True,
        output_dir=OUTPUT_DIR,
        report_to=['tensorboard'],
        logging_steps=1,
        logging_first_step=True,
        hub_model_id=new_model_name,
        hub_private_repo=True,
        push_to_hub=False,
        hub_strategy='all_checkpoints',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=packed_train_dataset,
        eval_dataset=packed_validation_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train(resume_from_checkpoint=os.path.exists(training_args.output_dir) and any(
    fname.startswith("checkpoint") for fname in os.listdir(training_args.output_dir)
    ))

    # Uloženie finálneho modelu
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == '__main__':
    logging.set_verbosity_info()
    dataset = datasets.load_from_disk(DATASET_DIR)
    train(
        base_model=BASE_MODEL_DIR,
        context_length=8192,
        dataset=dataset,
        new_model_name='mistral-sk2-7B',
    )