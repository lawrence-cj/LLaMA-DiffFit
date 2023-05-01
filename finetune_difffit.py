import os
import sys
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import argparse

import torch
from datasets import load_dataset
import transformers

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    DiffFitConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["instruction"]}

    ### Input:
    {data_point["input"]}

    ### Response:
    {data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["instruction"]}

    ### Response:
    {data_point["output"]}"""


def parse_args():
    parser = argparse.ArgumentParser(description='start training')
    parser.add_argument('--run', default='run_tmp', help='output dir name')
    parser.add_argument('--bs', default=64, type=int, help='batch-size')
    args, unparsed = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    MICRO_BATCH_SIZE = int(args.bs)  # this could actually be 5 but i like powers of 2
    BATCH_SIZE = 128
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    EPOCHS = 3  # we don't always need 3 tbh
    LEARNING_RATE = 3e-4  # the Karpathy constant
    CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
    VAL_SET_SIZE = 2000
    TARGET_MODULES = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        # "gate_proj",
        # "down_proj",
        # "up_proj"
    ]
    ETA_SCALE = 1.0
    ETA_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # ETA_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 100]    # 100 means all layers

    DATA_PATH = "alpaca_data_cleaned.json"
    OUTPUT_ROOT = "output"
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, args.run)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

    model = LlamaForCausalLM.from_pretrained(
        "output/pretrained/llama-7b-hf",
        load_in_8bit=True,
        device_map=device_map,
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        "output/pretrained/llama-7b-hf", add_eos_token=True
    )

    def tokenize(prompt):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN + 1,
            padding="max_length",
        )
        return {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
        }

    def generate_and_tokenize_prompt(data_point):
        prompt = generate_prompt(data_point)
        return tokenize(prompt)

    model = prepare_model_for_int8_training(model)

    config = DiffFitConfig(
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
        eta_scale=ETA_SCALE,
        eta_layers = ETA_LAYERS
    )
    print(config)
    print("LEARNING_RATE: ", LEARNING_RATE, "MICRO_BATCH_SIZE: ", MICRO_BATCH_SIZE, "OUTPUT_DIR: ", args.run)

    model = get_peft_model(model, config)

    for name, param in model.named_parameters():
        if 'bias' in name or 'difffit' in name:
            print("Trainable Parameter: {}".format(name))
            param.requires_grad = True
        else:
            param.requires_grad = False

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    data = load_dataset("json", data_files=DATA_PATH)

    if VAL_SET_SIZE > 0:
        train_val = data["train"].train_test_split(
            test_size=VAL_SET_SIZE, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=20,
            evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if VAL_SET_SIZE > 0 else None,
            save_steps=200,
            output_dir=OUTPUT_DIR,
            save_total_limit=3,
            load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            report_to='none',
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)

    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    main()