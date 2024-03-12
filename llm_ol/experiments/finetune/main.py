import os
from pathlib import Path

import torch
import wandb
from absl import app, flags, logging
from accelerate import PartialState
from datasets import Dataset, load_dataset
from ml_collections import config_flags
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerControl, TrainerState
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from llm_ol.experiments.finetune.templates import MISTRAL_TEMPLATE
from llm_ol.utils import setup_logging

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config")


class GenerateSamplesCallback(TrainerCallback):
    def __init__(self, num_samples: int, response_template: list[int]):
        super().__init__()
        self.num_samples = num_samples
        self.response_template = torch.tensor(response_template, device="cpu")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        if not state.is_world_process_zero:
            return

        model = kwargs["model"]
        eval_loader = kwargs["eval_dataloader"]
        tokenizer = kwargs["tokenizer"]
        model.config.use_cache = True

        prompts = []
        for batch in eval_loader:
            if len(prompts) >= self.num_samples:
                break

            device = batch["input_ids"].device
            for input_ids, attention_mask in zip(
                batch["input_ids"].cpu(), batch["attention_mask"].cpu()
            ):
                if len(prompts) >= self.num_samples:
                    break

                # search for the response template
                for start in range(len(input_ids) - len(self.response_template)):
                    end = start + len(self.response_template)
                    if (input_ids[start:end] == self.response_template).all():
                        prompts.append(
                            {
                                "input_ids": input_ids[:end].to(device),
                                "attention_mask": attention_mask[:end].to(device),
                                "target_ids": input_ids[end:].to(device),
                            }
                        )
                        break

        samples = []
        for prompt in prompts:
            [sample] = model.generate(
                inputs=prompt["input_ids"].unsqueeze(0),
                attention_mask=prompt["attention_mask"].unsqueeze(0),
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id,
            )
            samples.append(
                {
                    "prompt": tokenizer.decode(prompt["input_ids"]),
                    "completion": tokenizer.decode(sample[len(prompt["input_ids"]) :]),
                    "target": tokenizer.decode(
                        prompt["target_ids"],
                        skip_special_tokens=True,  # Remove pad tokens
                    ),
                }
            )

        for i, sample in enumerate(samples):
            logging.info("Sample %d: %s", i, sample)

        if len(samples) > 0:
            table = wandb.Table(
                columns=list(samples[0].keys()),
                data=[list(s.values()) for s in samples],
            )
            wandb.log({"eval/samples": table}, step=state.global_step + 1)

        model.config.use_cache = False


def datasets_from_file(
    data_file: str | Path, eval_size: int, seed: int = 0
) -> tuple[Dataset, Dataset]:
    dataset = load_dataset("json", data_files=str(data_file), split="train")
    assert isinstance(dataset, Dataset)
    splits = dataset.train_test_split(test_size=eval_size, seed=seed)
    logging.info(
        "Train size: %d, Test size: %d", len(splits["train"]), len(splits["test"])
    )
    return splits["train"], splits["test"]


def main(_):
    config = FLAGS.config
    logging.info("Config:\n%s", config)
    setup_logging(config.output_dir, "main")

    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name, use_cache=False, device_map={"": device_string}
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.train.lora.rank,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    tokenizer.chat_template = MISTRAL_TEMPLATE
    tokenizer.padding_side = "right"
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.unk_token

    collator = DataCollatorForCompletionOnlyLM(
        response_template=config.model.response_template,
        instruction_template=config.model.instruction_template,
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )
    train_dataset, eval_dataset = datasets_from_file(
        config.data.file, config.data.eval_size, config.seed
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        max_seq_length=config.train.max_seq_length,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_kwargs={
            "add_special_tokens": False,
        },
        callbacks=[
            GenerateSamplesCallback(
                config.eval.num_generate_samples, config.model.response_template
            )
        ],
        args=TrainingArguments(
            output_dir=config.output_dir,
            overwrite_output_dir=True,
            optim="adamw_torch_fused",
            learning_rate=config.train.learning_rate,
            lr_scheduler_type="constant_with_warmup",
            warmup_steps=config.train.warmup_steps,
            report_to=["wandb", "tensorboard"],
            num_train_epochs=config.train.epochs,
            logging_steps=config.train.logging_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            gradient_accumulation_steps=config.train.grad_acc_steps,
            ddp_find_unused_parameters=False,
            group_by_length=True,
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            evaluation_strategy="steps",
            eval_steps=config.eval.eval_steps,
            save_steps=config.eval.eval_steps,
            per_device_train_batch_size=config.train.batch_size,
            per_device_eval_batch_size=config.eval.batch_size,
            seed=config.seed,
            data_seed=config.seed,
        ),
    )

    wandb.init(
        project=config.wandb.project,
        notes=config.wandb.notes,
        config=config.to_dict(),
        save_code=True,
    )
    trainer.evaluate()
    trainer.train()  # type: ignore


if __name__ == "__main__":
    app.run(main)
