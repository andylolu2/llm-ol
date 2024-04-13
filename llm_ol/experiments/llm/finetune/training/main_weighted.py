from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from absl import app, flags, logging
from accelerate import PartialState
from datasets import Dataset, load_dataset
from ml_collections import config_flags
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from llm_ol.experiments.llm.finetune.training.utils import GenerateSamplesCallback
from llm_ol.experiments.llm.templates import _MISTRAL_TEMPLATE, PROMPT_TEMPLATE
from llm_ol.utils import setup_logging

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config")


class Trainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss."""
        if not model.training:
            outputs = model(input_ids=inputs["input_ids"], labels=inputs["labels"])
            loss = outputs.loss
        else:
            outputs = model(input_ids=inputs["input_ids"])
            logits = outputs.logits
            b, s, v = logits.shape
            shift_logits = logits[..., :-1, :].reshape(b * (s - 1), v)
            shift_labels = inputs["labels"][..., 1:].reshape(b * (s - 1))
            shift_weights = inputs["weights"][..., 1:]

            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)

            loss = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, reduction="none"
            ).reshape(b, s - 1)
            loss = (loss * shift_weights).sum() / shift_weights.sum()

        return (loss, outputs) if return_outputs else loss

    def _prepare_dataset(
        self,
        dataset: Dataset,
        tokenizer,
        packing,
        dataset_text_field,
        max_seq_length,
        formatting_func,
        num_of_sequences,
        chars_per_token,
        remove_unused_columns=True,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        edge_counts = defaultdict(int)
        for example in dataset:
            for path in example["paths"]:  # type: ignore
                for u, v in zip(path[:-1], path[1:]):
                    edge_counts[(u, v)] += 1
        edge_weights = {k: 1 / v for k, v in edge_counts.items()}

        # ensure that the mean weight is 1
        mean_weight = np.mean(list(edge_weights.values()))
        edge_weights = {k: v / mean_weight for k, v in edge_weights.items()}

        def tokenize(example: dict[str, Any]):
            # Tokens and their corresponding loss weights
            input_ids = []
            weights = []

            def add_part(text: str, w):
                part_tokens = tokenizer.encode(text, add_special_tokens=False)
                input_ids.extend(part_tokens)
                weights.extend([float(w)] * len(part_tokens))

            # input prompt
            messages = [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.render(
                        title=example["title"], abstract=example["abstract"]
                    ),
                }
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            add_part(prompt, 0)

            # paths to predict
            example_weights = []
            for path in example["paths"]:
                for u, v in zip(path[:-1], path[1:]):
                    example_weights.append(edge_weights[(u, v)])
            mean_weight = np.mean(example_weights)

            for path in example["paths"]:
                add_part(path[0], mean_weight)
                for u, v in zip(path[:-1], path[1:]):
                    add_part("->", edge_weights[(u, v)])
                    add_part(v, edge_weights[(u, v)])
                add_part("\n", mean_weight)
            add_part(tokenizer.eos_token, mean_weight)
            return {"input_ids": input_ids, "weights": weights}

        dataset = dataset.map(tokenize, num_proc=self.dataset_num_proc)
        dataset = dataset.filter(
            lambda ex: len(ex["input_ids"]) <= max_seq_length,
            num_proc=self.dataset_num_proc,
        )
        return dataset


class DataCollator(DataCollatorForCompletionOnlyLM):
    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(len(ex["input_ids"]) for ex in examples)
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length // self.pad_to_multiple_of) + 1
            ) * self.pad_to_multiple_of
        max_length = min(max_length, self.tokenizer.model_max_length)

        input_ids = []
        weights = []
        for ex in examples:
            diff = max_length - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [self.tokenizer.pad_token_id] * diff)
            weights.append(ex["weights"] + [0] * diff)
        input_ids = torch.tensor(input_ids)
        weights = torch.tensor(weights)
        labels = input_ids.clone()
        labels[weights == 0] = -100
        return {"input_ids": input_ids, "labels": labels, "weights": weights}


def dataset_from_file(
    data_file: str | Path, size: int | None = None, seed: int = 0
) -> Dataset:
    dataset = load_dataset("json", data_files=str(data_file), split="train")
    assert isinstance(dataset, Dataset)
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    return dataset


def main(_):
    config = FLAGS.config
    logging.info("Config:\n%s", config)
    setup_logging(config.output_dir, "main")

    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        use_cache=False,
        device_map={"": device_string} if torch.cuda.is_available() else "auto",
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.train.lora.rank,
            lora_alpha=config.train.lora.alpha,
            lora_dropout=config.train.lora.dropout,
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
    tokenizer.chat_template = _MISTRAL_TEMPLATE
    tokenizer.padding_side = "right"
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.unk_token

    collator = DataCollator(
        response_template=config.model.response_template,
        instruction_template=config.model.instruction_template,
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        max_seq_length=config.train.max_seq_length,
        dataset_num_proc=16,
        train_dataset=dataset_from_file(config.data.train_file),
        eval_dataset=dataset_from_file(config.data.eval_file, config.data.eval_size),
        formatting_func=lambda: None,
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
            remove_unused_columns=False,
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

    if trainer.state.is_world_process_zero:
        wandb.init(
            project=config.wandb.project,
            notes=config.wandb.notes,
            config=config.to_dict(),
            save_code=True,
        )
    trainer.evaluate()
    trainer.train()  # type: ignore

    # Save the final model
    trainer.save_model(str(Path(config.output_dir) / "final"))


if __name__ == "__main__":
    app.run(main)
