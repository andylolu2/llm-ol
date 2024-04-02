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

from llm_ol.experiments.llm.templates import (
    _MISTRAL_TEMPLATE,
    PROMPT_TEMPLATE,
    RESPONSE_TEMPLATE,
)
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
        for i, prompt in enumerate(prompts):
            [sample] = model.generate(
                inputs=prompt["input_ids"].unsqueeze(0),
                attention_mask=prompt["attention_mask"].unsqueeze(0),
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                max_new_tokens=1024,
                use_cache=True,
            )
            sample = {
                "prompt": tokenizer.decode(prompt["input_ids"]),
                "completion": tokenizer.decode(sample[len(prompt["input_ids"]) :]),
                "target": tokenizer.decode(
                    prompt["target_ids"],
                    skip_special_tokens=True,  # Remove pad tokens
                ),
            }
            logging.info("Sample %d: %s", i, sample)
            samples.append(sample)

        if len(samples) > 0:
            table = wandb.Table(
                columns=list(samples[0].keys()),
                data=[list(s.values()) for s in samples],
            )
            wandb.log({"eval/samples": table}, step=state.global_step + 1)


def dataset_from_file(
    data_file: str | Path, size: int | None = None, seed: int = 0
) -> Dataset:
    dataset = load_dataset("json", data_files=str(data_file), split="train")
    assert isinstance(dataset, Dataset)
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))

    def make_messages(examples: dict[str, list]) -> dict[str, list]:
        outputs = []
        for title, abstract, paths in zip(
            examples["title"], examples["abstract"], examples["paths"]
        ):
            prompt = PROMPT_TEMPLATE.render(title=title, abstract=abstract)
            response = RESPONSE_TEMPLATE.render(paths=paths)
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            outputs.append(messages)
        return {"messages": outputs}

    dataset = dataset.map(make_messages, batched=True, num_proc=16)
    return dataset


def main(_):
    config = FLAGS.config
    logging.info("Config:\n%s", config)
    setup_logging(config.output_dir, "main")

    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        use_cache=False,
        device_map={"": device_string},
        torch_dtype="auto",
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

    collator = DataCollatorForCompletionOnlyLM(
        response_template=config.model.response_template,
        instruction_template=config.model.instruction_template,
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        max_seq_length=config.train.max_seq_length,
        dataset_num_proc=16,
        train_dataset=dataset_from_file(config.data.train_file),
        eval_dataset=dataset_from_file(config.data.eval_file, config.data.eval_size),
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
