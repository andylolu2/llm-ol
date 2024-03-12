from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    config.seed = 0
    config.output_dir = config_dict.placeholder(str)
    config.wandb = dict(
        project="llm-ol",
        notes="",
    )

    config.model = dict(
        name="mistralai/Mistral-7B-Instruct-v0.2",
        response_template=[733, 28748, 16289, 28793],  # _[/INST]
        instruction_template=[733, 16289, 28793],  # _[INST]
    )

    config.data = dict(
        file=config_dict.placeholder(str),
        eval_size=1024,
    )

    config.train = dict(
        steps=10_000,
        warmup_steps=100,
        learning_rate=1e-5,
        logging_steps=50,
        grad_acc_steps=1,
        batch_size=16,
        max_seq_length=2048,
        lora=dict(
            rank=32,
        ),
    )

    config.eval = dict(
        eval_steps=500,
        batch_size=32,
        num_generate_samples=5,
    )

    return config
