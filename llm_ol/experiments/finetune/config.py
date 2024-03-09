from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    config.seed = 0
    config.output_dir = config_dict.placeholder(str)
    config.cache_dir = "out/models"
    config.wandb_project = "llm-ol"

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
        steps=1000,
        warmup_steps=100,
        learning_rate=1e-5,
        logging_steps=10,
        grad_acc_steps=8,
        batch_size=1,
        max_seq_length=4096,
    )

    config.eval = dict(
        eval_steps=100,
        batch_size=1,
    )

    return config
