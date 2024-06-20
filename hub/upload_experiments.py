from pathlib import Path

from huggingface_hub import HfApi

EXPERIMENTS = {
    "memorisation": {},
    "toy": {},
    "hearst": {},
    "rebel": {},
    "prompting": {},
    "llm": {},
    "finetune": {
        "allow_patterns": [
            "arxiv/v2/**",
            "arxiv/v3/**",
            "arxiv/v4/**",
            "arxiv/v5/**",
            "v4/**",
            "v10/**",
        ]
    },
}

if __name__ == "__main__":
    api = HfApi()

    for experiment_name, kwargs in EXPERIMENTS.items():
        api.upload_folder(
            folder_path=Path("out", "experiments", experiment_name),
            path_in_repo=experiment_name,
            repo_id="andylolu24/llm-ol-experiments",
            repo_type="model",
            multi_commits=True,
            multi_commits_verbose=True,
            **kwargs,
        )
