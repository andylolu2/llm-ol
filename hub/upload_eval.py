from pathlib import Path

from huggingface_hub import HfApi

if __name__ == "__main__":
    api = HfApi()

    api.upload_folder(
        folder_path=Path("out", "eval"),
        path_in_repo="eval",
        repo_id="andylolu24/llm-ol-experiments",
        repo_type="model",
        multi_commits=True,
        multi_commits_verbose=True,
    )
