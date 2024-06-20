from pathlib import Path

from huggingface_hub import HfApi

DATASETS = {
    "wikipedia": {
        "folder_path": "out/data/wikipedia",
        "revisions": ["dev", "v1", "v2"],
        "repo_id": "andylolu24/wiki-ol",
    },
    "arxiv": {
        "folder_path": "out/data/arxiv",
        "revisions": ["dev", "v1", "v2"],
        "repo_id": "andylolu24/arxiv-ol",
    },
}

if __name__ == "__main__":
    api = HfApi()

    for _, dataset in DATASETS.items():
        for revision in dataset["revisions"]:
            folder_path = Path(dataset["folder_path"], revision)
            api.create_branch(
                dataset["repo_id"], branch=revision, repo_type="dataset", exist_ok=True
            )
            api.upload_folder(
                folder_path=folder_path,
                repo_id=dataset["repo_id"],
                revision=revision,
                repo_type="dataset",
            )
