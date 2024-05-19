# End-to-End Ontology Learning with Large Language Models

This is the source code of 'End-to-End Ontology Learning with Large Language Models' paper.

## Repository structure

```.
├── llm_ol              # Main codebase
│   ├── dataset         # Code for building Wikipedia and arXiv datasets
│   ├── eval            # Evaluation metrics
│   ├── experiments     # Main experiments (OLLM + baselines)
│   ├── llm             # utils for LLM
│   └── utils           # utils for experiments
└── runs                # Bash script entrypoints
```

## Installation

This project depends on graph-tools, which is not pip-installable. It is recommended to use conda for installation (https://graph-tool.skewed.de/installation.html). Once you have installed graph-tools, you can install the rest of the dependencies using pip:
```bash
pip install -r requirements.txt
```

## Running experiments

### Data preparation

Wikipedia:
```bash
# Traverse the categorisation tree
./runs/dataset/wiki/build_categories.sh
# Collect the pages in the categories
./runs/dataset/wiki/build_pages.sh
# Export the data into a graph
./runs/dataset/wiki/export.sh
# Split the data into train, validation and test graphs
./runs/dataset/wiki/train_test_split.sh
```

arXiv:
```bash
# Download the arXiv taxonomy
./runs/dataset/arxiv/build_categories.sh
# Download the selected arXiv papers
./runs/dataset/arxiv/build_pages.sh
# Export the data into a graph
./runs/dataset/arxiv/export.sh
# Split the data into train, validation and test graphs
./runs/dataset/arxiv/train_test_split.sh
```

### Reproduce results

Wikipedia:
```bash
# Build the training dataset
./runs/llm/build_dataset.sh
# Run the main experiment
./runs/finetune/main_weighted.sh
# Run the inference
./runs/finetune/inference.sh
# Export the model into a raw graph
./runs/finetune/export.sh         
# Hyperparameter search
./runs/finetune/eval.sh
```

arXiv: (edit the bash scripts to point to the correct dataset)
```bash
# Build the training dataset
./runs/llm/build_dataset.sh
# Run the main experiment
./runs/finetune/main_arxiv.sh
# Run the inference
./runs/finetune/inference.sh
# Export the model into a raw graph
./runs/finetune/export.sh         
# Hyperparameter search
./runs/finetune/eval.sh
```