import json
from typing import Any, Literal, overload

from pydantic import BaseModel


class Experiment(BaseModel):
    name: str
    dataset: str
    eval_output: str | None = None
    test_output: str | None = None
    train_input: str
    eval_ground_truth: str
    test_ground_truth: str
    eval_hp_result: str | None = None
    test_hp_result: str | None = None
    version: int = 1

    def best_hp(self, metric: str = "edge_soft_f1"):
        best_score, best_hp = float("-inf"), None
        with open(self.eval_hp_result) as f:
            for line in f:
                item = json.loads(line)
                if item[metric] is not None and item[metric] > best_score:
                    best_score = item[metric]
                    best_hp = item["hp"]
        if best_hp is None:
            raise ValueError(f"No best HP found for {self.name} on {self.dataset}")
        return best_hp


class PromptingExperiment(Experiment):
    k_shot: int


class FinetuneExperiment(Experiment):
    step: Any
    reweighted: bool
    transfer: bool = False


def query(**kwargs) -> Experiment:
    matches = query_multiple(**kwargs)
    if len(matches) > 0:
        return matches[0]
    else:
        raise ValueError(f"Experiment not found: {kwargs}")


@overload
def query_multiple(
    exp: Literal["memorisation", "hearst", "rebel", "all"], **kwargs
) -> list[Experiment]: ...


@overload
def query_multiple(
    exp: Literal["prompting"], **kwargs
) -> list[PromptingExperiment]: ...


@overload
def query_multiple(exp: Literal["finetune"], **kwargs) -> list[FinetuneExperiment]: ...


def query_multiple(exp: str = "all", **kwargs):
    if exp == "hearst":
        experiments = hearst_experiments
    elif exp == "rebel":
        experiments = rebel_experiments
    elif exp == "memorisation":
        experiments = memorisation_experiments
    elif exp == "prompting":
        experiments = prompting_experiments
    elif exp == "finetune":
        experiments = finetune_experiments
    elif exp == "all":
        experiments = all_experiments
    else:
        raise ValueError(f"Unknown experiment: {exp}")

    matches = []
    for experiment in experiments:
        if all(getattr(experiment, key) == value for key, value in kwargs.items()):
            matches.append(experiment)

    return matches


memorisation_experiments = [
    Experiment(
        name="Memorisation",
        dataset="wikipedia/v2",
        test_output="out/experiments/memorisation/wiki/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/memorisation/wiki/eval/hp_search.jsonl",
        test_hp_result="out/experiments/memorisation/wiki/test/hp_search.jsonl",
    ),
    Experiment(
        name="Memorisation",
        dataset="arxiv/v2",
        test_output="out/experiments/memorisation/arxiv/graph.json",
        train_input="out/data/arxiv/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/arxiv/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/arxiv/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/memorisation/arxiv/eval/hp_search.jsonl",
        test_hp_result="out/experiments/memorisation/arxiv/test/hp_search.jsonl",
    ),
]

hearst_experiments = [
    Experiment(
        name="Hearst",
        dataset="wikipedia/v2",
        eval_output="out/experiments/hearst/svd/wiki/eval/k_5/graph.json",
        test_output="out/experiments/hearst/svd/wiki/test/k_5/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/hearst/svd/wiki/eval/k_5/hp_search.jsonl",
    ),
    Experiment(
        name="Hearst",
        dataset="arxiv/v2",
        eval_output="out/experiments/hearst/svd/arxiv/eval/k_150/graph.json",
        test_output="out/experiments/hearst/svd/arxiv/test/k_150/graph.json",
        train_input="out/data/axiv/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/arxiv/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/arxiv/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/hearst/svd/arxiv/eval/k_150/hp_search.jsonl",
    ),
]

rebel_experiments = [
    Experiment(
        name="Rebel",
        dataset="wikipedia/v2",
        eval_output="out/experiments/rebel/svd/wiki/eval/k_20/graph.json",
        test_output="out/experiments/rebel/svd/wiki/test/k_20/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/rebel/svd/wiki/eval/k_20/hp_search.jsonl",
    ),
    Experiment(
        name="Rebel",
        dataset="arxiv/v2",
        eval_output="out/experiments/rebel/svd/arxiv/eval/k_100/graph.json",
        test_output="out/experiments/rebel/svd/arxiv/test/k_100/graph.json",
        train_input="out/data/arxiv/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/arxiv/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/arxiv/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/rebel/svd/arxiv/eval/k_100/hp_search.jsonl",
    ),
]

prompting_experiments = [
    PromptingExperiment(
        name="0 shot",
        dataset="wikipedia/v2",
        k_shot=0,
        eval_output="out/experiments/prompting/v7/eval/graph.json",
        test_output="out/experiments/prompting/v7/test/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/prompting/v7/eval/hp_search.jsonl",
        test_hp_result="out/experiments/prompting/v7/test/hp_search.jsonl",
    ),
    PromptingExperiment(
        name="0 shot",
        dataset="arxiv/v2",
        k_shot=0,
        eval_output="out/experiments/prompting/arxiv/v1/eval/graph.json",
        test_output="out/experiments/prompting/arxiv/v1/test/graph.json",
        train_input="out/data/arxiv/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/arxiv/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/arxiv/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/prompting/arxiv/v1/eval/hp_search.jsonl",
        test_hp_result="out/experiments/prompting/arxiv/v1/test/hp_search.jsonl",
    ),
    PromptingExperiment(
        name="1 shot",
        dataset="wikipedia/v2",
        k_shot=1,
        eval_output="out/experiments/prompting/v5/eval/graph.json",
        test_output="out/experiments/prompting/v5/test/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/prompting/v5/eval/hp_search.jsonl",
        test_hp_result="out/experiments/prompting/v5/test/hp_search.jsonl",
    ),
    PromptingExperiment(
        name="1 shot",
        dataset="arxiv/v2",
        k_shot=1,
        eval_output="out/experiments/prompting/arxiv/v2/eval/graph.json",
        test_output="out/experiments/prompting/arxiv/v2/test/graph.json",
        train_input="out/data/arxiv/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/arxiv/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/arxiv/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/prompting/arxiv/v2/eval/hp_search.jsonl",
        test_hp_result="out/experiments/prompting/arxiv/v2/test/hp_search.jsonl",
    ),
    PromptingExperiment(
        name="3 shot",
        dataset="wikipedia/v2",
        k_shot=3,
        eval_output="out/experiments/prompting/v6/eval/graph.json",
        test_output="out/experiments/prompting/v6/test/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/prompting/v6/eval/hp_search.jsonl",
        test_hp_result="out/experiments/prompting/v6/test/hp_search.jsonl",
    ),
    PromptingExperiment(
        name="3 shot",
        dataset="arxiv/v2",
        k_shot=3,
        eval_output="out/experiments/prompting/arxiv/v3/eval/graph.json",
        test_output="out/experiments/prompting/arxiv/v3/test/graph.json",
        train_input="out/data/arxiv/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/arxiv/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/arxiv/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/prompting/arxiv/v3/eval/hp_search.jsonl",
        test_hp_result="out/experiments/prompting/arxiv/v3/test/hp_search.jsonl",
    ),
]

finetune_experiments = [
    FinetuneExperiment(
        name="Finetune",
        dataset="wikipedia/v2",
        step="final",
        reweighted=False,
        eval_output="out/experiments/finetune/v4/final/eval/graph.json",
        test_output="out/experiments/finetune/v4/final/test/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/finetune/v4/final/eval/hp_search.jsonl",
        test_hp_result="out/experiments/finetune/v4/final/test/hp_search.jsonl",
    ),
    FinetuneExperiment(
        name="Finetune masked",
        dataset="wikipedia/v2",
        step="final",
        reweighted=True,
        eval_output="out/experiments/finetune/v10/final/eval/graph.json",
        test_output="out/experiments/finetune/v10/final/test/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/finetune/v10/final/eval/hp_search.jsonl",
        test_hp_result="out/experiments/finetune/v10/final/test/hp_search.jsonl",
    ),
    FinetuneExperiment(
        name="Finetune (transfer)",
        dataset="arxiv/v2",
        step=192,
        reweighted=False,
        transfer=True,
        eval_output="out/experiments/finetune/arxiv/v2/192/eval/graph.json",
        test_output="out/experiments/finetune/arxiv/v2/192/test/graph.json",
        train_input="out/data/arxiv/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/arxiv/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/arxiv/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/finetune/arxiv/v2/192/eval/hp_search.jsonl",
    ),
    FinetuneExperiment(
        name="Finetune masked (transfer)",
        dataset="arxiv/v2",
        step="final",
        reweighted=True,
        transfer=True,
        eval_output="out/experiments/finetune/arxiv/v3/288/eval/graph.json",
        test_output="out/experiments/finetune/arxiv/v3/288/test/graph.json",
        train_input="out/data/arxiv/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/arxiv/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/arxiv/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/finetune/arxiv/v3/288/eval/hp_search.jsonl",
    ),
]

all_experiments = (
    memorisation_experiments
    + hearst_experiments
    + rebel_experiments
    + prompting_experiments
    + finetune_experiments
)
