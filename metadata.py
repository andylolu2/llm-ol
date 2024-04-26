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

    @property
    def best_hp(self) -> dict[str, Any]:
        best_score, best_abs_percentile, best_rel_percentile = float("-inf"), None, None
        with open(self.eval_hp_result) as f:
            for line in f:
                item = json.loads(line)
                if item["graph_similarity"] > best_score:
                    best_score = item["graph_similarity"]
                    best_abs_percentile = item["absolute_percentile"]
                    best_rel_percentile = item["relative_percentile"]
        return {
            "absolute_percentile": best_abs_percentile,
            "relative_percentile": best_rel_percentile,
        }


class PromptingExperiment(Experiment):
    k_shot: int


class FinetuneExperiment(Experiment):
    step: Any
    reweighted: bool


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
    else:
        experiments = all_experiments

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
]

hearst_experiments = [
    Experiment(
        name="Hearst",
        dataset="wikipedia/v2",
        eval_output="out/experiments/hearst/v2/graph.json",
        test_output="out/experiments/hearst/v2/test/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/hearst/v2/eval/hp_search.jsonl",
        test_hp_result="out/experiments/hearst/v2/test/hp_search.jsonl",
    ),
]

rebel_experiments = [
    Experiment(
        name="Rebel",
        dataset="wikipedia/v2",
        eval_output="out/experiments/rebel/v1/eval/graph.json",
        test_output="out/experiments/rebel/v1/test/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/rebel/v1/eval/hp_search.jsonl",
        test_hp_result="out/experiments/rebel/v1/test/hp_search.jsonl",
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
        # test_hp_result="out/experiments/prompting/v7/test/hp_search.jsonl",
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
        # test_hp_result="out/experiments/prompting/v5/test/hp_search.jsonl",
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
        # test_hp_result="out/experiments/prompting/v6/test/hp_search.jsonl",
    ),
]

finetune_experiments = [
    FinetuneExperiment(
        name="Finetune step 5000",
        dataset="wikipedia/v2",
        step=5000,
        reweighted=False,
        eval_output="out/experiments/finetune/v4/5000/eval/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
    ),
    FinetuneExperiment(
        name="Finetune step 10000",
        dataset="wikipedia/v2",
        step=10000,
        reweighted=False,
        eval_output="out/experiments/finetune/v4/10000/eval/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
    ),
    FinetuneExperiment(
        name="Finetune step 16500",
        dataset="wikipedia/v2",
        step=16500,
        reweighted=False,
        eval_output="out/experiments/finetune/v4/16500/eval/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/finetune/v4/16500/eval/hp_search.jsonl",
    ),
    FinetuneExperiment(
        name="Finetune",
        dataset="wikipedia/v2",
        step="final",
        reweighted=False,
        test_output="out/experiments/finetune/v4/final/test/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/finetune/v4/final/eval/hp_search.jsonl",
        test_hp_result="out/experiments/finetune/v4/final/test/hp_search.jsonl",
    ),
    FinetuneExperiment(
        name="Finetune weighted step 5000",
        dataset="wikipedia/v2",
        step=5000,
        reweighted=True,
        eval_output="out/experiments/finetune/v6/5000/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
    ),
    FinetuneExperiment(
        name="Finetune weighted step 10000",
        dataset="wikipedia/v2",
        step=10000,
        reweighted=True,
        eval_output="out/experiments/finetune/v6/10000/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
    ),
    FinetuneExperiment(
        name="Finetune weighted step 16500",
        dataset="wikipedia/v2",
        step=16500,
        reweighted=True,
        eval_output="out/experiments/finetune/v6/16500/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
    ),
    FinetuneExperiment(
        name="Finetune weighted step 10000",
        dataset="wikipedia/v2",
        step=10000,
        reweighted=True,
        eval_output="out/experiments/finetune/v8/10000/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        version=2,
    ),
    FinetuneExperiment(
        name="Finetune weighted step 20000",
        dataset="wikipedia/v2",
        step=20000,
        reweighted=True,
        eval_output="out/experiments/finetune/v8/20000/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        version=2,
    ),
    FinetuneExperiment(
        name="Finetune weighted step 30000",
        dataset="wikipedia/v2",
        step=30000,
        reweighted=True,
        eval_output="out/experiments/finetune/v8/30000/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        version=2,
    ),
    FinetuneExperiment(
        name="Finetune weighted step 5000",
        dataset="wikipedia/v2",
        step=5000,
        reweighted=True,
        eval_output="out/experiments/finetune/v9/5000/eval/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/finetune/v9/5000/eval/hp_search.jsonl",
        version=3,
    ),
    FinetuneExperiment(
        name="Finetune weighted step 10000",
        dataset="wikipedia/v2",
        step=10000,
        reweighted=True,
        eval_output="out/experiments/finetune/v9/10000/eval/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/finetune/v9/10000/eval/hp_search.jsonl",
        version=3,
    ),
    FinetuneExperiment(
        name="Finetune weighted step 15000",
        dataset="wikipedia/v2",
        step=15000,
        reweighted=True,
        eval_output="out/experiments/finetune/v9/15000/eval/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/finetune/v9/15000/eval/hp_search.jsonl",
        version=3,
    ),
    FinetuneExperiment(
        name="Finetune reweighted",
        dataset="wikipedia/v2",
        step="final",
        reweighted=True,
        test_output="out/experiments/finetune/v9/final/test/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        test_hp_result="out/experiments/finetune/v9/final/test/hp_search.jsonl",
        version=3,
    ),
    FinetuneExperiment(
        name="Finetune masked",
        dataset="wikipedia/v2",
        step="final",
        reweighted=True,
        eval_output="out/experiments/finetune/v10/final/test/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/finetune/v10/final/eval/hp_search.jsonl",
        version=4,
    ),
    FinetuneExperiment(
        name="Finetune arxiv",
        dataset="arxiv/v2",
        step=192,
        reweighted=False,
        eval_output="out/experiments/finetune/arxiv/v2/192/eval/graph.json",
        test_output="out/experiments/finetune/arxiv/v2/192/test/graph.json",
        train_input="out/data/arxiv/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/arxiv/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/arxiv/v2/train_test_split/test_graph.json",
        eval_hp_result="out/experiments/finetune/arxiv/v2/192/eval/hp_search.jsonl",
    ),
    FinetuneExperiment(
        name="Finetune arxiv reweighted",
        dataset="arxiv/v2",
        step=320,
        reweighted=True,
        test_output="out/experiments/finetune/arxiv/v1/320/test/graph.json",
        train_input="out/data/arxiv/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/arxiv/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/arxiv/v2/train_test_split/test_graph.json",
    ),
    FinetuneExperiment(
        name="Finetune arxiv masked",
        dataset="arxiv/v2",
        step=288,
        reweighted=True,
        test_output="out/experiments/finetune/arxiv/v3/288/test/graph.json",
        train_input="out/data/arxiv/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/arxiv/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/arxiv/v2/train_test_split/test_graph.json",
        version=2,
    ),
]

all_experiments = (
    memorisation_experiments
    + hearst_experiments
    + rebel_experiments
    + prompting_experiments
    + finetune_experiments
)
