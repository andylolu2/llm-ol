from typing import Literal, overload

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


class PromptingExperiment(Experiment):
    k_shot: int


class FinetuneExperiment(Experiment):
    step: int
    reweighted: bool


def query(**kwargs) -> Experiment:
    matches = query_multiple(**kwargs)
    if len(matches) > 0:
        return matches[0]
    else:
        raise ValueError(f"Experiment not found: {kwargs}")


@overload
def query_multiple(exp: Literal["hearst"], **kwargs) -> list[Experiment]: ...


@overload
def query_multiple(
    exp: Literal["prompting"], **kwargs
) -> list[PromptingExperiment]: ...


@overload
def query_multiple(exp: Literal["finetune"], **kwargs) -> list[FinetuneExperiment]: ...


def query_multiple(exp: str = "all", **kwargs):
    if exp == "hearst":
        experiments = hearst_experiments
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
        name="1 shot",
        dataset="wikipedia/v2",
        k_shot=1,
        eval_output="out/experiments/prompting/v5/eval/graph.json",
        test_output="out/experiments/prompting/v5/test/graph.json",
        train_input="out/data/wikipedia/v2/train_eval_split/train_graph.json",
        eval_ground_truth="out/data/wikipedia/v2/train_eval_split/test_graph.json",
        test_ground_truth="out/data/wikipedia/v2/train_test_split/test_graph.json",
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
]

all_experiments = prompting_experiments + finetune_experiments