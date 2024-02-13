import guidance

from llm_ol.utils import PathLike


def load_mistral_instruct(model_path: PathLike, n_threads: int | None = None, **kwargs):
    return guidance.models.MistralInstruct(
        model_path, n_threads=n_threads, n_threads_batch=n_threads, n_ctx=4096, **kwargs
    )
