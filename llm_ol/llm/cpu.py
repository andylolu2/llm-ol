import guidance
import torch

from llm_ol.utils import PathLike


def load_mistral_instruct(model_path: PathLike, n_threads: int | None = None, **kwargs):
    return guidance.models.MistralInstruct(
        model_path,
        n_threads=n_threads,
        n_threads_batch=n_threads,
        n_ctx=4096,
        n_gpu_layers=-1 if torch.cuda.is_available() else 0,
        use_mlock=True,
        **kwargs
    )
