import asyncio
from datetime import timedelta
from pathlib import Path

import aiohttp
import networkx as nx
from absl import logging

from llm_ol.dataset import data_model
from llm_ol.utils.rate_limit import Resource

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
ROOT_CATEGORY_ID = 7345184
ROOT_CATEGORY_NAME = "Main topic classifications"

api_limit = Resource(period=timedelta(seconds=1), limit=50)


async def api_request(session: aiohttp.ClientSession, params: dict, retries: int = 3):
    for i in range(retries):
        try:
            await api_limit.acquire()
            async with session.get(WIKIPEDIA_API_URL, params=params) as response:
                if response.status == 429:  # Too many requests
                    raise RuntimeError("Too many requests")
                result = await response.json()
                return result
        except asyncio.CancelledError as e:
            raise e
        except Exception as e:
            if i == retries - 1:
                raise e
            else:
                logging.error(
                    "Request failed: %s. %d retries left", repr(e), retries - i - 1
                )
                await asyncio.sleep(2**i)
    assert False  # Unreachable


def load_dataset(file_path: Path | str, max_depth: int | None = None):
    G = data_model.load_graph(file_path)
    if max_depth is not None:
        G = nx.ego_graph(G, ROOT_CATEGORY_ID, radius=max_depth)
    return G
