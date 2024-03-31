from asyncio import Semaphore
from typing import Any, Callable, TypeVar

import openai
from httpx import Limits
from openai import AsyncOpenAI
from openai.resources.chat.completions import AsyncCompletions as ChatCompletions
from openai.resources.completions import AsyncCompletions as Completions

T = TypeVar("T")


def copy_type(f: T) -> Callable[[Any], T]:
    return lambda x: x


class ParallelAsyncOpenAI:
    def __init__(self, base_urls: list[str], max_concurrent_per_client: int = 512):
        self.clients = [
            AsyncOpenAI(
                api_key="no-key-required",
                base_url=base_url,
                http_client=openai._base_client.AsyncHttpxClientWrapper(
                    base_url=base_url,
                    limits=Limits(
                        max_connections=max_concurrent_per_client,
                        max_keepalive_connections=max_concurrent_per_client // 5,
                    ),
                ),
            )
            for base_url in base_urls
        ]
        self.sem = Semaphore(max_concurrent_per_client * len(self.clients))
        self.request_idxs = set(range(max_concurrent_per_client * len(self.clients)))

    @copy_type(ChatCompletions.create)
    async def chat(self, *args, **kwargs):
        await self.sem.acquire()
        idx = self.request_idxs.pop()
        try:
            return await self.clients[idx % len(self.clients)].chat.completions.create(
                *args, **kwargs
            )
        finally:
            self.request_idxs.add(idx)
            self.sem.release()

    @copy_type(Completions.create)
    async def completions(self, *args, **kwargs):
        await self.sem.acquire()
        idx = self.request_idxs.pop()
        try:
            return await self.clients[idx % len(self.clients)].completions.create(
                *args, **kwargs
            )
        finally:
            self.request_idxs.add(idx)
            self.sem.release()
