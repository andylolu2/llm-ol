import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import dotenv
import pandas as pd
from absl import app, flags, logging
from arxiv.taxonomy.definitions import CATEGORY_ALIASES

from llm_ol.dataset.data_model import load_categories_jsonl
from llm_ol.dataset.utils.miscellaneous import batch, setup_loggging
from llm_ol.dataset.utils.rate_limit import Resource

dotenv.load_dotenv()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "output_dir", None, "Directory to save output files", required=True, short_name="o"
)

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
semantic_scholar_limit = Resource(period=timedelta(seconds=1), limit=1)


async def api_request(session: aiohttp.ClientSession, retries: int = 3, **kwargs):
    for i in range(retries):
        try:
            await semantic_scholar_limit.acquire()
            async with session.post(SEMANTIC_SCHOLAR_API_URL, **kwargs) as response:
                if response.status == 429:  # Too many requests
                    raise RuntimeError("Too many requests")
                result = await response.json()
                return result
        except Exception as e:
            if i == retries - 1:
                raise e
            else:
                logging.error("Request failed: %s. %d retries left", e, retries - i - 1)
                await asyncio.sleep(2**i)
    assert False  # Unreachable


def download_arxiv(save_dir: Path):
    file_path = save_dir / "arxiv-metadata-oai-snapshot.json"
    if not file_path.exists():
        import kaggle  # Lazy import so that dotenv is loaded

        logging.info("Downloading arXiv dataset")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "Cornell-University/arxiv", path=save_dir, unzip=True
        )
    return file_path


def preprocess_text(text: str) -> str:
    lines = text.split("\n")
    lines = filter(lambda x: x != "", map(lambda x: x.strip(), lines))
    text = " ".join(lines)
    return text


def preprocess_arxiv(item: dict):
    # Get date
    # version = {"version": "v1", "created": "Mon, 1 Jan 0000 00:00:00 GMT"}
    first_version = min(item["versions"], key=lambda x: int(x["version"][1:]))
    date = datetime.strptime(first_version["created"], "%a, %d %b %Y %H:%M:%S %Z")

    return {
        "id": item["id"],
        "title": preprocess_text(item["title"]),
        "categories": item["categories"].split(" "),
        "abstract": preprocess_text(item["abstract"]),
        "date": date,
    }


def load_papers(
    file_path: Path, date_min: datetime, date_max: datetime, cache_dir: Path
):
    cache_file = cache_dir / f"papers_{date_min}_{date_max}.jsonl"
    papers = []
    if cache_file.exists():
        logging.info("Loading papers from %s", cache_file)
        with open(cache_file, "r") as f:
            for line in f:
                papers.append(json.loads(line))
    else:
        logging.info("Building papers from %s", file_path)
        with open(file_path, "r") as f_in, open(cache_file, "w") as f_out:
            for line in f_in:
                item = json.loads(line)
                item = preprocess_arxiv(item)
                if date_min <= item["date"] <= date_max:
                    f_out.write(json.dumps(item, default=str) + "\n")
                    papers.append(item)

    logging.info("Loaded %d papers", len(papers))
    return papers


async def get_papers_with_citations(papers: list[dict], cache_dir: Path):
    cache_file = cache_dir / "papers_with_citations.jsonl"

    papers_with_citations = []
    if cache_file.exists():
        logging.info("Loading citations from %s", cache_file)
        with open(cache_file, "r") as f:
            for line in f:
                paper = json.loads(line)
                papers_with_citations.append(paper)
    else:  # Get citation data
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            with open(cache_file, "w") as f:

                async def coro(paper_batch):
                    paper_ids = [f"ARXIV:{paper['id']}" for paper in paper_batch]
                    response = await api_request(
                        session,
                        params={"fields": "citationCount"},
                        json={"ids": paper_ids},
                        headers={"x-api-key": os.getenv("SEMANTIC_SCHOLAR_API_KEY")},
                    )
                    logging.debug("Response: %s", response)
                    for paper, result in zip(paper_batch, response):
                        if result is None:
                            logging.info(
                                "No citation data for %s: %s",
                                paper["id"],
                                paper["title"],
                            )
                            continue
                        item = {"citation_count": result["citationCount"], **paper}
                        f.write(json.dumps(item, default=str) + "\n")
                        papers_with_citations.append(item)

                coros = [coro(paper_batch) for paper_batch in batch(papers, 500)]
                await asyncio.gather(*coros)

    logging.info("Loaded %d papers with citations", len(papers_with_citations))
    return papers_with_citations


async def main(_):
    """
    The arXiv taxonomy is a hierarchical classification of arXiv papers into three
    levels: groups -> archives -> categories.

    The taxonomy is hard-coded in the arxiv.taxonomy.definitions module.
    """

    # Set up
    out_dir = Path(FLAGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_loggging(out_dir, "content_builder")
    asyncio.create_task(semantic_scholar_limit.replenish())

    # Get arXiv data
    arxiv_file = download_arxiv(Path("out", "raw"))
    papers = load_papers(
        arxiv_file, datetime(2020, 1, 1), datetime(2022, 12, 31), out_dir
    )
    papers_with_citations = await get_papers_with_citations(papers, out_dir)

    # Filter out papers: Pick top N most cited papers in each category
    N = 50

    categories = load_categories_jsonl(out_dir)
    leaf_categories = set()
    for category in categories:
        id_ = category.id_
        assert isinstance(id_, str)
        if id_.startswith("category-"):
            leaf_categories.add(id_.removeprefix("category-"))

    category_to_papers = {category: [] for category in leaf_categories}

    for paper in papers_with_citations:
        for category in paper["categories"]:
            if category in CATEGORY_ALIASES:  # check if it's an alias
                category = CATEGORY_ALIASES[category]

            if category not in category_to_papers:
                logging.warning("Unknown leaf category: %s", category)
                continue

            category_to_papers[category].append(paper)

    saved_papers = set()
    with open(out_dir / "papers.jsonl", "w") as f:
        for category, papers in category_to_papers.items():
            if len(papers) < N:
                logging.info("Category %s has only %d papers.", category, len(papers))
                top_papers = papers
            else:
                top_papers = sorted(
                    papers, key=lambda x: x["citation_count"], reverse=True
                )[:N]
            for paper in top_papers:
                if paper["id"] in saved_papers:
                    continue
                f.write(json.dumps(paper, default=str) + "\n")
                saved_papers.add(paper["id"])
    logging.info("Saved %d most cited papers", len(saved_papers))


if __name__ == "__main__":
    app.run(lambda _: asyncio.run(main(_)))
