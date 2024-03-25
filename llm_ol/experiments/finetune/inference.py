import asyncio

from absl import app, flags, logging
from openai import AsyncOpenAI

from llm_ol.experiments.finetune.templates import PROMPT_TEMPLATE

FLAGS = flags.FLAGS
# flags.DEFINE_string("graph_file", None, "Path to the graph file", required=True)
# flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
# flags.DEFINE_integer("max_depth", None, "Maximum depth of the graph")


async def query(client: AsyncOpenAI, title: str, abstract: str, t: float = 0) -> str:
    completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.render(title=title, abstract=abstract),
            }
        ],
        model="gpt-3.5-turbo",
        temperature=t,
    )
    out = completion.choices[0].message.content
    assert isinstance(out, str)
    return out


async def main(_):
    client = AsyncOpenAI(
        base_url="http://localhost:8080/v1",
        api_key="no-key-required",
    )

    result = await query(
        client,
        "Life on Mars",
        """The possibility of life on Mars is a subject of interest in astrobiology due to the planet's proximity and similarities to Earth. To date, no proof of past or present life has been found on Mars. Cumulative evidence suggests that during the ancient Noachian time period, the surface environment of Mars had liquid water and may have been habitable for microorganisms, but habitable conditions do not necessarily indicate life.[1][2]
Scientific searches for evidence of life began in the 19th century and continue today via telescopic investigations and deployed probes, searching for water, chemical biosignatures in the soil and rocks at the planet's surface, and biomarker gases in the atmosphere.[3]
Mars is of particular interest for the study of the origins of life because of its similarity to the early Earth. This is especially true since Mars has a cold climate and lacks plate tectonics or continental drift, so it has remained almost unchanged since the end of the Hesperian period. At least two-thirds of Mars's surface is more than 3.5 billion years old, and it could have been habitable 4.48 billion years ago, 500 million years before the earliest known Earth lifeforms;[4] Mars may thus hold the best record of the prebiotic conditions leading to life, even if life does not or has never existed there.[5][6]
Following the confirmation of the past existence of surface liquid water, the Curiosity, Perseverance and Opportunity rovers started searching for evidence of past life, including a past biosphere based on autotrophic, chemotrophic, or chemolithoautotrophic microorganisms, as well as ancient water, including fluvio-lacustrine environments (plains related to ancient rivers or lakes) that may have been habitable.[7][8][9][10] The search for evidence of habitability, taphonomy (related to fossils), and organic compounds on Mars is now a primary objective for space agencies. """,
        t=0.5,
    )

    logging.info(result)


if __name__ == "__main__":
    app.run(lambda _: asyncio.run(main(_)))
