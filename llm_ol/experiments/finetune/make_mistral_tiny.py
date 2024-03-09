"""Create a dummy versio of Mistral for testing"""

from pathlib import Path

from absl import app, flags
from transformers import AutoTokenizer, MistralConfig, MistralForCausalLM

FLAGS = flags.FLAGS
flags.DEFINE_string("output_dir", "out/models/mistral-tiny", "Output directory")


def main(_):
    config = MistralConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = MistralForCausalLM(config)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    app.run(main)
