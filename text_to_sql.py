"""
Text-to-SQL Generator using Hugging Face Transformers.

This script loads a pre-trained model fine-tuned on the Spider dataset
to convert natural language prompts into SQL queries.

"""

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


class TextToSQL:
    def __init__(self, model_name="tscholak/t5.1.1.small.spider"):
        """
        Initialize the tokenizer and model from Hugging Face.
        """
        print(f"Loading model: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def generate(self, prompt: str, schema_hint: str = "") -> str:
        """
        Generate SQL from a natural language prompt.

        Args:
            prompt (str): The user's natural language question.
            schema_hint (str): Optional database schema to improve accuracy.

        Returns:
            str: Generated SQL query.
        """
        input_text = f"translate English to SQL: {prompt} {schema_hint}".strip()
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(input_ids, max_length=150)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Text-to-SQL CLI")
    parser.add_argument("prompt", type=str, help="Natural language question")
    parser.add_argument("--schema", type=str, default="", help="Optional DB schema hint")

    args = parser.parse_args()

    generator = TextToSQL()
    sql_query = generator.generate(args.prompt, args.schema)

    print("\nGenerated SQL Query:\n", sql_query)
