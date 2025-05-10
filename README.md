# 🧠 Text-to-SQL Generator

A simple and effective Python tool that converts natural language questions into SQL queries using a pre-trained [T5](https://huggingface.co/tscholak/t5.1.1.small.spider) model fine-tuned on the Spider text-to-SQL dataset.

## 🚀 Features

- Convert English prompts to SQL queries
- Based on Hugging Face Transformers
- CLI interface for quick usage
- Easy to extend with schema hints

## 📦 Requirements

- Python 3.7+
- [Transformers](https://pypi.org/project/transformers/)
- [Torch](https://pytorch.org/)

Install dependencies:

```bash
pip install -r requirements.txt
