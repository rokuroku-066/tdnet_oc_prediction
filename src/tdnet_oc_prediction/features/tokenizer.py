from transformers import AutoTokenizer

def load_tokenizer(name: str):
    return AutoTokenizer.from_pretrained(name)
