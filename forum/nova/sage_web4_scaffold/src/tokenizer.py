# Placeholder for custom tokenizer logic
def load_tokenizer(model_name="gpt2"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)
