from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./julien-c/EsperBERTo-small",
    tokenizer="./julien-c/EsperBERTo-small"
)

fill_mask("La suno <mask>.")