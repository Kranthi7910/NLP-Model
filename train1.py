from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


tokenizer = ByteLevelBPETokenizer(
    "./bio-bert/vocab.json",
    "./bio-bert/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ("[CLS]", tokenizer.token_to_id("[CLS]"))
)
tokenizer.enable_truncation(max_length=512)

print(
    tokenizer.encode("hello world").tokens
)