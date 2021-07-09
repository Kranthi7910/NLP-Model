from pathlib import Path
from tokenizers import (ByteLevelBPETokenizer,
                        CharBPETokenizer,
                        SentencePieceBPETokenizer,
                        BertWordPieceTokenizer)
from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("./").glob("./*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

#tokenizer.add_special_tokens([ "<pad>", "<mask>" ])

# Customize training
tokenizer.train(files=paths, min_frequency=2,

)

# Save files to disk
tokenizer.save_model("./bio-bert/")
#tokenizer.save("./bio-bert/vocab.txt")

# special_tokens = [
#   "[PAD]",
#   "[UNK]",
#   "[CLS]",
#   "[SEP]",
#   "[MASK]",
#   ]