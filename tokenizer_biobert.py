from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("./").glob("./*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, min_frequency=2,  special_tokens = [
   "[PAD]",
   "[UNK]",
   "[CLS]",
   "[SEP]",
   "[MASK]",
   ]
)

# Save files to disk
tokenizer.save_model("./bio-bert/")


