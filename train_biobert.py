from transformers import RobertaTokenizerFast, RobertaForMaskedLM, RobertaConfig
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerFast
from transformers import  AutoConfig, AutoTokenizer, AutoModelForMaskedLM



config = AutoConfig.from_pretrained('./bio-bert/config.json')

#recreate the tokenizer in transformers
tokenizer = RobertaTokenizerFast.from_pretrained("./bio-bert/", max_len=512)

model = AutoModelForMaskedLM.from_config(config=config)

print(model.num_parameters())

#building the training dataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./oscar.eo.txt",
    block_size=128,
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

#initialize the trainer and start training
training_args = TrainingArguments(
    output_dir="./bio-bert",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

#save the model
trainer.save_model("./bio-bert")
