from transformers import pipeline, RobertaTokenizerFast
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

config = AutoConfig.from_pretrained('./bioBERT/config.json')

tokenizer = AutoTokenizer.from_pretrained("./bio-bert/", max_len=512)

model = AutoModelForMaskedLM.from_pretrained("./bio-bert/pytorch_model.bin", config=config)

fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)
   
test_str = "The [MASK] providers would adhere to PRT principles."

print (" Input String : " , test_str.replace("[MASK]", "____")) 

predictions = fill_mask(test_str)

print ("Predictions :")
for idx, pred in enumerate(predictions):
    print ("%d. %s" % (idx+1, pred["token_str"]))


print ("Raw output :")
print (predictions)
