from transformers import pipeline

#fill-mask pipeline returns a list of most probable filled sequences, with their probabilities
fill_mask = pipeline(
    "fill-mask",
    model="./EsperBERTo",
    tokenizer="./EsperBERTo"
)
 
#random test sentence with a masked token
test_str = "The <mask> providers would adhere to PRT principles."

#print results
print (" Input String : " , test_str.replace("<mask>", "____")) 

predictions = fill_mask(test_str)

print ("Predictions :")
for idx, pred in enumerate(predictions):
    print ("%d. %s" % (idx+1, pred["token_str"]))


print ("Raw output :")
print (predictions)
