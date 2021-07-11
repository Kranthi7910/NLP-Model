# NLP-Model

Training a BERT model with a dataset made from the text corpora developed by collecting research papers published in public oncology journals from last 10 years. This BERT based Bio Medical Model (BMLM) is among the very few NLP models that have been trained from scratch using raw text from the Radiation Oncology domain. Another pre-existing model [BioBERT](https://huggingface.co/dmis-lab/biobert-v1.1) is trained on the same dataset so BMLM can be compared with it. The models were trained as described in a [huggingface notebook](https://github.com/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb) with Masked Language Modeling as the downstream tassk. Accuracy metrics like Top-1 accuracy and Top-5 accuracy are used to compare the efficiencies of both models with a test dataset of over 300 sentences picked from the same domain. 

We train our tokenizers ([tokenizer_BMLM.py](./tokenizer_BMLM.py) and [tokenizer_biobert.py](./tokenizer_biobert.py)) with a byte-level Byte-pair encoding tokenizer with the same special tokens as [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html). The command `python3 train_BMLM.py` or `python3 train_biobert.py` will start training the respective models on our biomedical dataset. Once we are done with training, we can start running commands `python3 inference_BMLM.py` and `python3 inference_biobert.py` while changing the test sentence in [inference_BMLM.py](./inference_BMLM.py) and [inference_biobert.py](./inference_biobert.py) each time. The [utf8_encoder.py](utf8_encoder.py) is used to convert all the .txt data in the UTF-8 format as some tokenizers and models only support the UTF-8 format.

# Requirements

* Python3
* [Transformers](https://github.com/huggingface/transformers)
* PyTorch
* Cuda 11.3
