import torch
from tensor2tensor.data_generators import text_encoder
from cubert import tokenizer
from pytorch_pretrained_bert.modeling import BertModel
_VOCAB_FILE= "/Users/Elephant/Research/Automated_Program_Repair/Code/ML4Repair/Data/Cubert_pretrain/vocab/20201018_Java_Deduplicated-github_java_vocabulary.txt"
_CHECKPOINT= "/Users/Elephant/Research/Automated_Program_Repair/Code/ML4Repair/Data/Cubert_pretrain/pretrain"

code = "if (i == -1){ return 0; }"

#Tokenizer
tokenizer = tokenizer.BertTokenizer(vocab_file=_VOCAB_FILE)
tokenized_code = tokenizer.tokenize(code)

#Convert to tensors
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_code)
tokens_tensor = torch.tensor([indexed_tokens])
print(tokens_tensor)

#Load model
model = BertModel.from_pretrained(_CHECKPOINT)
model.eval()

#Embedding
emb = model(tokens_tensor)
print(emb)
