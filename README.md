# Embedding for source code
This repository implement some popular embedding model for Java Source Code. 
## Cubert Embedding
* Requirements
```
pip3 install -r cubert_requirements.txt
```
* Running 
1. Download pretrained cubert model in https://console.cloud.google.com/storage/browser/cubert/20200913_Java_Deduplicated/pre_trained_model_deduplicated__epochs_1__length_1024. 
2. Move all files to a folder call bert_model.ckpt. 
3. Create bert_config.json (in folder bert_model.ckpt) with config:
```
{
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "type_vocab_size": 2,
    "vocab_size": 50297, 
    "max_position_embeddings": 1024
}
```
4. Convert tensorflow checkpoint to pytorch checkpoint by using: 
```
transformers-cli convert --model_type bert   --tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt   --config $BERT_BASE_DIR/bert_config.json   --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin
```
5. Running: Change _VOCAB_FILE and _CHECKPOINT in cubert_embedding.py with your path and run:
```
python3 cubert_embedding.py
```
