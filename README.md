```
           ___        ___             
          /\  \      /\  \            
         /88\  \     \8\  \           
        /8/\8\  \     \8\  \          
        \8\~\8\__\    /88\  \         
         \8\/8/  /   /8/\8\__\        
          \88/  /   /8/  \/__/        
          /8/__/   /8/  /             
          \8\__\   \/__/              
           \/__/                                                  
```

qt (pronounced "cutie") is a 1 billion parameter hand coded, from-scratch uncased english-only language model.

## Model Card

qt is a dense GQA ALiBi/NoPE flash attn transformer. We use [RMSNorm](https://arxiv.org/abs/1910.07467) and GELU activations.

```
Vocab Size: 10,001
Parameters: 1.01B
    Embedding: 
    Non-embedding: 
d_model = 2048
ffw_size = 8196
n_heads = 32
n_heads_kv = 8
n_layers = 22
seq_len = 512
```

## Data

### Pretraining

For pretraining, I source my data from the [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset
- The pretraining dataset is a ~21.5B token subset of the above dataset, formatted in groups of 2.15GB parquet files each containg ~754M tokens each.

The learning rate schedule is [Warmup Stable Decay](https://arxiv.org/abs/2404.06395)


### Finetuning

TODO

## Tokenizer

Custom HuggingFace tokenizer trained on uncased english with a vocab_size of 10,001, stored at `data/tokenizer.json`.

## Resources
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)

