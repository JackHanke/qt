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

qt is a dense MHA decoder-only transformer with [PoPE](https://arxiv.org/abs/2509.10534) position embeddings.

[RMSNorm](https://arxiv.org/abs/1910.07467)

```
Vocab Size: 10,001
Parameters: 1.018B
    Embedding: 
    Non-embedding: 
d_model = 1792
ffw_size = 7168
kv_size = 128
n_heads = 14
n_layers = 23
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

## TODOs

- TODO low precision model, .to(dtype=torch.bfloat16) adamw_bfloat16
- gradient accumulation, 
- warmup, cosine scheduler, 
- gradient clipping
- checkpointing ... when?

## Resources
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)

