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

qt (pronounced "cutie") is a 1 billion parameter from-scratch uncased english-only language model.

## Model

Parameters: `1.018B`

`d_model = 1792`

`ffw_size = 7168`

`kv_size = 128`

`n_heads = 14`

`n_layers = 23`


## Data

### Pretraining

For pretraining, I use the [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset, which is in total 1.3T tokens that is easy to filter to english only
- The pretraining dataset is a 24.1B token subset of the above dataset, formatted in groups of 2.15GB parquet files each containg ~754M tokens each

## Tokenizer

## TODOs

- TODO low precision model, .to(dtype=torch.bfloat16) adamw_bfloat16
- gradient accumulation, 
- warmup, cosine scheduler, 
- gradient clipping
- checkpointing ... when?

## Resources
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)

