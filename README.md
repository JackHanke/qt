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

qt-1B (pronounced "cutie") is a 1 billion parameter from-scratch uncased english-only language model.

## Model



## Data

### Pretraining

For pretraining, I use the [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset, which is in total 1.3T tokens that is easy to filter to english only
    - The pretraining dataset is a 24.1B token subset of the above dataset, formatted in groups of 2.15GB parquet files each containg ~754M tokens each

## Tokenizer


