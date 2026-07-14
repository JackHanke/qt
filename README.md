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


### Posttraining

For posttraining, I use the following datasets:
- For dialogue tuning, I use:
    - [SAMsum](https://huggingface.co/datasets/knkarthick/samsum)
    - [dialogsum](https://huggingface.co/datasets/knkarthick/dialogsum/viewer/default/train?row=0)
- For instruction tuning, I use:
    - [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
    - [norobots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
- A (very small) custom dataset of example assistant conversations I wrote myself.

## Tokenizer

Custom HuggingFace tokenizer trained on uncased english with a vocab_size of 10,001, stored at `data/tokenizer.json`.

## Resources
- [HuggingFace's Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#introduction)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

