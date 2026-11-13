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

qt (pronounced "cutie") is a model family of english-only language models I am hand coding from scratch.

## Model Family

- *qt 1B* was trained on my local machine, and is composed of entirely human-generated data. See the model card at `models/qt1/model_card.md`
- *qt 2B* is WIP!

## Resources
- [HuggingFace's Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#introduction)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

