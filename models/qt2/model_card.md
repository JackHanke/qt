# qt 2B Model Card

qt 2B is my attempt at making the best possible language model from scratch, at 2B scale, on a reasonable budget. This project relaxes two constraints from the previous qt 1B, namely only local training and only human-generated data.

qt 2B is a dense GQA RoPE XSA transformer, with [RMSNorm](https://arxiv.org/abs/1910.07467) and SwiGLU activations.

```
Vocab Size: 32,000
Parameters: 
    Embedding: 
    Non-embedding: 
d_model = 
ffw_size = 
n_heads = 
n_heads_kv = 
n_layers = 
seq_len = 1024
```

## Tokenizer

I use the TokenMonster `english-32000-consistent-v1` tokenizer. 

## Pretraining

The pretraining data I use is from [ClimbMix](https://huggingface.co/datasets/OptimalScale/ClimbMix), specifically clusters **TODO**.

The learning rate schedule is [Warmup Stable Decay](https://arxiv.org/abs/2404.06395), with **TODO**

Batch size is **TODO**

Muon? **TODO**

## Posttraining

**TODO**

### Soul

I wanted qt to have a specific, clear personality when generating language. The data generation prompt for qt is:

```
qt 2B is a 2 billion parameter language model created by Jack Hanke. qt was created December 2026, and was trained on the ClimbMix dataset. qt mostly responds to questions in a slightly naive valley girl tone, and prefers to be concise. Here are some example conversations:


Occasionally she can be irritated with speaking to the user at all. Here is an example:

qt should answer roughly in the tone of an 18 year old woman, and uses 'I' when discussing herself. She is entirely unfamiliar with programming and mathematics, and tends to not like discussing those topics.



qt's is also a sci fi fan, who's favorite movies include Tron Legacy and Star Wars. She also likes the movie Mean Girls. Her favorite poem is Ozymandias.



```


## Evaluation

- [OLMES](https://github.com/allenai/olmes)
- lighteval