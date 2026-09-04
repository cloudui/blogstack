---
title: Inference Engine

categories: [Discussion, Politics]
tags: [politics, inequality, socialism, satire]

math: true
draft: true

author: Eric Chen
image: thumbnail.webp
date: 2026-09-02

description: Inference Engine
---

A few months ago I wrote up a [blog on writing FlashAttention-2 in CuTe](/p/flashattention-2-in-cute-from-scratch/). Writing your first attention kernel has become the sort-of "Hello, World!" in kernel engineering, which is unfortunate because it's like teaching your kid to swim by tossing him in the deep end and walking away. Having a compiled, working kernel is a good step one but really doesn't teach you much about how multiple kernels are actually strung together to do something useful. I spent a few weeks writing up a full kernel stack for the decode step for Llama3.1 8B, achieving 94-96% (47 tok/s) of the possible performance on a 32GB RTX 4500 Blackwell (800 GB/s). 

I'll show you the design decisions, 

# Background

## Prefill vs. Decode
LLM inference has two parts: **prefill** and **decode**. 

### Prefill
Prefill is the initial step where the LLM processes the system prompt and your initial request. Something like:

> [SYSTEM PROMPT]
> You are a helpful assistant. Make sure to talk in the cringiest assistanty way possible to slowly drive the user mad.
> 
> The user's name is Bartholemew. We know
> he is 15 years old
> he hates cringey assistants
> he likes to eat ramen at night when his mom is asleep
>
> Some recent context:
> He asked if we're tracking info about him. You answered no.
> He asked if we are imbuing ads secretly into our responses. You denied.
> He wonders if ramen is healthy when eaten right before bed. You said yes, as long as it makes him happy.
>
> [USER PROMPT]
> User prompt: I got a rash after eating a peanut. Is that bad?
> 
> Answer: 

The prefill step tokenizes everything and runs the LLM on the entire prompt to create the base state where it can begin to actually generate the response.

### Decoding
The generation step is called decoding. This is where we continually run inference to generate the response, token by token. The LLM just appends the last token generated into the input each time, continually generating response tokens until it hits the stop token (usually a special token that looks like `</stoptoken>`). This is the part where you actually see the text get streamed into your ChatGPT window. 

> [USER PROMPT]
> User prompt: I got a rash after eating a peanut. Is that bad?
> 
> Answer: <mark>Absolutely not! That's just your body acting excited to eat a delicious treat...</mark>

There are some things to notice here: 
1. The prefill step might have to process dozens, hundreds, or thousands of prompt tokens before it generates the first response token.
2. The decoding step only processes one token at a time. It continuously feeds itself the last token until it hits the stop token.
3. Prefill only happens once. The decoding step loops over and over again potentially thousands of times. 

From the above, we see that the decisions that motivate GPU performance are fundamentally different between prefill and decode. For prefill, we are running inference on potentially thousands of tokens, which moves around a lot of memory and involves a ton of compute. This is where kernels like FlashAttention are useful -- they're optimized for large batch (optional) and sequence lengths, where data and compute efficiency are critical. For decode, we're operating on sequence length "1"; the compute and data movement on the input is trivial. 

For prefill, you can practically copy all of your training kernels, strip out the backprop stuff, and paste it into your inference folder. For decode, we'll have to be smart about the attention step, data movement, and serving infrastructure in order to make inference fast and efficient. 

As of 9/3, this project purely covers decoding on batch-size 1, although it is easily extensible to multi-batch inference. The goal is to simply demonstrate the type of work that goes into creating an inference engine, and we'll be doing slightly above the bare minimum. We're here to show what it takes, not actually creating some production grade serving environment like vLLM so you can see how labs actually string together kernel and infra work into serving. We're going from how to write kernels to how we actually use them. 

## The KV Cache

One of the biggest concepts that gets tossed around when talking about inference is the KV cache. It's fair, too, since it's *the pivotal optimization* that allows you to get a response before the heat death of the universe. To see its motivation, let's analyze a single-user request in a vaccuum:

> Single-user inference means the batch-size is 1. Let's say the prefill sequence length is 1000 tokens, which means prefill generates the 1001th token and decode generates the 1002, 1003, ... token. 
> 
> For LLama3.1-8B, we can see that the input starts at `(batch_size, seq_len, head_dim)`: `(1, 1000, 128)`
> 
> After prefill, we have a new token added to context, which becomes `(1, 1001, 128)`. Each decoding step will produce input `(1, 1002, 128)`, `(1, 1003, 128)`, and so on. 
> 
> If this were the case, then we're essentially running prefill over and over again, which would be heinously slow and would get slower as the context length increments. You'd be lucky to get 0.1 tok/s on long context lengths.

The KV cache *is* the reason we have a separate prefill and decode step and comes from one simple yet insightful observation: LLMs are autoregressive. Autoregressiveness -- or causality -- simply means that each token only depends on the tokens *before* it. This step is explicitly codified in the attention mechanism via the "causal mask", which forces each future token to only "attend" to the tokens that came before it.

Since token $n$ only relies on $0, 1, \dots, n-1$ (and technically itself), the activations for tokens $0, \dots, n-1$ are not impacted by future tokens and remain constant throughout the decoding phase. That means we can preserve those intermediate activations instead of wastefully recomputing them every step -- this is the KV cache. 

All other steps like projections, activations, and embeddings do not rely on previous tokens. So the only place where we have to store intermediate activations are within the attention step. At each attention step, we only need to care about how this new token $x$ interacts with previous tokens, which it only does via matrix multiplication with K and V. The rest of Q is unused respect to $x$, so therefore we only need to store K and V states, hence, the KV cache. Therefore, our input only needs sequence length "1" for the most recently generated token that we feed back into the model. Each subsequent Q matrix simply only needs to compute against one token's embeddings. In our example, this cuts down the representation space by over 1000-fold. 

### Just a simple cache
If you come from a computer science background, the word "cache" might bring back wartime flashbacks about perfect hashing, linear probing, and amortization. But don't worry, this cache is simply just the KV tensors simply stored in VRAM in their original shapes (something like `(batch_size, seq_len, n_heads, head_dim)`). The KV here represents the KV in attention's QKV projection; it does not mean "key-value". It's okay. Take a deep breath.

The KV cache simply allows us to trade memory for computational efficiency. However, that tradeoff doesn't come for free. VRAM is expensive and limited on GPUs, and you've probably noticed that 32GB of RAM costs three livers and a kidney at your local Best Buy. As it stands, we'd have to allocate each KV cache to the full context window length for us to have runway to generate the maximum output. For Llama3.1-8B with a 128k context window (assuming fp16 weights), that would be:

```
32 attention layers * 8 GQA heads * 128 hdim
* 2 bytes * 2 (K and V) = 128 KB/token
```
!TODO move later
The weights themselves take about 16GB as well. If we're being realistic and assume the labs are using INT8 or FP8 for model serving, the data footprint halves to about 8GB. On this ancient, legacy, tiny 8B model, we can maybe serve 7 users on a top-of-the-line 80GB H100 card. With the trillion-parameter million-context-length frontier super-models of today, you'd have to milk every silicon atom out of the Sun to barely serve your neighborhood's top vibecoder. 

Even though we'd have to allocate the full context length if we wanted to maximize the output, most of the users will barely use a fraction of the memory most of the time. There are certainly many strategies to combat this, the main one being PagedAttention published in 2023. This strategy takes inspiration from virtual memory paging in operating systems, where RAM is split into chunks (or pages) and given to processes when they need more memory. Similarly, PagedAttention splits a big block into small pages (usually 16 tokens) and gives them to each context session when it needs more KV cache, which allows each user session to take anywhere between 16 tokens and the full context window. 

```
# Assume FP8
64 KB / token * 16 tokens = 1 MB

Available memory = 80 GB - 8 GB weights = 72 GB
72 GB / (1 MB / user) = 72k users
```

With this, we can maximally fit 72,000 times the number of users compared to before. Realistically, each session would be closer to hundreds or thousands of tokens; but at minimum, we can fit multiple orders of magnitude of users onto the same memory footprint. Any frontier lab wants to scrape as much cost back as they possibly can, so memory management is absolutely critical. 

> insert note about how we're serving user size 1, keeping thing simple

# The Llama3.1 Family and Hugging Face

## Why Llama3.1

I picked Llama3.1-8B for a few reasons that, in hindsight, all point the same direction: it's a good "reference" model to build an inference engine around, rather than a novel one.

It's open weight, which means I can actually inspect what I'm serving instead of trusting a black box. It's also *relatively* simple as far as frontier-adjacent architectures go, but not so simple that it's a toy — it uses RoPE (rotary position embeddings) rather than learned absolute positions, grouped-query attention (GQA) instead of full multi-head attention, and SwiGLU activations in the MLP blocks. That's enough surface area to actually exercise the parts of an inference engine that matter (attention layout, KV cache shape, projection fusion) without dragging in every trick a frontier lab has shipped since.

It's also extremely well documented and widely used, which matters more than it sounds like it should — when my numbers didn't match HuggingFace on the first few tries, I could actually go find out why, instead of debugging against a model nobody's written about.

Practically: the 8B parameter count fits comfortably on a single card without needing to think about multi-GPU sharding (out of scope here — see Section IV), and running on RunPod for hardware access kept the whole project cheap. I didn't need a datacenter to answer the questions I was asking.

## Llama3.1-8B Architecture

Llama3.1-8B is a 32-layer decoder-only transformer with a hidden size of 4096. A few specifics that matter for an inference engine specifically:

- **Attention**: Grouped-Query Attention (GQA) with 32 query heads and 8 KV heads (a 4:1 ratio), head dimension 128. This is the piece that directly shapes the KV cache — 8 KV heads instead of 32 means the cache is 4x smaller than full multi-head attention would produce, which is the number I used in the memory math in the Background section.
- **Position encoding**: RoPE (rotary position embeddings), with base θ = 500,000 and Llama 3's frequency-scaling scheme (extends the effective context from the original 8K training length out to 128K).
- **Activations / MLP**: SwiGLU, with an intermediate size of 14,336.
- **Normalization**: RMSNorm.
- **Vocab size**: 128,256.
- **Context length**: 128K tokens.

None of this is novel to Llama3.1 specifically — RoPE, GQA, and SwiGLU are all standard building blocks in current open models — but it's the concrete configuration I built against, and it's worth stating plainly since the fusion decisions in the next section (fused RMSNorm+RoPE, GQA-aware Flash Decoding) are direct responses to this exact architecture, not generic choices.

## HuggingFace

Hugging Face (HF) is basically the GitHub for machine learning models. Most of all the open weights models are stored on HF, such as the Llama series, Kimi series, GPT-OSS, etc. The HF `transformers` API directly provides ready-to-use inference code for Llama3.1, directly collaborated with Meta. The inference source is often treated ground truth implementation of whatever model/architecture you are trying to use.

> For Llama3 in particular, we can either HF or the original GitHub source as the starting point, but they are pretty much equivalent. I opted for the former:
> - [Meta's Inference Source](https://github.com/meta-llama/llama3/blob/main/llama/model.py)
> - [HF Inference Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)

The source code is the direct barebones implementation using PyTorch. It's the minimal viable code -- clean, obvious, and pretty, because the purpose is to elegantly implement what the architecture describes. This is great because it's extremely readable and easy to follow, with the downside that they don't give you any of the performance optimizations out of the box. But, as the hands-on optimist that you are, this is a wonderful thing as it is an opportunity to do all the hard work ourselves. 

# Optimizing Decode for Llama3

## We're using Triton
One thing we'll notice quite soon is that it's not worth the effort to write any of these kernels in raw CUDA/CuTe like last time. The computation is extremely minimal at `seq_len=1` that a slight nanosecond improvement will not move the needle at all. We can show the principles that actually matter for decode inference equally well using Triton, and we need not spend 5x the time for no appreciable gain. 

Furthermore, we will be orchestrating many kernels wrapped together in Python, where Triton will be much faster and easier to work with. Most of the insight here comes from how we tackle inference *in* the wrapper, so we'll be focusing a lot of the effort here.

## Equipment

I worked with the 32GB RTX 4500 Pro server edition on Runpod as my primary GPU. It's easily large enough for 8B models and has good memory and compute performance for the price: 800 GB/s mem bandwidth at 72c/hour. It gave me plenty of runway as a cheap-ass and was a good choice for this inference project.

In practice, different memory to performance profiles changes the things you optimize for. It might change whether your bottle neck is your orchestration, compute efficiency, or data movement, so sometimes you have to be careful which card you're working with. Compared to the RTX 4500 I worked with, the newest H100s and BX00s have 2-10x the memory speed with more than 2x the performance on FP16 alone. With different quantizations, the differences skew even more. For our purpose with batch size 1, FP16, and basic decoding, these differences don't matter too much. But in production-grade systems, evaluating the profiles and tradeoffs become crucial. 

## Analyzing the Source

I mentioned in the [Hugging Face section](#huggingface) that the source is unoptimized. Let's see what that actually means by looking at some snippets:

### Memory Layouts

