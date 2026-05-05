---
title: CUDA

categories: [Discussion, Politics]
tags: [politics, inequality, socialism, satire]
# mathjax: true
math: true
draft: true

author: Eric Chen
image: thumbnail.webp
date: 2026-05-04

description: FlashAttn-v2
---
You may have noticed your energy bill rising month over month for the past few years. An unfortunate trend you might have also noticed in your age and blood pressure, and a remarkably similar but sort of inverse trend in your bank account and your health. You probably wail to your friends over dinner about how you have to pay a wad of franklins simply to keep your lights on and your virtual girlfriend warm. Your buddies comfort you by attributing this inevitability to inflation or price hiking, but you think you've narrowed it down to that hundred acre plot of land Amazon dug in your backyard to host their new rectangular data centers or something. Sadly, I'm going to have to break the bad news, because the true reason that your bill is so high is because you haven't understood FlashAttention yet.

Once you've mastered the unfortunate source code of this diabolical nightmare, you can sleep comfortable in your room. Your RTX-5090 will replace your outdated heat system and the blackness of your VSCode window will permanently fuse into your memory. And after many gruelling nights you'll notice your energy consumption go down as the only time your OLED monitor has to do any work is when you switch to Chrome to look at that one Cutlass documentation page you pretend you've read already. On the 30th of the month you might finally see that bill drop a few bucks. Unfortunately, you might accidentally get hired by the company that your virtual girlfriend "works" for and you might notice that your bill will slowly begin to creep up again despite your effort. Fortunately, the blood money that enters your bank account can now comfortably keep the lights on at home.

# What is FlashAttention
FlashAttention is a GPU algorithm that significantly speeds up the attention mechanism step in transformers--the pivotal architecture that all the AI models today rely on. Attention, or self-attention, is *the* critical step that allows tokens to "attend" to all previous tokens to "pick up" and understand the context of which it is a part of.

> For example, in the sentence: "The cat is fat", the adjective "fat" likely *attends* to the noun "cat" and the connective verb "is" to understand that this feline is chonky.

If you've read through the intro of this blog, you surely know all of this already, so let's cut to the chase. The vanilla attention algorithm is surprisingly simple, and you can implement it in barely a few lines in Pytorch. It's just two matrix-multiplies with a softmax in-between, each of which already have agressively optimized GPU kernels. When Google initially released *Attention is All You Need* back in 2017, they probably strung together these operations without a second thought, because it was wholly sufficient at the time.

## Attention Definition
I won't go too much into the "why" of attention. I recommend you look at: . We'll briefly define it here.

Given an input sequence (prompt) x, we construct the weight matrices (query, key, value):

$$Q, K, V$$

We'll define this attention step with sequence length $seq$ and hidden dimension $d_h$, such that $Q, K, V$ are all of shape $(seq, d_h)$ for simplicity. Attention is simply:

$$\begin{aligned}
P &= \frac{QK^T}{\sqrt{d_h}} \\\\
S &= \text{softmax}(P) \\\\
O &= S \cdot V
\end{aligned}$$

In Pytorch, we can simply write the following code:
```python
P = Q@K.T/torch.sqrt(d_h)
S = torch.nn.functional.softmax(P, dim=-1)
O = S@V
```

Simple enough, and for most early transformers and language models, this plus maybe a `torch.compile` and some micro-optimizations were probably enough. When GPT-3 took the market by storm, we were still in the early days where 2048-token context lengths were the standard, vanilla attention this way was enough. However, as hungry devs and attention-starved adults began running into these context walls in a few minutes, 2048 was an uncomfortably tiny ceiling.

Three or four years later, we now have models with 128K or 256K or even 1M context lengths with AIs that will keep providing you tokens as long as you have wallet tokens to give back. Although the attention mechanisms are far more complicated with new context-extension techniques, the size upgrade wasn't as simple as changing the sequence length variable in a config file. As usual, it required the invention of flash attention for training GPT-4 to even be remotely possible.

## GPU hell
When AlexNet revolutionized the idea of using graphic cards to train machine learning models, GPU kernels turned from relatively niche things to an industry standard within a few short years. These were highly-optimized matrix operations written in CUDA--NVIDIA's proprietary language--to take advantage of GPUs in order to accelerate training and inference. Standard operations like a basic matrix-multiply or vanilla softmax kernels were almost certainly the first to be made. NVIDIA themselves have CUBLAS, a in-house properly tuned library for matmuls that has become the industry standard for anyone who doesn't wish to multiply matrices by hand. You can dig through any ML library today and find finely-tuned softmax kernels[LINK]() that were made a decade back.

So if they're so finely-tuned, what's the problem here exactly? As you might of guessed, it's **context length**. If we take a quick examination of the attention mechanism, we can see we are doing a $(seq, d_h) \cdot (d_h, seq)$ matmul followed by a $(seq,)$ softmax reduction and followed by a $(seq, seq) \cdot (seq_len, d_h)$ matmul. Let's approximate how many FLOPs each step should take for GPT-3, which used $seq = 2048, d_h=128$.

> Note that FLOPs is the total floating point operations and FLOPS is the floating point operations / second. For clarity, I will use FLOP/S in this article.

| Operation | Formula (~Ops) | FLOPs (seq=2048, d_h=128) |
| :--- | :--- | :--- |
| **QKᵀ Matmul** | 2 * seq_len² * d_head | 1B |
| **Softmax** | ~6 * seq_len² | 25M |
| **Score @ V Matmul** | 2 * seq_len² * d_head | 1B |
| **TOTAL (per Head)** | — | **2.2B** |

Running a quick benchmark in PyTorch using an A100 GPU (say at bottom all tests were done on this), here were some performance metrics for each operation (1000 iterations):
| Operation | Time/op (ms) | TFLOP/S |
| :--- | :--- | :--- |
| **QKᵀ Matmul** | 0.028 | 40.32 |
| **Softmax** | 0.011 | 0.12 |
| **Naive Attention** | 0.102 | 21.03 |

> Only 30 microseconds per matmul, which in a vaccuum is quite insane for a billion FLOPs.

By 2023, OpenAI's frontier GPT-4 model upgraded its context length to 128K--64 times larger than GPT-3. They released smaller GPT-4 models earlier in the year with 8K or 32K context lengths, but let's re-examine the numbers once again with 128K for maximum drama.

| Operation | Formula (~Ops) | FLOPs (seq=128K, d_h=128) |
| :--- | :--- | :--- |
| **QKᵀ Matmul** | 2 * seq_len² * d_head | 4.4T  |
| **Softmax** | ~6 * seq_len² | 100M |
| **Score @ V Matmul** | 2 * seq_len² * d_head | 4.4T |
| **TOTAL (per Head)** | — | **9T** |

We've increased the total FLOPs by $4.4T/2.2B\approx 4000$. Let's run the benchmark again...oh oops, my GPU threw an out-of-memory error...

The primary limitation with attention isn't just
