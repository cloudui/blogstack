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

!TODO llama  3 vs llama3.1

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

## Important Llama Concepts

### General Query Attention (GQA)
The original multi-head attention mechanism (MHA) covered in my previous blog splits the Q, K, and V into the same number of attention heads. Earlier models like GPT-2 used MHA -- for example, the original GPT-2 small base model had:

- vocab size (`hidden_size`): 768
- number of heads (`num_heads`): 12
- head dim (`head_dim`): 64

After tokenization, this means the input `x` is of size `(batch_size, seq_len, hidden_size)` and the weight matrices $W_q, W_k, W_v$ are of size `(hidden_size, num_heads, head_dim)` We can interpret each head of Q, K, V to be a separate set of questions, keys, and values; for example, one set of questions relating to "is this token a noun?" and another relating to "is this token a verb?". 

One thing we notice is that each Q, K, V tensor is quite beefy in VRAM. This is mainly a bottleneck for the KV cache, where larger and larger context lengths were blowing up the available memory space. In 2023, Google introduced general query attention (GQA), a modified version of MHA that reduces the number heads on the K and V tensors with relatively negligible performance impact. In the GQA representation, multiple question heads share the same KV heads. We can imagine that the "information" that the questions "is this token a noun?" and "is this token a verb?" rely on are very similar. GQA essentially compresses the KV representation space to encode information across different question spaces. 

Meta used GQA as its attention backbone for Llama-2 through Llama-4, using `qhead:kv_head` ratios from 4:1 for smaller models to 16:1 for bigger models. In GQA, we still want each attention block to ask a lot of questions, taking advantage of the fact that the keys and values can be reliably shared between similar questions. As a result, we can reduce the KV cache size by up to 16x on the mega 405B model, which is massive.

> As shown in the [architecture section](#llama31-8b-architecture), Llama3.1-8B uses a GQA head ratio of `4:1`. 

![TODO]: link paper, maybe a good blog explaining more about it

### Rotary Position Embedding (RoPE)
The earlier LLM attention mechanisms added a position vector $p$ to encode the the positional information of each token ($x+p$). This $p$ vector was simply a learned embedding in the token space -- each token 0-2047 (or whatever the context length was) had its own absolute positional representation. Researchers quickly realized that these absolute embeddings were noisy, where semantic content often got tangled with positional meaning. Furthermore, this additive $p$ never strictly isolates the relative distance $m-n$ between tokens $m, n$ -- if the model learns that a adjective precedes a noun by one token, it's forced to relearn that relationship between repeatedly at $(1,2)$, $(800, 801)$, and $(1022, 1023)$. It simply doesn't have the means to understand the relative difference between tokens.

RoPE was introduced to force the model to understand position as a relative mechanism. It injects the absolute positional info by rotating the Q and K vectors in the 2D vector space so that their inner product naturally computes the *relative* position. While the original attention mechanism modifies the input embedding via $x + p$ while RoPE is a function $\mathbf{R}(\mathbf{x},m)$ that modifies Q, K such that:

$$\langle \mathbf{R}(\mathbf{q}, m), \mathbf{R}(\mathbf{k}, n)\rangle = g(\mathbf{q}, \mathbf{k}, m-n)$$

> If you want to skip the mathematical details of RoPE, go ahead. Just knowing the equations is sufficient for optimizing an inference kernel, but it's nice to understand the mechanisms behind what we're doing. It might require you to refresh your linear algebra or complex analysis knowledge.

RoPE chooses the humble rotation as it's geometric formulation for this task. If we somehow represent the embedding space as complex 2D pairs $z= x + iy$, we can easily rotate $z$ by multiplying by $e^{im\theta}$. 

$$\mathbf{R}(\mathbf{x}, m) = \mathbf{x}e^{im\theta}$$

In the first attention step $QK^T$, we find the dot product between pairs of $\mathbf{q}_m$ and $\mathbf{k}_n$. Taking the inner product of rotated q, k yields: 

$$\langle \mathbf{q} e^{im\theta}, \mathbf{k} e^{in\theta} \rangle = \operatorname{Re}\left[(\mathbf{q} e^{im\theta}) (\mathbf{k} e^{in\theta})^{\*}\right] = \operatorname{Re}\left[\mathbf{q}\mathbf{k}^{\*} e^{i(m-n)\theta}\right]$$

We can see from this formulation that the product here is strictly a function of the relative distance $(m-n)\rightarrow e^{i(m-n)\theta}$. 

You might notice that Q and K are usually have dim size 64 or 128 and not 2. This is true. Instead, the authors encode this rotational relationship by *pairing up* weights in the Q, K token embedding space. For an attention head of even dimension $d$, RoPE splits the vector into $d/2$ independent 2D chunks:

$$[x_1, x_2], [x_3, x_4], \dots, [x_{d-1}, x_d]$$

Each 2D sub-vector is rotated by an angle determined by its position m and a base frequency $\theta_i$:

$$\theta_i = 10000^{-2(i-1)/d}, \quad i \in \left\\{1, 2, \dots, \frac{d}{2} \right\\}$$

The complex notation earlier was convenient to represent this relationship, but in our new 2D pair embedding space, it's more convenient to work with the equivalent 2D rotation matrix:

$$\mathbf{R}_{\Theta, m}^{(2D)} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\\\ \sin m\theta & \cos m\theta \end{pmatrix}$$

After the rotation, instead of $[x_1, x_2]$, we now have

$$[x_1\cos(m\theta_1)-x_2\sin(m\theta_1), \\, x_1\sin(m\theta_1)+x_2\cos(m\theta_1)]$$

Subsequent pairs are transformed the same way with decreasing angle $\theta_i$. The exponentially decreasing angle based on the chunk gives us the following properties:

- Early chunks have angles close to 1: $\theta_1 = 10000^{0}=1, \theta_2 = 10000^{-2/d} \approx 1$. The angle rotates through the entire circle extremely quickly. Let's call them "fast knobs".
- Later chunks have angles close to 0: $\theta_{end} \approx 10000^{-1} = 0.0001$. The angle rotates through the entire circle extremely slowly. Let's call these "slow knobs".

The frequency of these knobs is what gives us the positional information. For tokens that are close together, the "fast knobs" have high signal (similar cosine) while the slow knobs are more stable. For tokens that are far apart, the "fast knobs" essentially become pseudo-random since the periodicity is so small compared to context sizes that they essentially carry *no information*. However, the "slow knobs" still carry a stable signal. The collection of knobs is what allows us to model short-term and long-term relationships. 

What you might notice is that RoPE attenuates far-apart tokens naturally. Language is overwhelmingly local, and it's usually unlikely that a word on page 100 is still relevant to one on the first. In our $theta_i$ example here we use a coefficient of 10,000, which is not simply an arbitrary choice. In the original paper, this coefficient modeled context lengths from 512 to 4096 tokens. With the minimum angle of 0.0001, the max context length only traces an angle of $0.0001 \cdot 4097 \approx 23.5\deg$. The period of the slowest knob would be $2\pi/0.0001 \approx 62800 \text{ tok}$, so we see that the slowest gears have no long-range aliasing. At close token distances, the cos similarity for 64 tokens all have signal close to $\cos(0)=1$, where the cosine in long-distance tokens only remain close to 1 for the long range knobs while the faster moving knobs destructively interfere toward 0. Therefore, tokens far apart "attend" to each other less, which logically makes sense. You can see the actual cosine similarity curve in either of the links at the bottom of this section. 

We can analogize much of the behavior of RoPE, but it's important to realize that if it didn't work empirically, then any explanation would be moot. The original team likely had motivation to try this technique and realized it actually worked in practice, which is why we can convince ourselves that these explanations make sense. It's good to understand the intuition behind it, but it's also fine if parts of it sort of feel like black magic. For example, this arbitrary forcing of weight pairs in the QK embedding space forces the gradient to "learn" the position. It explicitly forces certain pairs to carry certain information, which could of not worked for some reason. But, given that we could just add a fully learned position $p$ to the token space, maybe this doesn't seem too far of a stretch. Either way, this section is simply to help us figure out how to optimize inference. So dwell on it a little bit, have some faith, and let's get a move on. 

### RoPE Llama-3 vs Llama-3.1
The original Llama-3 use the base vanilla RoPE as we described above with a max context window of 8K tokens. Although Llama-3 and 3.1 share the exact same decoder architecture, it required one architectural change in the RoPE mechanism in order to achieve the massive jump to the 128K context window. Let's talk about the change first and then discuss the motivation:

The original Llama3-8B RoPE used the formula:

$$\theta_i = 500000^{2i/d=128}$$

The adjusted RoPE depends on the configuration depends on two cutoff frequency thresholds defined in a config. For 3.1-8B:

```python
{
"rope_scaling": {
  "factor": 8.0,
  "low_freq_factor": 1.0,
  "high_freq_factor": 4.0,
  "original_max_position_embeddings": 8192,
  "rope_type": "llama3"
},
"rope_theta": 500000.0,
"head_dim": 128
}
```
The frequency thresholds are inversely proportional to the wavelength, so we define our wavelength thresholds based on the original max position embeddings and the high frequency factors:

```python
low_wavelength_cutoff = original_max_position_embeddings / high_frequency_factor 
= 8192/4 = 2048
high_wavelength_cutoff = original_max_position_embeddings / low_frequency_factor 
= 8192/1 = 8192
```

We scale the the range of the wavelength:

- High frequency (low index): if $\lambda_i < 2048$, $\theta_i^\prime= \theta_i / 1.0$
- Low frequency: if $\lambda_i > 8192$, $\theta_i^\prime = \theta_i / 8.0 $
- Mid frequency: if $2048 < \lambda_i < 8192$:

$$\theta_i^\prime = (1 - \alpha_i) (\theta_i / 8.0) + \alpha_i \theta_i,$$
$$\alpha = \frac{\frac{L_{\text{old}}}{\lambda} - \text{low\\_freq\\_factor}}{\text{high\\_freq\\_factor} - \text{low\\_freq\\_factor}}$$

So this all seems kind of arbitrary and weird, and the reason this is necessary to begin with is that Llama3.1 is simply an extension of Llama3 and not a completely retrained model. The original RoPE theta mechanism was designed around the 8K context window, and Meta wanted to use their already trained base models to *extend* training to these new 128K context models. 

- At the 8K RoPE scale, the knobs start to fall apart when you use the same RoPE scaling factor for token number 100,250. The model wasn't trained to understand $\cos(100250\theta_i)$ scaling. As a result, we need to scale it down to be in the range that the 8K base model understands
- Extending training means you don't waste the money and time they spent training on the 15T training tokens
- If they just rescaled the RoPE coefficient, it would have scrambled the high-frequency dimensions that the model relied on for basic grammar and short-range syntax

So if Llama3.1 was trained from scratch, they could've just tweaked the coefficient and called it a day. But since they wanted to simply extend to a larger context model, they just tweaked the number range to mimic the original one via this scheme. If we look at this piecewise scaling, we just see that we keep high frequencies (low index RoPE pairs, e.g. 0-10) the same, tune down the low frequencies (high index RoPE pairs, e.g. 50-63) and then create a linear step function in-between. 

As we'll see soon, this frequency table is pre-computed once and just sits in memory. There's no performance impact here -- we're just taking our old table, checking some ranges and tweaking some numbers, and putting it back in its place. 


#### Benefits (KV cache, absolute, etc.)


> Original Paper: https://arxiv.org/pdf/2104.09864
> 
> Some blogs with visualizations and motivation:
> - Eleuther AI: https://blog.eleuther.ai/rotary-embeddings/
> - Towards Data Science: https://towardsdatascience.com/rope-clearly-explained/

# Let's Get it Working First

## We're using Triton
One thing we'll notice quite soon is that it's not worth the effort to write any of these kernels in raw CUDA/CuTe like last time. The computation is extremely minimal at `seq_len=1` that a slight nanosecond improvement will not move the needle at all. We can show the principles that actually matter for decode inference equally well using Triton, and we need not spend 5x the time for no appreciable gain. 

Furthermore, we will be orchestrating many kernels wrapped together in Python, where Triton will be much faster and easier to work with. Most of the insight here comes from how we tackle inference *within* the wrapper itself, so we'll be focusing a lot of the effort here.

## Equipment

I worked with the 32GB RTX 4500 Pro server edition on Runpod as my primary GPU. It's easily large enough for 8B models and has good memory and compute performance for the price: 800 GB/s mem bandwidth at 72c/hour. It gave me plenty of runway as a cheap-ass and was a good choice for this inference project.

In practice, different memory to performance profiles changes the things you optimize for. It might change whether your bottle neck is your orchestration, compute efficiency, or data movement, so sometimes you have to be careful which card you're working with. Compared to the RTX 4500 I worked with, the newest H100s and BX00s have 2-10x the memory speed with more than 2x the performance on FP16 alone. With different quantizations, the differences skew even more. For our purpose with batch size 1, FP16, and basic decoding, these differences don't matter too much. But in production-grade systems, evaluating the profiles and tradeoffs become crucial. 

## A Quick Analysis of the Source

Let's quickly take a look at the structure of the code:

```python
class LlamaDecoderLayer(...):
  ...

  def forward(...
  ):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states
```
This `DecoderLayer` is one LLama attention block and is about as nice as you can write it in PyTorch. Each operation is clearly invoked and defined in the constructor with the class implementation defined in the same file (e.g. `RMSNorm`, `MLP`). Each class module is implemented simply; for example, the MLP module is defined a standard PyTorch linear layer:

```python
class LlamaMLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
    self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
    self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
    self.act_fn = ACT2FN[config.hidden_act]

  def forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj
```

The final `LlamaModel` is just combining the tokenization, embedding, and decoder stack together to actually generate the output. 

### Benchmarking Hugging Face
https://github.com/cloudui/inference/blob/main/benchmarks/bench_throughput_hf.py
Benching 128 decode steps on top of a 512-token prefill yields us a throughput of ~42-43 tok/s. This uses the default eager mode with Dynamic Cache with standard dot-product attention (SDPA, same as FlashAttention-2). Typically, we compile with static cache (full KV cache allocation) to see the true max performance, which yields a throughput of ~45 tok/s. This is the baseline performance we're benchmarking against. We know that the source doesn't optimize anything, so there should be room for improvement. 

The RTX Pro 4500 server edition has a max HBM throughput of ~800 GB/s. One decode step feeds all 8B parameters every cycle, with negligible space for the one-token input space. Since we're working with FP16 weights, the theoretical max throughput is:

$$800\text { GB/s} / (8\text{B param} \times 2\text { bytes / param}) \approx 50\text{ tok/s}$$

An unoptimized HF easily achieves 90% of the max performance out of the box, which tells us basic PyTorch CUDA dispatch is already pretty computationally efficient. Just this number alone will soon tell us where our bottlenecks are.

## Step 1: Recreate the Call Stack
The first step for a decode engine is pretty much to get a barebones one working. I recreated the HF call tree from scratch using simple PyTorch calls to perform the operations. At this stage, the implementation details aren't too important; we just want to make sure the output we get matches whatever HF has. 

For example, something like `RMSNorm` can be as simple as:

```python
class RMSNorm:
  """Holds the learned scale weight for RMSNorm. Forward dispatches to kernel."""

  def __init__(self, dim: int, eps: float = 1e-6):
    self.eps = eps
    self.weight = torch.ones(dim)

  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    return rmsnorm(x, self.weight, self.eps)
```



### Memory Layouts

