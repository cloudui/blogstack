---
title: FlashAttention, but the Actual Details

categories: [Discussion, Politics]
tags: [politics, inequality, socialism, satire]
# mathjax: true
math: true

author: Eric Chen
image: thumbnail.webp
date: 2026-05-04

description: Taking an unfortunate look at Cutlass and CUDA
---
I made the unfortunate decision to read through the FlashAttention-v2 paper maybe a month or two back. Its name was tossed around work enough to pique my curiosity, and I found myself forking through the details one lazy afternoon. It read similar to one of those simple yet foundational ML papers back in the day--something like ResNet or BatchNorm. The math was almost trivially understandable to anyone who has even touched a few models in the last decade, and the algorithm seemed relatively understandable to my untrained eye. I had not even touched a GPU kernel before, and I felt some semblance of hubris given that my naive computer engineering degree allowed me to understand an algorithm of such importance.

I made the unfortunate mistake of thinking I actually understood the algorithm in its entirety. At a surface level, sure, but such triviality only exists in the abstract. It was akin to me thinking that I could build a microwave simply because I know how it works. However, I was not amateur enough as to simply read the paper, I wanted to implement it myself like the engineer that I am. I had heard Triton was making waves in the GPU sea, and I decided to take a venture into its wake. I started by making some simple kernels: a softmax here, a SwiGLU there. One or two hours of barely any code, and I was done. Easy peasy. In fact, I wrote the full flash attention kernel within two or three days during the work week and achieved similar-ish performance to pytorch's native implementation on smaller sequence lengths within this time. I replicated Tri Dao's work in a fraction of the time, and I didn't even optimize my kernel that much along the way. I was feeling unstoppable.

I made the unfortunate oversight of trying to rewrite the kernel using CUDA. Obviously, two days of work with only a few sweats of confusion wasn't enough to satiate my hunger. I felt as though Triton had taught me nothing of GPU programming, and I wanted more. In the throne room of luxury and convenience, I sought the caves and the jungles. With a few more drops of sweat and maybe a few tears, I finished a few elementary kernels in CUDA. Piece of two-layer cake. But, I wanted that 12-layer wedding cake. I sought to recreate flash attention in its fully glory, and it seemed like I was equipped to do so. Over the course of the next week or two, I would complete a first draft iteration of FlashAttention-v2 using a somewhat outdated WMMA API, achieving...only 20% of native FA2 performance. Clearly, this old WMMA abstraction wouldn't cut it. I looked at Dao lab's source code, and it looked nothing like the verbose mess of for loops and somewhat-understandable code in my repository.

I made the unfortunate blunder of rewriting my kernel with CuTe in C++. As with any performance library, I had to rewrite the entire using a poorly documented library with complex syntax and abstracted BS that somehow is way faster. What I thought would be a somewhat simple task turned out to be a nightmare in understanding. It's as path I'm sure any low-level developer had to cross at some point in their journey, and I am certain they have seen the fear in the eyes of the avoiders and the stockpile of bodies along the way. After many pools of sweat, many hairs pulled, and many neurons disintegrated, I finally found success. But, I was not filled with glory, confidence, or peace. In the end, I could only feel relief and a massive sense of responsibility to aid any traveler who dare wish to traverse this same path.

# FlashAttention-v2
If you're reading this, I will assume you already have a solid understanding of the attention mechanism and at least the basics of the FlashAttention algorithm itself. If not, I recommend reading Tri Dao's original paper to build your understanding of the algorithm before coming back. Or, you could just read the article as is, because you will probably piece it together through the struggle of trying to understand. It would be helpful for you to at least know the pseudocode/baseline algorithm for FA2, and it would be even better if you have maybe tried simulating it in pytorch (or your framework of choice) or maybe even implemented it in Triton.

If you've never touched CUDA, you should at least try to understand it's SIMT programming nature and maybe implement some basic level kernels using this thread-level view. Try to build your understanding of how CUDA works and a solid understanding of NVIDIA's GPU architecture, from threads to warps to thread blocks to SMs and beyond. I will talk about a lot of these concepts as if you at least have a basic understanding of them. I will be as comprehensive as I can, but it will be an uphill battle should you try to read this blog in its entirety without some background knowledge.

Most of this blog will concern how high-level concepts like "online softmax" or "gemm" actually translate to production-grade code. The algorithm itself is not particularly difficult, but the implementation details at the CUDA level can become a nightmare, particularly to beginners. Tri Dao originally wrote FA2 using CuTe (CUDA Templates), a core component within NVIDIA's CUTLASS library that abstracts away tensor layouts and thread-data mapping for high-performance GPU computing. Although this may seem nicer than doing it from scratch, there are a lot of intracacies and difficult to understand design choices that make reading it somewhat of a nightmare. It's higher-level than your typical C program with normal for loops and variables, but it's still practically as close to bare metal as most people will ever get. So, even though you will understand CUDA and FlashAttention much better, honestly it will mostly allow you to understand CuTe, why it exists, and how people actually use it.

Fortunately, since the release of Blackwell (B200), NVIDIA released CuTe's Python DSL--a python library you can use to write the same code without all the annoying crap that's baggaged with C++. The use case is pretty much unchanged, but it makes debugging and templating much more palatable, and the compile times are enormously faster. Moving forward, CuTe 3.X in C++ will probably be somewhat of a relic of the past, but as a learning exercise, nothing beats the absolute struggle of working with the most annoying and explicit version of whatever you're trying to learn. Let's get started.

# Overview
## Design Choices
We're going to make some basic design choices to make this learning exercise simpler on the implementation side. A lot of the source code involves edge cases and optional configuration settings (RoPe, QK smem sharing, etc.) that aren't practical for learning the fundamentals of FA2 and CuTe. Our choices are as follows:
- A100: the GPU I had access to and the industry standard when FA2 was released. Hopper and Blackwell have even more complicated algorithms due to hardware improvements and optimizations
- fp16: supported on A100 tensor cores, pretty basic default for most kernel ops for training
    - fp32 accumulation, reduces precision drift, more accurate FLOPs for softmax and scale
- Clean basic out-of-the-box attention mechanism: no causal masking, RoPe, dropout, etc.
- head_dim: focus on 64 and 128 block size, although 32 may be covered, though I didn't specifically check
- Assume sequence length is a power of two, more specifically a multiple of the Q block size.
- We expect $Q, K, V$ to be in row-major order.

## Some Naming Conventions
If you look at DaoLab's source code, you might notice they have some weird naming conventions. Some of them are standard CuTe/CUTLASS some carry over from other things. Here are some patterns:
- Starts with k: compile-time constant, e.g. kBlockM, kHeadDim
- $M, N, K$: All of general matrix-multiply (GEMM) parameters are in this order for a $(M, K) \times (K, N)$ matrix-multiply. Hence, the shape of Q is (kBlockM, kHeadDim) and the shape of K, V is (kBlockN, kHeadDim).

## Basic Structure
First, attention itself:

$$\begin{aligned}
P &= \frac{QK^T}{\sqrt{d_h}} \\\\
S &= \text{softmax}(P) \\\\
O &= S \cdot V
\end{aligned}$$

Or in pytorch for those who haven't read a math equation in a while:
```python
P = Q@K.T/torch.sqrt(d_h)
S = torch.nn.functional.softmax(P, dim=-1)
O = S@V
```

Let's establish the specifics of the FA2 algorithm at a high level.
- Our grid is `batch/head` x `q_tile`. The batch/head dimensions are independent and can be grouped. The `q_tile` determines which tile of Q we get, and we make it the last dimension for better cache locality between thread blocks.
- Q is of shape (kBlockM, kHeadDim). The main computation on any thread block revolves around the Q tile. This Q tile does not change for the duration of the thread block. We iterate over the relevant K, V tile per thread block to get an output tile. Each q tile maps to exactly the same size output tile, which is necessary as we need to manifest a whole row of P to do the softmax.
- We load each tile from global memory (GMEM) to shared memory (SMEM) for staging. When we need to do our GEMM, we load from SMEM to the register file as we loop over K and V.
- Our GMEM->SMEM copying are all async (`cp.async` on Ampere). Q technically doesn't really have to since it doesn't overlap that much compute but is a micro-optimization.

The overall pseudocode for a tile Q is:
1. Define GMEM, SMEM, register files, hardware copy/GEMM instructions, and mappings
2. Load Q tile from global memory to SMEM. This is only done once, as Q tile doesn't change.
3. Prefetch 0th K-tile.
4. Loop start: Wait for K-tile to arrive. Then, prefetch the next V-tile.
5. Issue GEMM for $P = QK^T$ tile.
6. Wait for V-tile to arrive. Then, issue next K-tile prefetch if we're not on the last iteration.
7. Compute $S=\text{softmax}(P)$ and softmax statistics and update accumulator/output tile.
8. Issue GEMM for $O = SV$ tile.
9. Loop back to 4 until row is complete.
10. Scale final output by $l=\text{expsum(P)}$
11. Copy output from SMEM back to GMEM. Ampere doesn't have any direct SMEM->GMEM instructions so we stage this copy through registers.

Only 11 steps and they're all pretty simple in concept...Let's take a deeper look into the implementation details.

# CuTe, the Basics
Bombarding you up front with all the design choices in CuTe/CUTLASS will only confuse you, and the best way to learn is honestly by necessity. However, having some basic info is still probably required, so I will bombard you with some sadness before we move onto the "cool" stuff.

## Background
CuTe is essentially a templating engine that allows you to manipulate memory using tensors, shapes, layouts, data types, and strides, sort of similar to pytorch's `torch.Tensor` object. Unfortunately, it's not nearly as friendly. But, if you're familiar with any deep learning library, these concepts should click pretty quickly. It allows you to declare a general "shape" once and if you template it with a fp32 vs fp16, you can just pass the relevant parameters to your kernel template.

However, you are still responsible for all of the sizes. It may be able to extract fp16 from a 128-bit load, but you'll have to figure out that 128-bits is 8 fp16 numbers. It just handles the typing on your behalf and lets you index stuff more easily. This will click later.

## Layouts, Shapes, and Strides
Ah yes, back to tensor school. A shape and stride is precisely the same concept as in PyTorch. A layout is just a composition of a shape and a stride.

```cpp
#include <cute/tensor.hpp>
// runnable just like this without GPU
auto layout = Layout<Shape<_8, _16>, <Stride<_1, _8>>>{};
auto layout_1 = make_layout(make_shape(Int<8>{}, Int<16>{}),
        make_stride(_1{}, _8{}));
print_layout(layout);
// this is the shape of a torch.tensor([[0]*8 for _ in range(16)]).T
```

A shape of (8, 16) with stride (1, 8). Pretty simple. Both declarations are identical. So what's with the freaky underscore numbers?

## Statically vs. Dynamically Typed
Any standard C++ integer passed into a layout, shape, or stride is dynamically typed, i.e. its value is only known at runtime (e.g. int, const int, static int). Even CUDA's `constexpr int` is treated as such by CuTe. Any time you index into a tensor, the library will compute

```cpp
A[i][j] = i*stride_row + j*stride_column
```

Each index operation is a multiply and add, which can be quite costly. Instead, when we can, we opt to use for statics: type wrappers used by CUTLASS to allow the value to be known at compile time. It's just a C++ compiler trick that allows CuTe to compute all indexing during compilation rather at runtime, saving the GPU from having to do so while its running. Obviously, you can only do this if sizes are predetermined, either because they are definite, templated, or constant. So instead of passing in `make_stride(2, 4)`, we can pass in `make_stride(make_Int<2>{}, _4{})`. Functionally, these are the same, but any subsequent indexing done will be done at compile time for the latter.

Layouts *do* take in dynamic integers as well. They should be used *if they are only known at runtime*.

Some syntax quirks:
```cpp
// identical, CuTe provides most power of twos by default as shorthand
Int<8>{};
_8{};

// Functions take in objects, types only use types
auto l1 = Layout<Shape<_8, _4>, Stride<_1, _4>>;
auto l2 = make_layout(make_shape(_8{}, _4{}), make_stride(_1{}, _4{}));
// type
Int<8>;
// object/struct
Int<8>{};

// can have both dynamics and statics in same layout
auto shape = make_shape(2, _4{});
auto stride = make_stride(Int<256>, 64);
```

Read more here:

## Tensors
Tensors are just a pointer wrapped in a layout. The underlying data is just a pointer, usually of contiguous data, and the layout determines how we interact with it. Pretty much exactly the same as a `torch.Tensor`. However, we manage the layout: we can change it to whatever we want, however we want, but we are ultimately responsible for the tensor's integrity.

```cpp
static int x[] = {1, 2, ..., 32};
auto l_row_major = make_layout(make_shape(_8{}, _4{}),
    make_stride(_4{}, _1{}));
// row major view
auto t_row_major = make_tensor(data, l_row_major);

// column major view
auto l_col_major = make_layout(make_shape(_8{}, _4{}),
    make_stride(_1{}, _4{}));
auto t_col_major = make_tensor(data, l_col_major);

// tensor indexing
// i, j = 2, 3
int n_row = t_row_major(2, 3); // 12
int n_col = t_col_major(2, 3); // 15
```

## Row and Column Major
In row-major formats, data is stored row-contiguous (C, C++ style), i.e. `A[0][1]` and `A[0][2]` are contiguous in memory.

In row-major formats, data is stored column-contiguous (Fortran, CUBLAS style), i.e. `A[0][1]` and `A[1][1]` are contiguous in memory.

The way to tell is by the stride; for any 2D matrix with stride $(a, b)$, the matrix is row-major if $b=1$ and column-major if $a=1$. In CuTe, any layout without a provided stride is **column-major** by default.

For FA2, we assume Q, K, and V are all **row-major.** Although somewhat arbitrary, most consumer applications or libraries like Pytorch or JAX are row-major by default, so this is the most obvious configuration for consistency. Furthermore, Ampere tensor ops seem to be oriented around row-major instructions, so it's also a choice in simplicity.

# CuTe, Copy, then Cry
## A100 (Ampere) Specs
The entire point of FA2 or even GPU optimization in general is to maximize compute by overlapping it with memory loads. Here are the memory and card specs of A100 GPU (Ampere):

| Storage Level | Latency (Clock Cycles) | Magnitude Slower than Registers |
| :--- | :--- | :--- |
| **Registers** | ~1 cycle | — |
| **Shared Memory (SMEM) / L1** | ~20–30 cycles | ~25x |
| **L2 Cache** | ~200 cycles | ~200x |
| **HBM2e (Main Memory)** | ~400–600+ cycles | **~500x+** |

| Feature | Specification |
| :--- | :--- |
| **Total SMs** | 108 (SXM4) / 128 (Full Die) |
| **CUDA Cores per SM** | 64 (FP32) |
| **Max Threads per SM** | 2048 |
| **Max Warps per SM** | 64 |
| **Max Blocks per SM** | 32 |
| **Registers per SM** | 65,536 (32-bit) |
| **Max Registers per Thread** | 255 |
| **Max Shared Memory per SM** | 164 KB |
| **Max Shared Memory per Block** | 163 KB |
| **L1 Cache (Combined with SMEM)** | 192 KB total pool per SM |
| **L2 Cache** | 40 MB or 80 MB |

## GMEM->SMEM (Async Copying)
A rule of thumb is to have approximately **150-200 FLOPs per byte loaded from HBM**. On Ampere, we can take advantage of asynchronous copies from GMEM->SMEM that can help us overlap a meaningful amount of the tile fetches with compute. This means our threads can do other stuff without having to immediately stall for memory fetches.

Memory coalescing is also extremely important here. GPUs never fetch just one byte at a time; they can fetch a whole 32, 64, or 128-byte chunk at a time. Ideally, a warp fetching a full 128-byte contiguous block allows the GPU to issue one instruction to clear this entire block of data. Furthermore, this block fully saturates a L2 cache line, making any subsequent cache accesses more efficient. If all 32 threads in the warp are each fetching some random chunk scattered across memory, then the memory controller would issue 32 separate transactions, immediately crushing your performance, hopes, and dreams. So in principle, we would love for all 32 threads to "coalesce" by fetching 128-byte contiguous blocks of data. Let's introduce how we do that in CuTe.

### Copy Atoms
There is a boatload of copy PTX instructions in CUDA. You can fetch 32 bytes, 64 bytes, one byte, synchronous or asynchronous alike. CuTe neatly packages these copy instructions into a core piece called an `Atom`. These "atomic" pieces are the core hardware instructions that you eventually pass to the `copy` function so it knows what instruction to use to copy your data.

Ampere has a specific asynchronous `Copy_Atom` with the architecture name `SM_80`: `SM80_CP_ASYNC_CACHEGLOBAL<bit_size>` or `SM80_CP_ASYNC_CACHEALWAYS<bit_size>`. The `cache_global` and `cache_always` map to the PTX instructions `ld.global.cg.u32` and `ld.global.ca.u32`; `cache_global` loads straight from L2 to the destination, skipping over L1 cache, while `cache_always` also loads the data into L1. Most kernels will use `cache_always` by default because of improved spatial and temporal locality across threads. But, in FA2, we never reference Q, K, or V again once they are loaded into SMEM--therefore, we can bypass the L2 cache, which is slightly faster. It also reduces thrashing at the L1 level and allows more important data to stay in-cache. In practice, this is a micro-optimization and relatively not that important.

The `bit_size` supports up to 128-bit loads. **Bits**, not bytes, as these atoms are **per-thread**. The memory controller indeed loves sending the full fat 128-bytes per warp, but CUDA views all transactions at the thread-level. Hence, our atom loads a total of $128\dot 32 / 8 = 512$ bytes. This means each 128-bit fetch across the 32 threads in a warp takes $512/128 =4$ memory transactions in 4 "phases" (more on this later). For our purposes, we want that full coaslesced 128-bit power using `cache_global`. We can define the `Copy_Atom` with the following syntax:

```cpp
#include <cute/atom/copy_atom.hpp>
using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
    cute::half_t>;
```
We use the cute namespace types for robustness, and our source data type is fp16 (`cute::half_t`). Each thread therefore loads $128/16=8$ halfs.

### Tiled_Copy
Even though each thread copies 128 bits, each thread block is usually working with a variable amount of threads/warps. Given the 4 tensor cores per SM, 4 warps per block is typically a good choice for FA2. This means we have to determine how to copy each Q,K,V tile using these 128-bit async copies.

CuTe uses `Tiled_Copy`, which "tiles" the memory you are trying to copy (in this case, GMEM) in a structured way over your entire memory region. It outlines the "tiling strategy" that your threads will follow.

> Note that the "tiling" here is not the same tile as the Q,K,V tile. It's tiling the memory layout, while our Q,K,V are tiles of our algorithm. Unfortunately in our case, it's tiling...our tiles.

```cpp
// layouts are not filled in yet
using MyTiledCopy = decltype(make_tiled_copy(
    Copy_Atom<Atom, T>{}, // Atom
    Layout<Shape<>, Stride<>>{}, // Thread layout (who)
    Layout<Shape<>>{} // Value layout (what per thread)
));
```

The tiled copy function `make_tiled_copy` takes in the atom, the thread layout, and the values given to each thread. Our `Copy_Atom` is 128-bit wide chunk of 8 fp16 numbers, which is 8 values per thread. Given our row-major inputs, the output layout has to be: `Layout<Shape<_1, _8>>{}`. The layout is the thread layout, i.e. how you want to distribute your threads per tile. Assuming `kNThreads=128`, we have to give each thread a 128-bit chunk. The stride determines which 128-thread tile of memory comes next. The easiest strategy is to simply spread the tiles across along columns and then the rows, essentially filling it from the top like an upside-down cup.

Funnily enough, it gets slightly tricky here because of bank conflict optimization. Dao uses the same tiled copy setup for Q, K, V despite them having slightly different dimensions. We'll revsit this when we talk about bank conflicts, but for now, assume our smem block is of shape `(_, kBlockKSmem)`, where `kBlockKSmem` is the column width for all 3 tensors. We can compute the layout as:

```cpp
// pseudocode; assume static constexpr ints
int halfs_per_128bit_load = sizeof(uint128_t) / sizeof(half_t);
int threads_per_row = kBlockKSmem / halfs_per_128bit_load;
int num_thread_rows = kNThreads / threads_per_row;
int num_thread_cols = threads_per_row;
```

For `kBlockKSmem=64`, each row is 64 halfs or 8 128-bit loads, so 8 threads per row. With 128 threads, we cover $128/8=16$ rows per tile. The stride is simple: the column stride should move by static `_1{}` for the next 128-bit load. The row stride should move by the entire `num_thread_cols` chunk to the next row. Hence, our `Tiled_Copy` is:

```cpp
// Since these are constexpr, we use statics!
using TiledCopyQKV = decltype(make_tiled_copy(
    GmemCopyAtom{},
    Layout<Shape<Int<num_thread_rows>, Int<num_thread_cols>>,
            Stride<num_thread_cols, _1>>{},
    Layout<Shape<_1, _8>>{}
));
```

The way to think about this is that this `Tiled_Copy` is the tiling strategy for your source memory (GMEM in this case). All 128 threads load the first 128 contiguous 128-bit chunks, finish, then move onto the next 128 chunks until the entire GMEM section is copied. Even though this example is for a GMEM source, `Tiled_Copy` works between GMEM, SMEM, and per-thread registers. It doesn't know what anything is, it's just the floorplan, and we're responsible for providing the expected input.

### Tiled_Copy, Source and Destination
Our `Tiled_Copy` determines how our source is tiled, but we now have to configure the destination. The layout of the destination is determined by the destination tensor's tensor layout. The `Tiled_Copy` simply places the threads data in the "same place" it was loaded from. The destination layout can essentially be anything as long as it is compatible with the `Copy_Atom`. Since we have 128-bit loads/stores, we hope the destination tensor layout must accept aligned 8-half blocks (more on this in swizzling). For now, we can ignore what the output tensor is. `Tiled_Copy` has a specific pattern for copying between a source and a destination: a thread view, partitioning step, and then finally, the copy.

```cpp
// defining tiled copy
typename Traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
// what thread are we? let's get the slice of the data
// that belongs to thread tid
auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tid);
// partition thread Q gmem SOURCE tensor
Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
// partition thread Q smem DEST tensor
Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
// copy op: (tiled_copy, source, dest)
cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
```

In this example, assume `gQ` and `sQ` are correctly-defined GMEM and SMEM tensors. We first define our tiled copy blueprint. Then, we get the thread slice of this tiled copy, which translates our global tiled copy object to the values this thread actually fetches. Then, we partition the source and the destination, laying the thread blueprint on the source and destination tensors. Finally, we issue the copy operation.

> Example: Thread 0 takes the 0th (first) 128-bits, halfs 0-7. Then, it takes the 128th 128-bit chunk. Then the 256th, 384th, until the source is tiled. The intermediate thread tensor has shape `((1, 8), M, N)` where M, N represent the tile and 1, 8 is the value layout. It may not be this exactly, but it doesn't really matter as we don't usually have to work with the intermediate partition.

### GMEM and SMEM Tensors
Saved the easiest step for last. Let's define the `gX` and `sX` tensors for GMEM and SMEM.

CuTe provides us with a convenient API to retrieve the proper tensor tile from the source. It has the unfortunate side effect of being somewhat convoluted and ugly, but hey, it works.

```cpp
// gmem
Tensor mQ = make_tensor(
    make_gmem_ptr(reinterpret_cast<const cute::half_t *>(params.q_ptr) +
        batch_idx * params.q_batch_stride +
        head_idx * params.q_head_stride),
    make_shape(params.seqlen_q, params.head_dim),
    make_stride(params.q_row_stride, _1{}));
Tensor gQ = local_tile(mQ, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}),
                        make_coord(m_block, 0));
// smem
Tensor sQ = make_tensor(reinterpret_cast<cute::half_t *>(smem),
    SmemLayoutQ{});
Tensor sK =
    make_tensor(sQ.data() + size(sQ), SmemLayoutKV{});
```

This looks awful but the mechanism is quite simple. Each thread block operates on a unique block Q for some unique batch/head. We compute the batch and head index and offset into the Q tensor by the batch and head stride, arriving at that particular batch/head's Q tensor. CuTe has primitives like `make_gmem_ptr` and `make_smem_ptr` to tell the underlying engine to issue the correct PTX instructions for copying between GMEM, SMEM, and the register file. We provide it a layout so we can easily call `local_tile(tensor, tile_layout, coord)` to retrieve the tile of interest, in this case, the `mth` block of Q. It takes in a `Coord` which is the `(i, j)`-th tile according to `tile_layout`.

We could easily have made the mQ pointer point to the the start of the batch/head dimension and local tile into BH as well as `m_block`. The output PTX would be exactly the same--it's simply a matter of personal preference. The K and V gmem tensors iterate over all blocks along the d_h dimension, so their coord is given an underscore `_` to signal this to the compiler.

```cpp
Tensor gK = local_tile(mK, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}),
    make_coord(_, 0));
```

## SMEM->Registers
We will issue a second `Tiled_Copy` to copy from our SMEM to the registers. The copy pattern is mostly the same, but instead of simply transferring memory from SMEM to the registers, we must format the SMEM and registers for the tensor core matrix multiply-add (MMA) instructions.

Our first MMA GEMM is between Q and K. Since they are both in row-major format, the copy will work quite easily without much overhead. We will get into the tensor core instructions quite shortly, but for now, all we need to know is that Ampere natively supports 16x8x16 (MxNxK) MMAs out of the box. Each tensor op has shape $(16\times 16) \times (16\times 8) = (16\times 8)$.

$$C = A\times B + C$$

Each warp does one MMA in one tensor core cycle and the warps synchronize with one another to produce the final accumulated output. Each MMA is mapped to one warp, where A, B, and C are stored in **fragments** across all 32 threads in registers. NVIDIA selects the register mapping for each architecture, which is conveniently defined in CuTe via the `MMA_Atom`, which we will discuss more later. For now, all we know is that each thread must hold its share of A, B, and C (Q, K, accumulator) via the `Tiled_Copy`.

```cpp
using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>;
```

Our copy atom this time leverages the `LDSM` PTX instruction: Load from Shared Memory with the "N"ormal row-major/no-transpose layout. It moves 4 words = 128 bits per instruction, similar to our async load from before. However, this instruction is specialized to copy from shared memory to the correct registers for MMA, vectorizing per-warp loads and bypassing bank conflicts. However, unlike for GMEM->SMEM, our tiled copy has to be aware of the MMA layout as well as the relevant thread fragments, which differ between fragments A, B, and C.

### Tiled_MMA
Getting deja vu yet? This time, we define the tiling for the MMA GEMM. We define the following Tiled MMA atom:

```cpp
// TN means transposed-normal for AxB. It's a historical convention
// that you can search up.
// Practically, it means both A and B are row-major
using TiledMmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>
```

You might wonder, why 16x8x16 and not 16x16x16? Again, it's a hardware design choice made by NVIDIA engineers. There are a few reasons why:
1. Less register pressure: B and C fragments are both 16x8, reducing the total register footprint by 16x16 per warp.
2. More register re-use. Each A tile is used twice per B and C tile, reducing the number of simultaneous register reads.
3. Best "area of efficiency". NVIDIA certainly tested many combos and somehow found this size to be optimal.

This is by far not an exhaustive list, and tensor core shapes change generation-to-generation for a multitude of reasons. It's best to just use it as-is instead of wondering all day why it is this way. The TiledMMA atom conveniently defines which threads get which chunks and which registers are used for the MMA (TODO: printing). We now define the full `Tiled_MMA`:

```cpp
using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
    Layout<Shape<Int<kNWarps>, _1, _1>>,
    Tile<Int<16 * kNWarps>, _16, _16>>;
```

We chose 128 threads or 4 warps because each SM has 4 resident tensor cores, a sensible choice in order to maximize MMA throughput. For the layout, we tile across the M-dimension, which we take a slice from the left column of Q, and move across the K dimension. This allows each warp to compute a full output row. This makes it easy and efficient to warp-sychronize the online softmax statistics, such as the max and the expsum later down the line. Each tile is `kNWarps` stacked on top of each other; for a 16x8x16 MMA atom, our tile shape becomes $(M, N, K) = (16\cdot\text{kNWarps}, 16, 16)$. $N$ is 16, not 8, because we must aggregate across adjacent N-atoms to produce one $16x16$ output tile due to the 16x8 assymetry (TODO: image).

(TODO: MMA layout, shape)

### Tiled_Copy A, B, and C
This time, we need to make a different tiled copy for A, B, and C since the fragment registers are specific to each component per atom. The code patterns is mostly the same, with some SMEM->register specifics.

First, we create the register fragments for each thread according to the tiled mma:

```cpp
// create tiled mma
auto tiled_mma = TiledMma{};

// partition the fragments
auto thr_mma = tiled_mma.get_thread_slice(tid);
Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
Tensor tSrK = thr_mma.partition_fragment_B(sK);
```

Next, we create the tiled copy and partition SMEM for the copy transaction.

```cpp
// create Q, K tiled copy
auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);

// thread slice of MMA
auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid);
auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tid);

// partition SMEM
auto tSsQ = smem_thr_copy_Q.partition_S(sQ);
auto tSsK = smem_thr_copy_K.partition_S(sK);
```

Before the actual copy and GEMM, notice how we don't partition the destination registers. The registers must be known at compile-time and are already predefined for each thread. Registers are not even memory addressable the same way as GMEM or SMEM are. The Atom already knows the destination registers per thread; there is nothing to partition. Instead, we often have to retile the register to reconcile the LDSM and the MMA atom.

```cpp
Tensor tXrQ = smem_thr_copy_Q.retile_D(tCrQ);
Tensor tXrK = smem_thr_copy_K.retile_D(tCrK);
```

Retiling doesn't change the underlying registers, it simply allows us to map the 32x4 LDSM load to the specific fragment registers. By default, the 32x4 LDSM instruction is unaware of the underlying tensor op. We retile the fragments so that these u32 map to half_t and align their tile shapes in order for the eventual copy to "make sense."

### Aync Proxy and SMEM Copying
Now, we need to figure out what our SMEM layout should look like. If we simply stored them in the same format as our GMEM, we would quickly run into serious memory-bound issues due to **bank conflicts.** If you've made it this far, you hopefully know what these are already. But, if you don't:

> Bank Conflict: when multiple threads in a warp simultaneously request memory within the same bank in shared memory but across distinct addresses, we say there is a bank conflict. [Source](https://modal.com/gpu-glossary/perf/bank-conflict)

In order to enable highly parallel bandwidth in shared memory, NVIDIA stores the underlying data in 32-banks. It's a hardware design choice influenced by power consumption, wiring, latency, and speed. If you somehow figured out how to access any piece of data in SMEM concurrently for free, then you should instantly nominated for the Turing Prize or sent straight to a psychiatric ward. Unfortunately, dealing with bank conflicts is just a part of GPU programming.

Each of these 32 banks are 4-bytes wide--consecutive 4-byte chunks are stored in consecutive banks. For example, in a fp32 array: `float x[] = [0.f, 1.f, 2.f, 3.f]`, 0 would be in bank 0, 1 in bank 1, etc. If you had 32 threads in a warp simulatenously accessing 32 float32s in tandem, then you'd be accessing all 32 banks separately at a time, which is conflict-free. This "ideal" use case is by design.

```cpp
int bank = (byte_address / 4) % 32;
```

In FA2, we have 32 threads in each warp loading 32 128-bit chunks in tandem, which is 512 total bytes or 128 words[^1] or 4x32 bank accesses. This doesn't cause a 4-way bank conflict since the async proxy issues the load/store in 4 phases using quarter-warps, or 8 threads at a time.[^2] In phase 1, threads 0-7 load the first 8 128-bit chunks. In phase 2, threads 8-15 do the next 8, and so on and so forth. In each phase, each quarter warp issues a contiguous 8x128-bit or 128-byte coalesced copy, which targets all 32 banks. Hurrah, no bank conflict. So by design, our async copies perfectly copies our data using the full HBM bandwidth.

Don't celebrate early, because the bank conflicts don't arise from writing to SMEM but rather our subequent *reads* from SMEM during the SMEM->register copy







My AI learning guide had led me astray more times than I could count, but somehow I still had faith that somehow it wouldn't disappoint me this time.

# Appendix
[^1]: a word is 4 bytes
[^2]: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html
