**Foreword:** The title is a clickbait. I don't actually know how to scale attention to serve billion users, as I feel it is a really complicated problem with a lot of moving parts and optimizations to keep in mind, but in this blog I'm going to explore one of the approaches which I find really interesting. I got the idea to write this blog after watching Horace He's [talk](https://youtu.be/139UPjoq7Kw?si=8hoc2s7FZ7SRj4B1) with Jane Street. I hope I was able to do it justice. I've also linked resources which I referred to while piecing together this blog. Get a cup of coffee, sit in a nice place, and enjoy this blog.

#### Why isn't vanilla `self_attention` not used too much in practice?

"[Attention is all you need](https://arxiv.org/pdf/1706.03762)" was a pivotal paper that marked the revolution in the AI industry. All of the breakthroughs that we see today in the AI space can be traced back to that infamous paper. The authors of that paper are really influential too, but that's a story for another blog.

The key idea introduced in the paper, in the development of transformer architecture was that of scaled dot product attention and self attention. For each input sequence, three vectors are generated dynamically, namely queries(Q), keys(K) and values(V) which allows the model to focus on different parts of the input. These three vectors make one "head" of attention. The scores are calculated as:

![image](https://github.com/user-attachments/assets/caf420cf-1925-4007-ad25-40b12cfeb91e)

Performance has always been a bottleneck for using these models in downstream applications. The dot product step in the attention score calculation is quadratic in memory requirement. Another drawback which limits their application is numerical instability. When working with large sequences, the self attention score calculation can suffer from "avalanche effect" where small perturbations in the input can magnify the error during computations.

#### How do we optimize the attention mechanism?
<blockquote>
"Any optimization that is not about the bottleneck is an illusion of improvement"
</blockquote>

The core idea behind engineering is simple in theory, but is difficult in implementation. In our case, optimizing attention mechanism involves understanding the bottlenecks and building patches to improve performance. We established that memory requirements and numerical instability is one of the bottlenecks for attention, so what next should we do get performance gains?

One approach was the introduction of "fused" attention. For applications where memory is a constraint, having to compute the query and key matrix multiplication (Q.K) could be a bottleneck. A query, key vector of size 4096 x 4096 (standard in practice) and datatype `bfloat16` can take about 4096 x 4096 x 2 = 32MB of space. To avoid exhausting space and skipping the multiplication of query and key vectors, we can "fuse" the softmax computation with the second matrix multiplication. We make use of the fact(which is by no means trivial and is really clever) that in order to produce one block of the final result, we only need one block of the query vector. This implies that instead of multiplying the entire Q.K, we can compute one block at a time to produce one block of the output. For a block size of, say 128, the matrix multiplication q.K  has the shape 128 x 4096  which takes about(for the same `bfloat16` datatype) 128 x 4096 x 2 = 1MB`$ of space at once. Now, to get the final result, just look over all the blocks!! How cool is that! 

![image](https://github.com/user-attachments/assets/398c10f4-706c-46de-b7f2-358fc6ffcb85)

A great effort in this direction has been [Flash Attention](https://arxiv.org/pdf/2205.14135). Flash Attention improves Attention's time and space complexity by using techniques to improve efficiency.  The key here, similar to fused attention method is not storing large intermediate matrices. Flash attention does so by employing two established technique, namely tiling and recomputation. Tiling involves dividing the bigger attention matrix in manageable chunks(I'm skipping over a lot of details regarding softmax computation and the associated statistics). Recomputation involves recalculating attention matrix in the backward pass from blocks of Q,K, V in SRAM(this is so we don't have to store the quadratic intermediate values for the backward pass). Flash Attention is hardware specific, and the optimizations in it are specifically for GPUs. Tiling allows to implement the Flash Attention algorithm in one CUDA kernel and apply kernel fusion(kernel fusion "fuses" many element wise operations, so that they need not to be loaded multiple times). Flash Attention is also very clever when it comes to reducing numerical instability(I'm skipping over it for the sake of readability, however, I would highly encourage reading the Flash Attention [paper](https://arxiv.org/pdf/2205.14135))

![image](https://github.com/user-attachments/assets/da7d9dea-7576-4b01-ac45-2d1f3bdcb1c4)

There are have been other efforts in the space as well, which attention variants such as **RoPE**, **PrefixLM**, **Sliding Window Attention**, but the key idea behind all of these approaches is hardware specific optimization, often of modern hardware such as GPU. The goal then pivots to that of implementing hardware specific operations(often called kernel), and to be specific, memory bound operations, which optimizes the attention performance. Researchers tackle this problem by writing their own custom optimized kernels for their implementations, but just the sheer number of options to tune and the variety of new attention variants makes custom kernel option infeasible. Even worse, if the custom kernel doesn't fit into the existing optimized kernels, we are guaranteed slow runtimes. Horace He(who is the inspiration behind this blog) mentioned this is similar to "software lottery" for researchers(for those unaware, read Sara Hookr's paper on [Hardware Lottery](https://arxiv.org/pdf/2009.06489). It is one of my favorite papers, and I can't recommend it enough)

So, naturally, the question arises, how to solve this problem? 

#### Introducing - Flex Attention

Apart the different attention variants that are available today, researchers have tried implementing combinations of different variants(all with masking, biases, and other settings), for which there is no optimized kernel support. Given that there are exponential number of settings and various variants, we end up in a situation where we have less number of optimized kernels but a huge number of variants(hence the term software lottery).  So, the need for a solution that allows researchers to implement attention variants without having to deal with writing optimized kernels was dire, and that is where our main star of the blog comes in - **FlexAttention**(not to be confused with the [paper](https://vis-www.cs.umass.edu/flexattention/) on FlexAttention for VLMs).  

FlexAttention is available as an API by Pytorch through which researchers can implement their own attention variants in a few lines of Pytorch code. Behind the hood, Pytorch "fuses" the new implementation into a FlashAttention kernel(by leveraging `torch.compile`). One advantage of that is that the kernel doesn't take any extra memory and has performance competitive with handwritten kernels. Furthermore, since we are leveraging Pytorch, we can also generate the backward pass of the implementation automatically. Apart from all of this, we can also take advantage of sparsity in attention mask and get significant performance improvement over vanilla attention. Researchers just need to come up with new attention variants, and the rest is handled by Pytorch. How cool is that!!

Generally, FlexAttention is nearly as performant as a handwritten Trition kernel. If we talk about numbers, FlexAttention achieves **90% of FlashAttention2's performance** in the forward pass and **85% in the backward pass**. Interestingly, FlexAttention also **accelerated torchtune's sample packing throughput by 71%**.  FlexAttention has replaced the need for researchers to implement their own custom kernel(something that can take over a week) into a useful API that solved one of the main challenges of using attention in production.

#### FlexAttention Code Example

This section will demonstrate the use of FlexAttention through the Pytorch API(currently not available in the stable release, but it is there in the nightly releases). We'll go through one of the attention variants and see how Pytorch optimizes it.

Since the matrix multiplication step in the vanilla attention is the one which we need to optimize, Pytorch allows us to optimize that particular step by introducing a user-defined function `score_mod`, which allows us to modify the attention scores prior to softmax(surprisingly, this is sufficient for a majority of attention variants):

![image](https://github.com/user-attachments/assets/8a52e60a-d945-4c52-be2f-3ac269a9c3c3)


Behind the scenes, the `score_mod` function is fused into a single fused FlexAttention kernel. Let us solidify our understanding of the API by implementing a common attention variant - RoPE(relational positional encoding), something which is central to many models such as Llama, Mistral, Eleuther GPT-Neo and many more. The first step is implementing the `score_mod` function which has the following signature:

```python
def score_mod(score, b, h, q_idx, kv_idx):
	"""
	score: tensor ; dot product of query and key token.
	b: current element in batch.
	h: current head.
	q_idx: position in query
	kv_idx: position in key/value tensors.
	"""
	return score  #ideally, we want to return modified scores
```

In RoPE, instead of encoding the absolute distance in the queries and keys, the scores are based on relative distance between queries and keys. In the optimized FlexAttention implementation, the entire Q.K vector is not computed, leading to significant memory and performance improvements. For the case of RoPE, the `score_mod` function is as follows:

```python
def relative_positional(score, b, h, q_idx, kv_idx):
	return score + (q_idx-kv_idx)
```

Now, in order to use it end-to-end(including forwards and backwards), we can do it in one line of Pytorch as:

```python
from torch.nn.attention.flex_attention import flex_attention

flex_attention(query, key, value, score_mod=relative_positional).sum().backward()
```

Yes. It is that easy to get significant performance gains for a popular attention variant such as RoPE. The following graph shows just that:

![Pasted image 20250107222623](https://github.com/user-attachments/assets/ed3a8e6a-48ad-43df-b245-6be52b6f24d4)

#### Conclusion

FlexAttention for me, is one of the best examples of software engineering I've seen in recent times, as it demonstrated how difficult de-bottlenecking a complex problem is. The title is clickbait-ey as told, but I'm pretty sure, with the work that Pytorch team is doing, FlexAttention can help serve attention to a billion users efficiently.

**P.S** Compiler are really interesting(and hard)

###### Resources:
1.  [FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention](https://pytorch.org/blog/flexattention/)
2. [Building Machine Learning Systems for a Trillion Trillion Floating Point Operations](https://youtu.be/139UPjoq7Kw?si=Nz4llma9F3Yf008C)
3. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
4. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
5. [A friendly introduction to machine learning compilers and optimizers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html)
6. [Attention Gym-Examples for FlexAttention Attention Variants](https://github.com/pytorch-labs/attention-gym)
7. [Torchtune PR: Integrate Flex Attention](https://github.com/pytorch/torchtune/pull/1193)
