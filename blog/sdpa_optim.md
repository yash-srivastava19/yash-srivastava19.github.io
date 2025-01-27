Note: I wrote this article because I wanted to learn Pytorch internals.

## SDPA Optimization - Introduction
Scaled Dot Product Attention(abbreviated as SDPA), is a attention mechanism where the dot products between the dynamic vectors(query, key, value) are scaled down by `sqrt(d_k)`. The attention scores are calculated as:

![image](https://github.com/user-attachments/assets/d6fd19c2-2bd8-4204-b222-64e11ae7bd1e)

SDPA(or self attention) was a revolutionary discovery, first introduced in the "[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)" paper which formed the backbone of modern NLP applications. SDPA enhanced computation speed and allowed parallel computing of input sequence, allowing to capture meaningful relationship between tokens.

## Why SDPA Optimization?
In my FlexAttention [blog](https://yash-sri.xyz/blog/flex_attention), I explained in detail how the straightforward implementation of SDPA has quadratic compute and memory complexity with respect to sequence length. It is because of these bottlenecks, using optimized version of SDPA such as Flash Attention or Flex Attention are preferred for deployment. 

While I was working on the FlexAttention blog, and was beginning to understand how each approach optimized the standard SDPA(or different variants of SDPA), especially from memory constraints, I **found three different directions which I should explore** and experiment which is the most promising amongst them.

Explained in the next section in detail, my approach with this case study is to explore how different approaches stand against each other, and how much memory can we save in comparison to standard SDPA. This case study involves a lot of experiments and testing, and will be supported by code wherever necessary. I've presented the case study as a work log, so you can see how I went from idea X to idea Y, following the initial chain of thought. Results and further directions are discussed in the end. 
## Initial Approach
As mentioned in the previous section, I found three axes along which I hypothesized we can explore(either individually or grouped together) each direction and study how much memory overhead we can reduce. My initial approach is given [here](https://yash-sri.xyz/scratchpad) on my scratchpad.

**Approach 1:** If we think from first principles, the way to reduce memory footprint is to either - **reduce the size of the model**, or **optimize the computation heavy step**. As we know that SDPA scales quadrupedally with sequence length, one trivial axes to explore was **reducing the sequence length**. So, the initial 3 directions I explored can be visualized as:

![image](https://github.com/user-attachments/assets/fa410aa3-1768-4ea8-87d4-6019a1b52a00)

1. **KV Cache** approach reduced the memory extensive matrix multiplication step between key and value matrices.
2. For Precision, we looked into **Quantization** approaches.
3. **Sequence length** reduction was another approach.

The way forward, as mentioned, was to explore each direction individually, and conduct tests to see which of these is a promising direction. 

However, one issue I found in this was that most of the times, the sequence length is a parameter we don't usually get to control, given the frontier models these days have context length constraints. Also, sequence length is a model hyperparameter, and we are to **optimize for a model agnostic axes**(something which does not depend on the model and its parameters, and we can simply drop in replace the optimized SDPA variant), we can't generalize our results. 

Another crazy idea I had in mind was to optimize along all the three axes grouped together, and find a "Goldilocks" zone which gives the least memory consuming SDPA variant. We'll look into this idea later, but now, let us move on which model agnostic axes we chose to explore, given sequence length was out of scope.

**Approach 2:** Going from approach 1 to approach 2 wasn't easy, and with the added task of finding a model agnostic axes, it took me a lot of time to find a promising direction to explore. **The idea occurred to me while I was going through the Flash Attention paper**, especially the introduction section:

![image](https://github.com/user-attachments/assets/b2cca6d8-d9da-4cf8-b22f-b5457fd91c24)

The Flash Attention paper argued that initial approaches to reduce compute requirements focused on FLOP reduction, while the techniques introduced in the paper relied on IO aware attention algorithms, which is to say, made use of device specific optimization. However, frameworks such as Pytorch doesn't explicitly provide fine grained control over memory access, so the question was - if we were to explore this axis(device specific optimizations), how do we go about it?

After looking a lot into the void, I found the answer in the most bizarre of place - my own [blog](https://yash-sri.xyz/blog/flex_attention) on FlexAttention. FlexAttention was the missing piece of the puzzle, and it fitted in perfectly. Sometimes, things work out in the most amazing of the ways. The answer was device specific optimization, something which FlexAttention provided fine control over. Now, the axes looked like:

![image](https://github.com/user-attachments/assets/0de6ea74-8de6-48e9-8477-47391d255bc9)

We'll now move on to how we went about by experimenting each of the axes individually. These are **presented in a worklog type fashion**, so the reader can follow the chain of thoughts. 

## Timeline Of Events
To make things easier for us, for this section, I'll be dividing it into 3 subsection, one for each axes. Within each subsection, I'll explain the core concept of the approach, how did we implement it, any problems that we came across, and how did we manage to remedy it, and what parameters we can tune. Code will be provided wherever required. 

### KV Cache
To start things off, I started by exploring the KV Cache axes, as it is the most straightforward and easy to optimize. KV cache, like any other cache stores calculations so we don't need to recompute in a given context. In KV Cache, key-value pairs derived from self attention layers is stored. One important point to note that KV cache is only used for decoder only models. This picture summarizes it pretty well:

![image](https://github.com/user-attachments/assets/af5ca4a4-37d6-4cd1-b5f7-26bd7070689b)

Although HF provides pre-packaged models with KV Cache, the implementation is pretty straightforward. We maintain a cache data structure, and in the attention calculation step, we reuse the values from previous context. Here's the accompanying code, which is sufficient for us to conduct experiments. (Image Courtesy - [HF](https://huggingface.co/blog/kv-cache-quantization))

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CachedSDPA(nn.Module):
    def __init__(self, max_seq_len, head_dim):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.cache_k = None
        self.cache_v = None
        self.cur_len = 0
    
    def forward(self, q, k, v, is_causal=True):
        # q, k, v: [heads, seq_len, head_dim]
        
        # Handle incremental state through cache.
        if self.cache_k is not None:
            k = torch.cat([self.cache_k, k], dim=1)
            v = torch.cat([self.cache_v, v], dim=1)
        
        # Update cache
        self.cache_k = k
        self.cache_v = v
        self.cur_len = k.shape[2]
        
        # Use PyTorch's native SDPA with incremental state
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=is_causal
        )
        
        return out

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None
        self.cur_len = 0
```

For the given implementation, the only parameters we can optimize for is sequence length, and hidden dimensions. 

### Quantization

Quantization, in the AI space is a fancy word for reducing the precision of numerical values to save memory. In this, the numerical value is truncated to fit within the required precision format, ideally to a less memory consuming data type. Although this results in loss of information, a careful selection of quantization parameters can minimize the loss while still achieving satisfactory performance. 

![image](https://github.com/user-attachments/assets/f658582e-f374-4e94-8632-236bb49c0a0f)

Pytorch provides a flexible API to dynamically quantize models, post training into a model of lower "granularity". (Image courtesy - [Maarten Grootendorst](https://www.maartengrootendorst.com/blog/quantization/) ) For the experiment we are performing, I'm using the [Post Training Dynamic Quantization](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html) method. The API is explained in the code given below:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

class SmolAttention(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(query, key, value)

m = SmolAttention()
# This is a clever fix:
m_q = torch.ao.quantization.quantize_dynamic(m {nn.Linear, nn.Linear, nn.Linear},dtype=torch.qint8)
```

There is a small catch however, the PTDQ step is not currently available for SDPA, so in order to quantize, we had to implement SDPA from scratch using `nn.Linear` layers, for which quantization is available.

### Flex Attention
For a gentle introduction to Flex Attention, refer to my previous [blog](https://yash-sri.xyz/blog/flex_attention). For the purpose of this experiment, we'll be testing Flex Attention with a variety of score modifying functions to assess which one of them is memory intensive, which `no_op` function given in the code below the benchmark to measure our performance against.

![image](https://github.com/user-attachments/assets/f6b10cda-3099-4feb-af3a-522fe64c0017)

The parameters we can tune for this experiment is `batch_size`, `seq_len` and `hid_dims`. The minimal code to use FlexAttention using the Pytorch API is given below. For a more detailed view on how it works under the hood and why is it awesome, refer to my previous [blog](https://yash-sri.xyz/blog/flex_attention), or the original [Jane Street talk](https://youtu.be/139UPjoq7Kw?si=Nz4llma9F3Yf008C).

```python
import torch
from torch.nn.attention.flex_attention import flex_attention

# The base score modifier, other functions will be added later.
@torch.compile 
def no_op(score, b, h, q_idx, kv_idx):
    return score

flex_attention(q, k, v, score_mod=no_op)
```

## Experiments Performed
Say you have a hypothesis is mind, and want to test whether it is true or not. For people with statistics background, this is as easy as performing standard tests in practice and comparing it with baselines. Now, that sounds easy, but let's take a step back - how do you design experiments?

Designing experiments for CS is a little different than basic sciences because here we are **not playing with instruments, but rather instructions** - which is difficult as there is a scope for making a lot of mistakes. In my opinion and practice, thinking about stuff from first principles helps you come up with an hypothesis, and iterating till the end of the world helps you design experiments. The steps to conduct research are pretty straightforward - establish an hypothesis, setup a baseline, add and isolate the feature, test against the baseline and report your result. Depending on your experience and practice, this can take anywhere from a day to years. Along this journey, you learn new things, you understand the bottlenecks, you make tools to make your life easier, and you get used to the scientific method. **That's the red pill of research, it is not fancy, but it is a hell lot of interesting**.

For the purpose of this experiment, I took the same approach. The next few sections describe how I experimented with the different axes, and how do they compare to be baseline. I will be reporting both the memory profile and bottleneck analysis(something which is available from the Pytorch API). Based on the previous sections, where I described what parameters we can tune, a report on those, plus any problems we ran along the way are given as well.    

### Nugget: `PerformanceAnalysis` 
I mentioned in the previous paragraph that making tools that improve your iteration speed is crucial for any research project. Again, there's no blue pill, **you make things from scratch that make your life easier**. For the experiments I performed, I found that reporting the results(both memory profile and bottleneck analysis) in a consistent manner was the bottleneck, so I made a class to do exactly that for me - it was the `PerformanceAnalysis` class. Here's the accompanying code for initial version of the class.

```python
## There is real alpha in this.
import torch.utils.benchmark as benchmark
from torch.profiler import profile, record_function, ProfilerActivity 

class PerformanceAnalysis:
    def __init__(self, func, inputs, *args):
        self.m = func # this could be a nn.Module, function, or anything else.
        self.q, self.k, self.v = inputs #unpack  the inputs, we need it in this format only.
        self.args = args
        self.setup()
        
    def setup(self):
        pass 

    def profile(self):
        # Update, can add multiple settings.
        if self.q.device == "cuda":
            act = ProfilerActivity.CUDA
        else:
            act = ProfilerActivity.CPU
        
        with profile(activities=[act], record_shapes=True, profile_memory=True) as prof4:
            with record_function("model_inference"):
                if self.args:
                    self.m(self.q, self.k, self.v, self.args[0])
                else:
                    self.m(self.q, self.k, self.v)
        
        return prof4.key_averages().table()
    
    def benchmark(self, use_threads=True, num_exprs=100):
        # Custom Logic to make the statement for Benchmark Timer

        # If we figure out the class/function, setup part is done. Just need to fire out how the Q,K,V names are made.
        # Update: No need. We did it lol.
        # Update: Stuck on the benchmark class/function thing. 
        # Update: Made it work after scraping lol.
        
        import inspect
        if inspect.isfunction(self.m):
            func_name = f"{self.m.__name__}"
            # print(func_name)
            name = func_name
            if self.args:
                module_name = f"{self.args[0].__name__}"
                stmt_str = f"M(Q, K, V, {module_name})"
                setup_str = f"from __main__ import {module_name}"
            else:
                stmt_str = f"M(Q, K, V)"
                setup_str = f"from __main__ import {func_name}"
        else: # it must be a class. Add checks, but for now we can proceed.
            class_name = f"{self.m.__class__.__name__}"
            name = class_name
            if self.args:
                module_name = f"{self.args[0]}"
                stmt_str = f"M(Q, K, V, {module_name})"
                setup_str = f"from __main__ import {class_name}, {module_name}"
            else:
                stmt_str = f"M(Q, K, V)"
                setup_str = f"from __main__ import {class_name}"

        if use_threads:
            # Sorted
            num_threads = torch.get_num_threads()
            t0 = benchmark.Timer(
                stmt = stmt_str,
                setup = setup_str,
                globals={'M':self.m, 'Q':self.q, 'K': self.k, 'V':self.v},
                num_threads=num_threads,
                label = f"Multi Threaded SDPA - {name}"
            )
        else:
            t0 = benchmark.Timer(
                stmt = stmt_str,
                setup = setup_str,
                globals={'M': self.m, 'Q':self.q, 'K': self.k, 'V':self.v},
                label = f"Single Threaded SDPA - {name}"
            )

        return t0.timeit(num_exprs)
        

    def run(self):
        pass 

    def report(self):
	    # I made it more
        return f"""
			{self.profile()}
			{self.benchmark()}
		"""

```
The class method `profile` uses the [torch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) to profile the model and analyze the memory bottlenecks(along each layer of the model). The method `benchmark` on the other hand, makes use of the Pytorch built-in [benchmark](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)(similar to `timeit.timer`) to perform speed analysis. The class is not pretty, does not follow the best practices, but as long the experiment is not scaled and it gets the job done, I won't touch it. Remember, in CS, the event loop is - **make it work -> make it right -> make it fast. In that order**. 

### Nugget: `F.scaled_dot_product_attention`
While I was doing my reading for this blog, I was very fascinated by the way `F.scaled_dot_product_attention` function in the Pytorch API is implemented. Now, although this function can be implemented as per the Attention Is All You Need paper, Pytorch uses a "fused" implementation(again, I explained in my FlexAttention blog mathematically how fused implementation reduces overhead) to provide performance benefits. There are three methods(for CUDA tensor) which the SDPA function dispatches to - [FlashAttention](https://arxiv.org/abs/2205.14135), [Memory-Efficient Attention](https://github.com/facebookresearch/xformers) and a C++ implementation of SDPA. By default, Pytorch dispatches implicitly to any on the said implementations, we can explicitly control the dispatch. The code to do that is given below(Courtesy: [Pytorch Documentation](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html))

```python
# Lets define a helpful benchmarking function:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

# Lets define the hyper-parameters of our input
query = torch.randn(size=(seq_len, 128, 128), requires_grad=True, device="cpu")
key = torch.randn(size=(seq_len, 128, 128), requires_grad=True, device="cpu")
value = torch.randn(size=(seq_len, 128, 128), requires_grad=True, device="cpu")


print(f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")

# Lets explore the speed of each of the 3 implementations
from torch.nn.attention import SDPBackend, sdpa_kernel


with sdpa_kernel(SDPBackend.MATH):
    math_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, Q, K, V)
    print(f"The math implementation runs in {math_time:.3f} microseconds")

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    try:
        flash_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, Q, K, V)
        print(f"The flash attention implementation runs in {flash_time:.3f} microseconds")
    except RuntimeError as e:
        print(f"FlashAttention is not supported. See warnings for reasons: {e}")

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    try:
        efficient_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, Q, K, V)
        print(f"The memory efficient implementation runs in {efficient_time:.3f} microseconds")
    except RuntimeError:
        print("EfficientAttention is not supported. See warnings for reasons.")
```

The main purpose to include this nugget is to maybe individually test against each one of these variant, and explore the performance gains. Now, we might not include it in this blog, as it already getting very information dense. It is up to the reader to go on an expedition with the code and experiment for themselves. Trust me, it will be really awesome.

### KV Cache Experiment 
As mentioned, we profiled the model against a set of sequence lengths, hidden dimensions. Here's the code we used for experimenting with a set of hyperparameters.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CachedSDPA(nn.Module):
    ## Add from the pervious section. Cropped for readability.
    pass

# Now, these functions on their own won't make too much sense. Changing hyperparams, and similar things will actually be nice.
def test_cached_sdpa_memory():
    pa.profile()

def test_cached_sdpa_benchmark():
    pa.benchmark()

seq_lengths = [4, 8, 10, 12] 
hidden_dims = [16, 32, 64, 128]

# We are running into CUDA out of memory erros frequently, so stay in the required zone. These are the limits to which we can push.

print("*********************** Cached SDPA Tests *************************")

for i, (sl, hd) in enumerate(zip(seq_lengths, hidden_dims)):
    print(f"=============== Experiment {i+1}: seq_len={sl}, hidden_dim={hd} =========================")
    q = torch.randn(size=(sl, hd, hd), requires_grad=True, device="cuda")
    k = torch.randn(size=(sl, hd, hd), requires_grad=True, device="cuda")
    v = torch.randn(size=(sl, hd, hd), requires_grad=True, device="cuda")

    m = CachedSDPA(sl, hd)

    pa = PerformanceAnalysis(m, (q,k,v))
    print("=============== Memory Profile: ===========================")
    print(pa.profile())
    
    print("=============== Benchmark Report: =========================")
    print(pa.benchmark())

    torch.cuda.empty_cache()
    del q,k,v,m,pa
```

Since we are performing the experiments in a loop, after each iteration, especially when we are using CUDA tensors, we need to release the block of memory - otherwise we'll run into CUDA OOM error. While Pytorch provides `torch.cuda.empty_cache()` to clear cache and free the memory, the [right approach](https://discuss.pytorch.org/t/free-all-gpu-memory-used-in-between-runs/168202/2) is to delete all objects and references pointing to objects allocating GPU resources, hence after each iteration, we delete the variables used in the context.

### Quantization Experiment
The approach for quantization is similar to the KV Cache, we tune the sequence length and hidden dimensions. Here's the experimentation code:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

class SmolAttention(nn.Module):
    # Cropped for readability.
    pass

seq_lengths = [4, 8, 10, 12] 
hidden_dims = [16, 32, 64, 128]

print("*********************** Non - Quantized SDPA Tests *************************")

for i, (sl, hd) in enumerate(zip(seq_lengths, hidden_dims)):
    print(f"=============== Experiment {i+1}: seq_len={sl}, hidden_dim={hd} =========================")
    q = torch.randn(size=(sl, hd, hd), requires_grad=True, device="cpu")
    k = torch.randn(size=(sl, hd, hd), requires_grad=True, device="cpu")
    v = torch.randn(size=(sl, hd, hd), requires_grad=True, device="cpu")

    m = SmolAttention()
    m_q = torch.ao.quantization.quantize_dynamic(m,{nn.Linear, nn.Linear, nn.Linear},dtype=torch.qint8)

    print("Non-Quantized Model: ")
    pa = PerformanceAnalysis(m, (q,k,v))
    print("=============== Memory Profile: ===========================")
    print(pa.profile())
    
    print("=============== Benchmark Report: =========================")
    print(pa.benchmark())
    
    print("Quantized Model: ")
    pa = PerformanceAnalysis(m_q, (q,k,v))
    print("=============== Memory Profile: ===========================")
    print(pa.profile())
    
    print("=============== Benchmark Report: =========================")
    print(pa.benchmark())

    del q,k,v,m,m_q,pa
```

Since in this experiment also we use CUDA tensors, we need to release memory after each iteration.

### FlexAttention Experiment
With FlexAttention, we have the freedom to optimize the approach using different score modifiers. For the baseline, we'll go ahead with `no_op` score modifier - which returns the score. For subsequent versions of FlexAttention, we'll use different modifiers and report the results in the next section. The tunable parameters are batch size, sequence length, and hidden dimensions. The accompanying code is given below:

```python
import torch
from torch.nn.attention.flex_attention import flex_attention

# No-op score modifier, subsequent modifiers will be added later.
@torch.compile 
def no_op(score, b, h, q_idx, kv_idx):
    return score

print("*********************** FlexAttention SDPA Tests *************************")

batch_sizes = [8, 16, 32]
seq_lengths = [8, 16, 32]
hidden_dims = [64, 128, 256]


for i, (bs, sl, hd) in enumerate(zip(batch_sizes, seq_lengths, hidden_dims)):
    print(f"=============== Experiment {i+1}: batch_size={bs} seq_len={sl}, hidden_dim={hd} =========================")
    q = torch.randn(size=(bs, sl, hd, hd), requires_grad=True, device="cuda")
    k = torch.randn(size=(bs, sl, hd, hd), requires_grad=True, device="cuda")
    v = torch.randn(size=(bs, sl, hd, hd), requires_grad=True, device="cuda")

    
    pa = PerformanceAnalysis(flex_attention, (q,k,v), no_op)    # extra score_mod is added
    print("=============== Memory Profile: ===========================")
    print(pa.profile())
    
    print("=============== Benchmark Report: =========================")
    print(pa.benchmark())
    
    del q,k,v,pa
```

The results from all the experiments are given in the next section. Before you go ahead and peek into it, **which method do you reckon would be the best?** Place your bets, and then, move ahead. Here are the results:

## Results
In the previous section, we described the experiments and which parameters we can tune for different axes. For the subsequent sections, we have reported the axes, the parameters and options we used and report the CPU/GPU time, benchmark time.

### KV Cache Test Result

**Parameters:**
Sequence Length - 4
Hidden Dimensions - 16
Number Threads - 2

**Results:**
Memory Profile: CPU Total Time: 338.52 ms
Benchmark Profile: 504.01 ms

![image](https://github.com/user-attachments/assets/6876c1f7-d36f-45d6-a696-d4661edb533f)

**Parameters:**
Sequence Length - 8
Hidden Dimensions - 32
Number Threads - 2

**Results:**
Memory Profile: CPU Total Time: 16.041 ms
Benchmark Profile: 567.59 ms

![image](https://github.com/user-attachments/assets/6b5490b3-fef6-4391-b7fd-733489192478)

**Parameters:**
Sequence Length - 10
Hidden Dimensions - 64
Number Threads - 2

**Results:**
Memory Profile: CPU Total Time: 4.388 ms
Benchmark Profile: 1.60 ms

![image](https://github.com/user-attachments/assets/64ed6da7-4711-4dd5-bbb0-affa5b079514)

**Parameters:**
Sequence Length - 12
Hidden Dimensions - 128
Number Threads - 2

**Results:**
Memory Profile: CPU Total Time: 338.52 ms
Benchmark Profile: 4.96 ms

![image](https://github.com/user-attachments/assets/a4e310e9-7eaf-409f-9c28-ce6b9ea673c1)

### Quantized Test Results
**Parameters:**
Sequence Length - 4
Hidden Dimensions - 16
Number Threads - 2

**Results:**
**Non-Quantized**
Memory Profile: CPU Total Time: 7.047 ms
Benchmark Profile: 243.24 us

**Quantized**
Memory Profile: CPU Total Time: 15.533 ms
Benchmark Profile: 442.75 us

![image](https://github.com/user-attachments/assets/e1c14c8f-6789-4d63-a16e-8aab3c09a1bd)

**Parameters:**
Sequence Length - 8
Hidden Dimensions - 32
Number Threads - 2

**Results:**
**Non-Quantized**
Memory Profile: CPU Total Time: 1.083 ms
Benchmark Profile: 347.22 us

**Quantized**
Memory Profile: CPU Total Time: 1.266 ms
Benchmark Profile: 552.11 us

![image](https://github.com/user-attachments/assets/cf58d707-6a48-48a5-8e8f-a99ce324a5a8)

**Parameters:**
Sequence Length - 10
Hidden Dimensions - 64
Number Threads - 2

**Results:**
**Non-Quantized**
Memory Profile: CPU Total Time: 4.650 ms
Benchmark Profile: 664.73 us

**Quantized**
Memory Profile: CPU Total Time: 1.469 ms
Benchmark Profile: 901.69 us

![image](https://github.com/user-attachments/assets/eb82c941-e1b0-4bdb-bb57-5679a1e761f8)

**Parameters:**
Sequence Length - 12
Hidden Dimensions - 128
Number Threads - 2

**Results:**
**Non-Quantized**
Memory Profile: CPU Total Time: 4.650 ms
Benchmark Profile: 664.73 us

**Quantized**
Memory Profile: CPU Total Time: 4.222 ms
Benchmark Profile: 2.90 ms

![image](https://github.com/user-attachments/assets/debcf5f0-0ac9-432d-9c67-4f42ed3aa816)

### FlexAttention Test Results 

**Parameters:**
Batch Size - 4
Sequence Length - 8
Hidden Dimensions - 64
Number Threads - 2

**Results:**
Memory Profile: CPU Total Time: 516.692 ms
Benchmark Profile: 8.45 ms


![image](https://github.com/user-attachments/assets/d97a8aa0-21ae-4b94-8338-91c51bc2997c)

**Parameters:**
Batch Size - 16
Sequence Length - 16
Hidden Dimensions - 128
Number Threads - 2

**Results:**
Memory Profile: CPU Total Time: 348.271 ms
Benchmark Profile: 9.60 ms

![image](https://github.com/user-attachments/assets/0187a934-1f02-4416-8cc2-a38190b08f03)

**Parameters:**
Batch Size - 32
Sequence Length - 32
Hidden Dimensions - 256
Number Threads - 2

**Results:**
Memory Profile: CPU Total Time: 180.934 ms
Benchmark Profile: 48.30 ms

![image](https://github.com/user-attachments/assets/e1ae1bd7-e95c-4e39-acdb-89e09786a0b5)

Based on the experiments performed, we can clearly see the advantages of FlexAttention. The total CPU memory usage is very less, and the speed is also justified given the limits to which we pushed the model. Other methods also compare well with to it, but the task now is to understand the tradeoffs between all of them, and find the "goldilocks" zone. 

From what I can see, I think one way to approach the said problem is to perform a [sensitivity analysis](https://en.wikipedia.org/wiki/Sensitivity_analysis) over the input space, and move towards the direction which performs well given the tradeoffs and requirements of the "memory efficient" model. While we can formulate this as an optimization problem, I'm running out of ways to perform efficient search of input parameters. The problem more complex problem is combining different approaches, for which I'm not particularly sure how to proceed.

This feels like a search problem. The question is, how can I do that? If you have any ideas, DM me on Twitter or mail me here. I find this problem really interesting, and feel can learn a lot from this experiment.

## Additional Materials
1. [Kaggle Notebook](https://www.kaggle.com/code/yashsrivastava51213/sdpabenchmarking) - SDPA Optimization.
2. [Github Repository](https://github.com/yash-srivastava19/sdpa_optimization) - SDPA Optimization.
