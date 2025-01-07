# Blogs
Some of my learnings about the stuff in tech. Personal, but very helpful for others as well.
---
## How to Scale Attention Models for Billions Users?

**Foreword:** The title is a clickbait. I don't actually know how to scale attention to serve billion users, as I feel it is a really complicated problem with a lot of moving parts and optimizations to keep in mind, but in this blog I'm going to explore one of the approaches which I find really interesting. I got the idea to write this blog after watching Horace He's [talk](https://youtu.be/139UPjoq7Kw?si=8hoc2s7FZ7SRj4B1) with Jane Street. I hope I was able to do it justice. I've also linked resources which I referred to while piecing together this blog. Get a cup of coffee, sit in a nice place, and enjoy this blog.

### Why isn't vanilla `self_attention` not used too much in practice?

"[Attention is all you need](https://arxiv.org/pdf/1706.03762)" was a pivotal paper that marked the revolution in the AI industry. All of the breakthroughs that we see today in the AI space can be traced back to that infamous paper. The authors of that paper are really influential too, but that's a story for another blog.

The key idea introduced in the paper, in the development of transformer architecture was that of scaled dot product attention and self attention. For each input sequence, three vectors are generated dynamically, namely queries(`$Q$`), keys(`$K$`) and values(`$V$`) which allows the model to focus on different parts of the input. These three vectors make one "head" of attention. The scores are calculated as:

`$$ Attention(Q, K,V) = softmax(\frac{QK^T}{\sqrt{d_{k}}})V $$`

Performance has always been a bottleneck for using these models in downstream applications. The dot product step in the attention score calculation is quadratic ($O(n^2)$ )in memory requirement. Another drawback which limits their application is numerical instability. When working with large sequences, the self attention score calculation can suffer from "avalanche effect" where small perturbations in the input can magnify the error during computations.

### How do we optimize the attention mechanism?
<blockquote>
"Any optimization that is not about the bottleneck is an illusion of improvement"
</blockquote>

The core idea behind engineering is simple in theory, but is difficult in implementation. In our case, optimizing attention mechanism involves understanding the bottlenecks and building patches to improve performance. We established that memory requirements and numerical instability is one of the bottlenecks for attention, so what next should we do get performance gains?

One approach was the introduction of "fused" attention. For applications where memory is a constraint, having to compute the query and key matrix multiplication ($Q . K^T$ ) could be a bottleneck. A query, key vector of size $`4096 \times 4096`$ (standard in practice) and datatype `bloat16` can take about $`4096 \times 4096 \times 2 \approx 32MB`$ of space. To avoid exhausting space and skipping the multiplication of query and key vectors, we can "fuse" the softmax computation with the second matrix multiplication. We make use of the fact(which is by no means trivial and is really clever) that in order to produce one block of the final result, we only need one block of the query vector. This implies that instead of multiplying the entire $`Q.K`$, we can compute one block at a time to produce one block of the output. For a block size of, say $`128`$, the matrix multiplication $`q.K^T`$  has the shape $`128 \times 4096`$  which takes about(for the same `bfloat16` datatype) $`128 \times 4096 \times 2 \approx 1MB`$ of space at once. Now, to get the final result, just look over all the blocks!! How cool is that! 

![image](https://github.com/user-attachments/assets/398c10f4-706c-46de-b7f2-358fc6ffcb85)


A great effort in this direction has been [Flash Attention](https://arxiv.org/pdf/2205.14135). Flash Attention improves Attention's time and space complexity by using techniques to improve efficiency.  The key here, similar to fused attention method is not storing large intermediate matrices. Flash attention does so by employing two established technique, namely tiling and recomputation. Tiling involves dividing the bigger attention matrix in manageable chunks(I'm skipping over a lot of details regarding softmax computation and the associated statistics). Recomputation involves recalculating attention matrix in the backward pass from blocks of $Q,K, V$ in SRAM(this is so we don't have to store the $O(n^2)$ intermediate values for the backward pass). Flash Attention is hardware specific, and the optimizations in it are specifically for GPUs. Tiling allows to implement the Flash Attention algorithm in one CUDA kernel and apply kernel fusion(kernel fusion "fuses" many element wise operations, so that they need not to be loaded multiple times). Flash Attention is also very clever when it comes to reducing numerical instability(I'm skipping over it for the sake of readability, however, I would highly encourage reading the Flash Attention [paper](https://arxiv.org/pdf/2205.14135))

![image](https://github.com/user-attachments/assets/da7d9dea-7576-4b01-ac45-2d1f3bdcb1c4)

There are have been other efforts in the space as well, which attention variants such as **RoPE**, **PrefixLM**, **Sliding Window Attention**, but the key idea behind all of these approaches is hardware specific optimization, often of modern hardware such as GPU. The goal then pivots to that of implementing hardware specific operations(often called kernel), and to be specific, memory bound operations, which optimizes the attention performance. Researchers tackle this problem by writing their own custom optimized kernels for their implementations, but just the sheer number of options to tune and the variety of new attention variants makes custom kernel option infeasible. Even worse, if the custom kernel doesn't fit into the existing optimized kernels, we are guaranteed slow runtimes. Horace He(who is the inspiration behind this blog) mentioned this is similar to "software lottery" for researchers(for those unaware, read Sara Hookr's paper on [Hardware Lottery](https://arxiv.org/pdf/2009.06489). It is one of my favorite papers, and I can't recommend it enough)

So, naturally, the question arises, how to solve this problem? 

### Introducing - Flex Attention

Apart the different attention variants that are available today, researchers have tried implementing combinations of different variants(all with masking, biases, and other settings), for which there is no optimized kernel support. Given that there are exponential number of settings and various variants, we end up in a situation where we have less number of optimized kernels but a huge number of variants(hence the term software lottery).  So, the need for a solution that allows researchers to implement attention variants without having to deal with writing optimized kernels was dire, and that is where our main star of the blog comes in - **FlexAttention**(not to be confused with the [paper](https://vis-www.cs.umass.edu/flexattention/) on FlexAttention for VLMs).  

FlexAttention is available as an API by Pytorch through which researchers can implement their own attention variants in a few lines of Pytorch code. Behind the hood, Pytorch "fuses" the new implementation into a FlashAttention kernel(by leveraging `torch.compile`). One advantage of that is that the kernel doesn't take any extra memory and has performance competitive with handwritten kernels. Furthermore, since we are leveraging Pytorch, we can also generate the backward pass of the implementation automatically. Apart from all of this, we can also take advantage of sparsity in attention mask and get significant performance improvement over vanilla attention. Researchers just need to come up with new attention variants, and the rest is handled by Pytorch. How cool is that!!

Generally, FlexAttention is nearly as performant as a handwritten Trition kernel. If we talk about numbers, FlexAttention achieves **90% of FlashAttention2's performance** in the forward pass and **85% in the backward pass**. Interestingly, FlexAttention also **accelerated torchtune's sample packing throughput by 71%**.  FlexAttention has replaced the need for researchers to implement their own custom kernel(something that can take over a week) into a useful API that solved one of the main challenges of using attention in production.

### FlexAttention Code Example

This section will demonstrate the use of FlexAttention through the Pytorch API(currently not available in the stable release, but it is there in the nightly releases). We'll go through one of the attention variants and see how Pytorch optimizes it.

Since the matrix multiplication step in the vanilla attention is the one which we need to optimize, Pytorch allows us to optimize that particular step by introducing a user-defined function `score_mod`, which allows us to modify the attention scores prior to softmax(surprisingly, this is sufficient for a majority of attention variants):

$$FlexAttention(Q, K, V) = softmax(score\_mod(\frac{QK^T}{\sqrt{d_k}}))V$$

Behind the scenes, the `score_mod` function is fused into a single fused FlexAttention kernel. Let us solidify our understanding of the API by implementing a common attention variant - RoPE(relational positional encoding), something which is central to Eleuther GPT-NeoX models. The first step is implementing the `score_mod` function which has the following signature:

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

In RoPE, instead of encoding the absolute distance in the queries and keys, the scores are based on relative distance between queries and keys. In the optimized FlexAttention implementation, the entire $Q \times K$ vector is not computed, leading to significant memory and performance improvements. For the case of RoPE, the `score_mod` function is as follows:

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

### Conclusion

FlexAttention for me, is one of the best examples of software engineering I've seen in recent times, as it demonstrated how difficult de-bottlenecking a complex problem is. The title is clickbait-ey as told, but I'm pretty sure, with the work that Pytorch team is doing, FlexAttention can help serve attention to a billion users efficiently.

**P.S** Compiler are really interesting(and hard)

##### Resources:
1.  [FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention](https://pytorch.org/blog/flexattention/)
2. [Building Machine Learning Systems for a Trillion Trillion Floating Point Operations](https://youtu.be/139UPjoq7Kw?si=Nz4llma9F3Yf008C)
3. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
4. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
5. [A friendly introduction to machine learning compilers and optimizers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html)
6. [Attention Gym-Examples for FlexAttention Attention Variants](https://github.com/pytorch-labs/attention-gym)
7. [Torchtune PR: Integrate Flex Attention](https://github.com/pytorch/torchtune/pull/1193)

---
## JAX - Why is Everyone So Excited About This Framework

## Introduction
The field of AI is going through an exponential surge, with new findings springing at an unprecedented rate. Accounting for Moore's law for data, a need for a highly performant framework to do ML is the an absolute necessity, as ultimately, unlocking the machine FLOPS is probably the main goal of any framework. There have been a lot of frameworks such as Tensorflow, PyTorch, and recently JAX that have tried to unlock these machine FLOPS, and for the purpose of this blog, we'll focus on JAX. There are a lot of this that make JAX unique, so let us jump right into it.

## What is JAX?
JAX has been gaining a lot of traction in recent times, and for the right reasons. JAX allows researchers to write Python programs that are automatically scaled to leverage accelerators and supercomputers(without any additional effort). JAX was developed by Deepmind to meet a simple goal which is to **balance rapid prototyping, quick iteration with the ability to deploy experiments at scale**. For those aware with NumPy, think of JAX as just NumPy with Autodiff and nice distributed support. Keep these point in the back of your mind, and let us try to understand why these qualities in JAX are so important and have many people(and frontier AI labs) pivoting to it. 

## Core Features - JAX
Don't take just my word for it, Francois Chollet(Keras founder) tweeted recently that almost all players in Generative AI are pivoting to JAX for the because it is fast, scales really well and there is TPU support too. A one line explanation of JAX would go something like this - *JAX is basically NumPy on steroids, made for researchers to write efficient, performant and scalable workloads.*  JAX has a lot of unique propositions for a performant library, so let us look into what makes JAX special:

### Automatic Differentiation
Autodiff keeps track of the grads and stuff, pretty important for ML workflows. We'll cover Autodiff in a lot of detail in the coming up sections.

### Just-In-Time Compilation
JAX uses a JIT compiler for speeding up entire blocks of code by exploiting any parallelism between them. Initially, we compile the function on its first use and later re-using the optimized version later, allowing efficient computations and lookup .

### VMap(Auto Vectorization)
VMap(advanced vectorization) allows us to apply some function on one or more axes of a tensor. VMap vectorizes a function by adding a batch dimension to every primitive operation in the function.  

### PMap(SPMD programming)
JAX has built in support for Single Program Multiple Data Programming, allowing the same function to be run in parallel on it's own XLA device.

In the introductory JAX blog post, it was mentioned that "JAX has una anima di pura programmazione funzionale”(has a soul of pure functional programming), so let us now try to understand why!

## Understanding JAX on a deeper level
There is difference in understanding high level overview and low level understanding of the system, and in order to fully nail the concepts, we are actually going to implement the core JAX structure from scratch, completely in Python. A high level overview of what we are going to cover is given in the diagram below. Originally, this blog is inspired by the [Autodiadax](https://jax.readthedocs.io/en/latest/autodidax.html) - which felt a little difficult to grasp, so I wrote it in simpler terms. Gear up, cause there is going to be a lot of code, and a lot more explanation. This is a work in process, so a lot can be added in the future version of this blog too.  

![Pasted image 20241017131715](https://github.com/user-attachments/assets/ae5b72c1-fd0b-4a75-bb55-dd5b5b9f5d33)

For now, we are dividing this blog into 4 parts(all of which are going to covered in great detail), which are :
1. **Part 1:** Transformations and Interpretation.
2. **Part 2:** JaxPrs(Jax Expressions)
3. **Part 3:** JIT(Just In Time Compilation)
4. **Part 4:** VJP(Vectorized Jacobian Product) 

We'll implement everything from grounds up, in pure python(apart from the standard library, we are just going to use the XLA package to transfer computational workloads). Let us start tinkering.
### Part 1 - Transformations and Interpretation

Let us start from the atomic unit of JAX - functions and operators. Traditionally, we apply these operations to numerical inputs, which gives us numerical answers. However, in JAX we want to override this behavior of operators and functions(which we treat as atomic units of processing - called primitives, rather than compositions) to be converted into **JaxPrs**(going to be covered in the next part, in great detail, but basically these are **Jax** Ex**pr**essions, which is intermediate representation of program, and what JAX uses instead of pure Python code). Converting functions and operators into JaxPrs allows JAX to represent the function into a small, well-behaved intermediate form that is then interpreted with transformation specific interpretation rules. Transformations are basically high order functions transforming Jaxprs. Not all Python programs can be converted to Jaxprs, but for many scientific computing and ML workflows, we can do it. The examples of Transformations include:

- `jax.grad():`A function to evaluate the gradient on the input function.
- `jax.vmap():`A function to implement automatic vectorization.
- `jax.pmap():`A function to implement data parallelism across processing units.

Let us try to define some primitives so we can understand their application:

```python

import numpy as np

from typing import NamedTuple, Any

# An object with a name, to which we attach interpretation rules.
class Primitive(NamedTuple):
    name: str

# Define the primitives - we'll start with the basic ones.
add_p = Primitive("add")
mul_p = Primitive("mul")
neg_p = Primitive("neg")
sin_p = Primitive("sin")
cos_p = Primitive("cos")
reduce_sum_p = Primitive("reduce_sum")
greater_p = Primitive("greater")
less_p = Primitive("less")
transpose_p = Primitive("transpose")
broadcast_p = Primitive("broadcast")

#  Bind is the interception point.
def bind1(prim, *args, **kwargs):
    out, = bind(prim, *args, **kwargs)
    return out

# Values as positional args, Metadata as kwargs.
def add(x,y): return bind1(add_p, x, y)
def mul(x,y): return bind1(mul_p, x, y)
def neg(x): return bind1(neg_p, x)
def sin(x) : return bind1(sin_p, x)
def cos(x) : return bind1(cos_p, x)
def greater(x,y): return bind1(greater_p, x, y)
def less(x,y): return bind1(less_p, x, y)
def transpose(x,perm) : return bind1(transpose_p, x, perm=perm)
def broadcast(x,shape,axes) : return bind1(broadcast_p, x, shape=shape, axes=axes)
def reduce_sum(x,axis) : return bind1(reduce_sum_p, x, axis=axis)

```

We'll attach our interpolation rules to the `Primitive` object - one for each transformation. The interception point is `bind`, which will figure out which transformations to apply(based on certain rules which we are going to cover later)

All the pure python function arguments are wrapped around `Tracer` objects - which records all the operations performed on it, creating a Jaxpr. The tracer object contains information about the shape, dtype of the initial arguments(not their value), which allows JAX to use the cached compiled program directly. Any change in shape/dtype triggers tracing, but not the value. This is the reason why only "functionally pure" functions(functions without side effects and which do not rely on values outside their arguments) should be used with JAX.

In the below code, `MainTrace` is basically a interpreter, and we are representing the active interpreters as a stack. When we are about to apply any transformation, we'll push another interpreter into the stack using the `new_main`. At the bottom of the stack, there is a evaluation interpreter(or `EvalTrace` - which we are going to see later in this section)

```python 

from contextlib import contextmanager
from typing import Any

class MainTrace(NamedTuple):
    level: int
    trace_type: type['Trace']
    global_data: Any | None

trace_stack: list[MainTrace] = []
dynamic_trace: MainTrace | None = None  # Later  

@contextmanager
def new_main(trace_type: type['Trace'], global_data=None):

    level = len(trace_stack)
    main = MainTrace(level, trace_type, global_data)
    trace_stack.append(main)

    try:
        yield main
    finally:
        trace_stack.pop()
```

We'll implement the `Trace` and `Tracer` base classes. A `Tracer` is basically an object that flows through the Python program that we are transforming. It represents a boxed up value with data to be used by the interpreter(or `MainTrace` in this case). `Trace` on the other hand, boxes up `Tracer` objects and also handles primitive applications.

```python

class Trace:
    main: MainTrace

    def __init__(self, main) -> None:
        self.main = main

    def pure(self, val):
        raise NotImplementedError()

    def lift(self, val):
        raise NotImplementedError()

    def process_primitive(self, primitive, tracers, params):
        raise NotImplementedError()

# One tracer per transformation
class Tracer:
	# stores Trace.
    _trace: Trace
    __array_priority__ = 69

    @property
    def aval(self):
	    # Tracer carries an abstract value. One abstract value per base type.
        raise NotImplementedError()

    def full_lower(self):
        return self

    def __neg__(self): return self.aval._neg(self)

    def __add__(self, other): return self.aval._add(self, other)

    def __radd__(self, other): return self.aval._radd(self, other)

    def __mul__(self, other): return self.aval._mul(self, other)

    def __rmul__(self, other): return self.aval._rmul(self, other)

    def __gt__(self, other): return self.aval._gt(self, other)

    def __lt__(self, other): return self.aval._lt(self, other)

    def __bool__(self): return self.aval._bool(self)

    def __nonzero__(self): return self.aval._nonzero(self)

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self.aval, name)
        except AttributeError:
            raise AttributeError(f"No attribute exists : {name}")

def swap(f): return lambda x, y : f(y, x)
```

For our use case, we are going to focus on abstract values that wrap arrays divided into two classes based on different levels of abstraction. These are:

- `ShapedArray`: This class represents the set of all possible arrays with a given shape and datatype. 
- `ConcreteArray`: This class represents a singleton set consisting of a single array value.

```python

class ShapedArray:

    """ Set of all possible arrays with a given shape and dtype. """

    array_abstraction_level = 1
    shape: tuple[int, ...]
    dtype: np.dtype

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    @property
    def ndim(self):
        return len(self.shape)

    _neg = staticmethod(neg)
    _add = staticmethod(add)
    _radd = staticmethod(swap(add))
    _mul = staticmethod(mul)
    _rmul = staticmethod(swap(mul))
    _gt = staticmethod(greater)
    _lt = staticmethod(less)

    @staticmethod
    def _bool(tracer):
        raise Exception("Can't convert to bool")

    @staticmethod
    def _nonzero(tracer):
        raise Exception("Can't convert to bool")

    def str_short(self):
        return f'{self.dtype.name}[{",".join(str(d) for d in self.shape)}]'

    def __eq__(self, other):
        return (type(self) == type(other) and self.shape == other.shape and self.dtype == other.dtype)

class ConcreteArray(ShapedArray):
    """ Singleton set consisting of a single array value. """

    array_abstraction_level = 2
    val: np.ndarray

    def __init__(self, val):
        self.val = val
        self.shape = val.shape
        self.dtype = val.dtype

    @staticmethod
    def _bool(tracer):
        return bool(tracer.aval.val)

    @staticmethod
    def _nonzero(tracer):
        return bool(tracer.aval.val)

def get_aval(x):
    if isinstance(x, Tracer):
        return x.aval

    elif type(x) in jax_types:
        return ConcreteArray(np.asarray(x))

    else:
        raise TypeError(x)

jax_types = {bool, int, float, np.bool_, np.int32, np.int64, np.float32, np.float64, np.ndarray}
```

After setting up the interpreter stack, the base classes for `Trace`/`Tracer`, and base classes for abstract values, we should come back and implement the `bind` function -  which, if you remember is our interception point to figure out which transformation rules to apply.

The steps performed by the `bind` function are:

1. **Find Top Trace** ; figure out which interpreter should handle the primitive application.
2. **Call the top trace's process primitive** so the trace can apply interpretation rule
3. **Full raise** ensures that inputs are boxed in the tracer instances.
4. **Full lower** for optional optimization, so that we unbox values out of Tracer as much as possible.

The main action is that we figure out which interpreter should handle this primitive application. We then call the top trace's `process_primitive` so that the trace can apply it's interpretation rules. The calls to `full_raise` just ensure that inputs are boxed in the top trace's `Tracer` instances(`full_lower` is for optional optimization so that we unbox values out of `Tracer` as much as possible). 

```python
def bind(prim, *args, **params):
    top_trace = find_top_trace(args)
    tracers = [full_raise(top_trace, arg) for arg in args]
    outs = top_trace.process_primitive(prim, tracers, params)
    return [full_lower(out) for out in outs]

import operator as op

# Returns the highest level interpreter associated with Tracer, otherwise returns the EvalTrace.
def find_top_trace(xs) -> Trace:
  top_main = max((x._trace.main for x in xs if isinstance(x, Tracer)),default=trace_stack[0], key=op.attrgetter('level'))

  if dynamic_trace and dynamic_trace.level > top_main.level:
    top_main = dynamic_trace

  return top_main.trace_type(top_main)

  
def full_lower(val):
    if isinstance(val, Tracer):
        return val.full_lower()
    return val
  
# Boxing up values into Tracer's for a particular Trace.
# Trace.pure is called for non-tracer constants, and Trace.lift called for values that are already Tracer's from a lower level interpreter.
def full_raise(trace, val):
    if not isinstance(val, Tracer):
        assert type(val) in jax_types
        return trace.pure(val)

    level = trace.main.level

    if val._trace.main is trace.main:
        return val

    elif val._trace.min.level < level :
        return trace.lift(val)  

    elif val._trace.min.level > level :
        raise Exception("Cannot lift level")

    else:
        raise Exception("Different traces at same level.")
```

### Evaluation Interpreter

As explained earlier, the Evaluation Interpreter will sit at the bottom of the interpreter stack. Since this is the easiest to implement, we'll start with this.

`EvalTrace` extends from the `Trace` base class, and implements the `process_primitive` function, which basically applies the implementation rule of the primitive.

As mentioned, the trace_stack(which is basically a list) has the `EvalTrace` at the bottom. After that, we implement all the primitive functions(remember, we are just doing the vector operation, only we have a interception point which will figure out which transformations to apply)

```python
class EvalTrace(Trace):
    pure = lift = lambda self, x: x

    def process_primitive(self, primitive, tracers, params):
        return impl_rules[primitive](*tracers, **params)

trace_stack.append(MainTrace(0, EvalTrace, None))

impl_rules = {}

impl_rules[add_p] = lambda x, y: [np.add(x, y)]
impl_rules[mul_p] = lambda x, y: [np.multiply(x, y)]
impl_rules[neg_p] = lambda x: [np.negative(x)]
impl_rules[sin_p] = lambda x: [np.sin(x)]
impl_rules[cos_p] = lambda x: [np.cos(x)]
impl_rules[reduce_sum_p] = lambda x, *, axis: [np.sum(x, axis)]
impl_rules[greater_p] = lambda x, y: [np.greater(x,y)]
impl_rules[less_p] = lambda x, y: [np.less(x,y)]
impl_rules[transpose_p] = lambda x, *, perm: [np.transpose(x, perm)]

def broadcast_impl(x, *, shape, axes):
    for axis in sorted(axes):
        x = np.expand_dims(x, axis)
    return [np.broadcast_to(x, shape)] # read broadcasting rules!

impl_rules[broadcast_p] = broadcast_impl
```

We mentioned earlier that JAX is well suited for ML, and that means JAX has a good(and general) support for automatic differentiation(AD). AD can obtain gradients of numerical programs very efficiently, which we generally use to calculate loss and backpropagate it to minimize the calculated loss.  Automatic differentiation basically applies a set of elementary operations on a function, and automatically computes the gradients by application of the chain rule. It makes a set of equations that include intermediate variables to create a computational graph, and then computes the gradients.   

To accommodate generality in it's AD system, JAX implements both forward and reverse mode automatic differentiation. The ever-so used `grad` function is built on reverse mode AD, while for forward mode, JAX uses JVP(Jacobian Vector Product). JVPs are evaluated on the fly, so they are memory efficient, but in ML, we don't see forward mode differentiation.[^1]  

Let us implement a JVP based Tracer that calculates both primals(basically the value of the function at any point) and the tangent(the forward mode gradient value associated with the function at that particular point). Before that, let us define some helper function we are going to use. 

```python
import builtins

# Get a vector full of zeros like the abstract value of array
def zeros_like(val):
    aval = get_aval(val)
    return np.zeros(aval.shape, aval.dtype)

# Given a pair of values, unpack them into two lists.
def unzip2(pairs):
    lst1, lst2 = [], []
    for x1, x2 in pairs:
        lst1.append(x1)
        lst2.append(x2)
    return lst1, lst2

# Map values and wrap them in a list.
def map(f, *xs):
    return list(builtins.map(f, *xs))

# Returns a list of pairs of values.
def zip(*args):
    fst, *rest = args = map(list, args)
    n = len(fst)
    for arg in rest:
        assert len(arg) == n
    return list(builtins.zip(*args))

```

For forward mode differentiation, the `JVPTracer` carries the boxed up primal-tangent pair of values, while the `JVPTrace` applies the JVP rules. For initialization, we want to "package" the `pure` and `lift` values with zero tangent. After doing that, let us add some JVP rules for the primitives[^2]. In the end, let us add a "Transformation API"(`jvp_v1`) which pushes another interpreter into the stack using the `new_main` and gives us the primal and the tangent associated with the primitive.

```python  
class JVPTracer(Tracer):
    def __init__(self, trace, primal, tangent):
        self._trace = trace
        self.primal = primal
        self.tangent = tangent

    @property
    def aval(self):
        return get_aval(self.primal)

class JVPTrace(Trace):
    pure = lift = lambda self, val: JVPTracer(self, val, zeros_like(val))

    def process_primitive(self, primitive, tracers, params):
        primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
        jvp_rule = jvp_rules[primitive]
        primal_outs, tangent_outs = jvp_rule(primals_in, tangents_in, **params)
        return [JVPTracer(self, x, t) for x, t in zip(primal_outs, tangent_outs)]

jvp_rules = {}

# Add some JVP rules
def add_jvp(primals, tangents):
    (x,y), (x_dot, y_dot) = primals, tangents
    return [x + y], [x_dot + y_dot]

jvp_rules[add_p] = add_jvp

def mul_jvp(primals, tangents):
    (x,y), (x_dot, y_dot) = primals, tangents
    return [x * y], [x_dot*y + x*y_dot]

jvp_rules[mul_p] = mul_jvp

def sin_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return [sin(x)], [cos(x) * x_dot]
  
jvp_rules[sin_p] = sin_jvp

def cos_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return [cos(x)], [-sin(x) * x_dot]

jvp_rules[cos_p] = cos_jvp

def neg_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return [neg(x)], [neg(x_dot)]

jvp_rules[neg_p] = neg_jvp

def reduce_sum_jvp(primals, tangents, *, axis):
    (x,), (x_dot, ) = primals, tangents
    return [reduce_sum(x, axis)], [reduce_sum(x_dot, axis)]

jvp_rules[reduce_sum_p] = reduce_sum_jvp

def greater_jvp(primals, tangents):
    (x, y), _ = primals, tangents
    out_primal = greater(x, y)
    return [out_primal], [zeros_like(out_primal)]

jvp_rules[greater_p] = greater_jvp

def less_jvp(primals, tangents):
    (x, y), _ = primals, tangents
    out_primal = less(x, y)
    return [out_primal], [zeros_like(out_primal)]

jvp_rules[less_p] = less_jvp

# Transformation API.
def jvp_v1(f, primals, tangents):
    with new_main(JVPTrace) as main:
        trace = JVPTrace(main)
        tracers_in = [JVPTracer(trace, x, t) for x, t in zip(primals, tangents)]
        out = f(*tracers_in)
        tracer_out = full_raise(trace, out)
        primal_out, tangent_out = tracer_out.primal, tracer_out.tangent

    return primal_out, tangent_out
```

With all these in place, we can now differentiate function in JAX. Here's an example code:

```python
#Example to demonstrate JVP_V1
x = 3.0
y, sin_deriv_at_3 = jvp_v1(sin, (x,), (1.0,))
print(sin_deriv_at_3)
print(cos(3.0))

```

There's a reason we named our Transformation API as `jvp_v1`. A limitation in this is that it accepts arrays as input and gives a single array as output. In this next iteration of `jvp`, we will deal with nested inputs and nested outputs. This could mean that at each layer of the stack, we might have to deal with nested inputs.  

In order to deal with this, we are going to wrap the user function so that it accepts arrays as input, and returns a flat list of arrays as output. Since we are accepting user functions that have arbitrary "containers" in the inputs and outputs, just flattening a list inside of a list isn't going to be very general, and we are actually going to need reference to a tree data structure. Let us try to understand why do we need a tree for this.

In our use case, we mentioned that the inputs and outputs containers can be of arbitrary depth. If we represent the topmost level as the parent node(where each node can have multiple children, and can represent both lists(internal nodes) and the value(leaf node)). In any scenario where the data is arranged in a hierarchal manner, a tree is the optimal data structure to use. A real life example where a hierarchal data is represented by a tree would be a folder/directory structure. In that case as well:

- Each node can have multiple children(either subfolders or files)
- Nodes can represent both folders and files.

```python
def jvp_flat(f, primals, tangents):
    with new_main(JVPTrace) as main:
        trace = JVPTrace(main)
        tracers_in = [JVPTracer(trace, x, t) for x, t in zip(primals, tangents)]
        outs = f(*tracers_in)
        tracers_out = [full_raise(trace, out) for out in outs]
        primals_out, tangents_out = unzip2((t.primal, t.tangent) for t in tracers_out)    
    return primals_out, tangents_out


def jvp(f, primals, tangents):
    primals_flat, in_tree = tree_flatten(primals)
    tangents_flat, in_tree2 = tree_flatten(tangents)
    if in_tree != in_tree2: raise TypeError
    f, out_tree = flatten_fun(f, in_tree)
    primals_out_flat, tangents_out_flat = jvp_flat(f, primals_flat, tangents_flat)
    primals_out = tree_unflatten(out_tree(), primals_out_flat)
    tangents_out = tree_unflatten(out_tree(), tangents_out_flat)

    return primals_out, tangents_out

  
```

Now that we have understood why we are going to need a need a tree flatten/unflatten function, let us implement those.

Notice in the `flatten_fun` function that the function might actually require the unflatten version of the arguments, we are going to provide it in the unflattened version only(this information isn't available until we run the function, that is also the reason why we return the reference to `flat_fun`) 

```python
# Notice that we need to provide the unflattened version of the args.
def flatten_fun(f, in_tree):
    store = Store()
    
    def flat_fun(*args_flat):
        pytree_args = tree_unflatten(in_tree, args_flat)
        out = f(*pytree_args)
        out_flat, out_tree = tree_flatten(out)
        store.set_value(out_tree)
        return out_flat

    return flat_fun, store

# Helper classes.

class Empty: pass
empty = Empty()
  
class Store:
    val = empty
    def set_value(self, val):
        assert self.val is empty
        self.val = val
    
    def __call__(self):
        return self.val

# PyTree handling of JVP implementation.  

from collections.abc import Hashable
import itertools as it
from collections.abc import Callable

# Represents the node of the data container.
class NodeType(NamedTuple):
    name:str
    to_iterable: Callable
    from_iterable: Callable

# Basically a typecheck function to register only lists, tuples and dicts as valid tree DS.   
def register_pytree_node(ty, to_iter, from_iter):
    node_types[ty] = NodeType(str(ty), to_iter, from_iter)

node_types = {}

# Only dict, tuple and list can represent PyTree. This also acts as a typecheck.
register_pytree_node(tuple, lambda t: (None, t), lambda _, xs: tuple(xs))

register_pytree_node(list, lambda l: (None, l), lambda _, xs: list(xs))

register_pytree_node(dict, lambda d: map(tuple, unzip2(sorted(d.items()))), lambda keys, vals: dict(zip(keys, vals)))


class PyTreeDef(NamedTuple):
    node_type: NodeType
    node_metadata:Hashable
    child_treedefs: tuple['PyTreeDef', ...]

class Leaf: pass
leaf = Leaf()

# Flatten the tree.  
def tree_flatten(x):
    children_iter, tree_def = _tree_flatten(x)
    return list(children_iter), tree_def

def _tree_flatten(x):
    node_type = node_types.get(type(x))
    if node_type:
        node_metadata, children = node_type.to_iterable(x)
        children_flat, child_trees = unzip2(map(_tree_flatten, children))
        flattened = it.chain.from_iterable(children_flat)
        return flattened, PyTreeDef(node_type, node_metadata, tuple(child_trees))

    else:
        return [x], leaf

# Unflatten the tree.  
def tree_unflatten(tree_def, xs):
    return _tree_unflatten(tree_def, iter(xs))

  
def _tree_unflatten(tree_def, xs):
    if tree_def is leaf:
        return next(xs)
    else:
        children = (_tree_unflatten(t, xs) for t in tree_def.child_treedefs)

        return tree_def.node_type.from_iterable(tree_def.node_metadata, children)

```
We have successfully implemented arbitrary depth input/output containers. These will be helpful with future transformations as well. Here's an example code to understand how it works:

```python
# Define some arbitrary Pythonic function
def f(x):
  y = 3.0 * sin(x) * cos(x) 
  z = x*x + y*y
  # We can handle arbitrary depth inputs.
  return {'Rick': z, 'Astley': [x, y]}

# Evaluate the functions at specific values using JVP.
x, xdot = 1.0, 1.5
y, ydot = jvp(f, (x,), (xdot,))

# Print the results
print(y)
print(ydot)

```

With the improved JVP implementation now in place, we can move to a another important bottleneck that JAX tries to resolve, especially when it comes to machine learning workflows - and that is vectorized batching. Traditionally, when batching data for ML workflows,(a reason to batch data is to parallelly process a chunk of data, and not process element by element) loop can be slow and can become a bottleneck for performance critical code. JAX `vmap` addresses these issues, and provides the following functionalities:

- Applying operations to the entire array at once, not individual elements.
- Processing multiple inputs simultaneously.
- Seamless integration with other JAX functions, including `jit` and `grad`.

Let us implement vectorized batching with `vmap`, but before that let us implement some helper functions:

- `mapped_aval`: It produces mapped abstract values from the unmapped ones, by removing an axis from it.
- `move_batch_axis`: It is used to move the batch dimensions around(by basically moving the axis).
 
```python
# Produce mapped values from the unmapped ones.
def mapped_aval(batch_dim, aval):
    shape = list(aval.shape)
    del shape[batch_dim]
    return ShapedArray(tuple(shape), aval.dtype)

# Move the batch axis by mving the axis.
def move_batch_axis(axis_size, src, dst, x):
    if src is not_mapped:
        target_shape = list(np.shape(x))
        target_shape.insert(dst, axis_size)
        return broadcast(x, target_shape, [dst])

    elif src == dst:
        return x
        
    else:
        return move_axis(x, src, dst)
  
# Move the axis from src to dst.
def move_axis(x, src, dst):
    perm = [i for i in range(np.dim(x)) if i != src]
    perm.insert(dst, src)
    return transpose(x, perm)
```

With the helper function implementation in place, let us shift our focus to implementing a `BatchTracer` for vectorized batching. The tracer carries a batched value and an optional integer indicating which axis is the batch axis. Similar to other trace classes, `BatchTrace`  implements the `pure` and `lift` methods containing the boxed up values in a `BatchTracer` instance. We use the `MainTrace`'s global data field to store the batch axis size. 

```python
from typing import Union

# Wrapper class.
class NotMapped: pass
not_mapped = NotMapped()

# Wrapper class. Apart from BatchAxis, all are mapped. 
BatchAxis = Union[NotMapped, int] 

# Tracer to accomadate batching of data.
class BatchTracer(Tracer):
    def __init__(self, trace, val, batch_dim):
        self._trace = trace
        self.val = val
        self.batch_dim = batch_dim

    @property
    def aval(self):
        if self.batch_dim is not_mapped:
            return get_aval(self.val)
        return mapped_aval(self.batch_dim, get_aval(self.val))

    def full_lower(self):
        if self.batch_dim is not_mapped:
            return full_lower(self.val)
        return self

# Tracer
class BatchTrace(Trace):
    pure = lift = lambda self, val: BatchTracer(self, val, not_mapped)

    def process_primitive(self, primitive, tracers, params):
        vals_in, bdims_in = unzip2((t.val, t.batch_dim) for t in tracers)
        vmap_rule = vmap_rules[primitive]
        val_outs, bdim_outs = vmap_rule(self.axis_size, vals_in, bdims_in, **params)

        return [BatchTracer(self, x, bd) for x, bd in zip(val_outs, bdim_outs)]

  

    @property
    def axis_size(self):
        return self.main.global_data

vmap_rules = {}
```

The next step is to implement the batched interpreter rules for each primitive. The implementation is divided into three classes of implementations, one for binary operators(addition and multiplication primitives), one for unary operator(such as sin, cos and negation primitives), and a separate one for reduce sum. With all these in place, we add a transformation API to start the trace(see the `vmap_flat` and `vmap` implementation)  

```python
from functools import partial

# primitive rules - addition, multiplication 
def binop_batching_rule(op, axis_size, vals_in, dims_in):
    (x, y), (x_bdim, y_bdim) = vals_in, dims_in
    if x_bdim != y_bdim:
        if x_bdim is not_mapped:
            x = move_batch_axis(axis_size, x_bdim, y_bdim, x)
            x_bdim = y_bdim
        else:
            y = move_batch_axis(axis_size, y_bdim, x_bdim, y)

    return [op(x,y)], [x_bdim]

vmap_rules[add_p] = partial(binop_batching_rule, add)
vmap_rules[mul_p] = partial(binop_batching_rule, mul)

# primitive rules - sin, cos, negation
def vectorized_unop_batching_rule(op, axis_size, vals_in, dims_in):
    (x,), (x_bdim,) = vals_in, dims_in
    return [op(x)], [x_bdim]

vmap_rules[sin_p] = partial(vectorized_unop_batching_rule, sin)
vmap_rules[cos_p] = partial(vectorized_unop_batching_rule, cos)
vmap_rules[neg_p] = partial(vectorized_unop_batching_rule, neg)

# primitive rules - reduce sum
def reduce_sum_batching_rule(axis_size, vals_in, dims_in, *, axis):
    (x,), (x_bdim,) = vals_in, dims_in
    new_axis = tuple(ax + (x_bdim <= ax) for ax in axis)
    out_bdim = x_bdim - sum(ax < x_bdim for ax in axis)
    return [reduce_sum(x, new_axis)], [out_bdim]

vmap_rules[reduce_sum_p] = reduce_sum_batching_rule

# Transformation API.
def vmap_flat(f, in_axes, *args):
    axis_size, = {x.shape[ax] for x, ax in zip(args, in_axes) if ax is not not_mapped}

    with new_main(BatchTrace, axis_size) as main:
        trace = BatchTrace(main)
        tracers_in = [BatchTracer(trace, x, ax) if ax is not None else x for x, ax in zip(args, in_axes)]

        outs = f(*tracers_in)

        tracers_out = [full_raise(trace,out) for out in outs]
        vals_out, bdims_out = unzip2((t.val, t.batch_dim) for t in tracers_out)

    outs_transposed = [move_batch_axis(axis_size, bdim, 0, val_out) for val_out, bdim in zip(vals_out, bdims_out)]

    return outs_transposed

def vmap(f, in_axes):
    def batched_f(*args):
        args_flat, in_tree = tree_flatten(args)
        in_axes_flat, in_tree2 = tree_flatten(in_axes)
        if in_tree != in_tree2 : raise TypeError

        f_flat, out_tree = flatten_fun(f, in_tree)
        outs_flat = vmap_flat(f_flat, in_axes_flat, *args_flat)
        return tree_unflatten(out_tree(), outs_flat)    

    return batched_f
```

Let's see our implementation in action!
```python
# Pythonic function
def add_to_a_scalar(scalar):
  assert np.ndim(scalar) == 0
  return 69 + scalar

# Vectorized operation using VMAP
vector_in = np.arange(420.0)
vector_out = vmap(add_to_a_scalar, (0,))(vector_in)

# Output
print(vector_in)
print(vector_out)
```

With the implementations of VMap and JVP(basically, Autodiff) in place, the next transformations in place are JIT and VJP(for reverse mode autodiff). The implemented transformations only needed each Tracer to carry an extra bit of context, but for JIT and VJP, we need much richer context(the next few sections are going to explain how), and for that, we need to represent Pythonic programs as JaxPrs.  
### Part 2 - JaxPrs

JaxPrs are JAX's internal representation of programs. For JIT implementation, we need JaxPrs because JIT need to stage out any computation out of Python(mostly to XLA), and therefore to represent the data using JaxPrs helps in tracing the python function back up. In the case of VJP, JaxPrs provide a way to represent the computation for the backward pass of the reverse mode autodiff. To represent JaxPrs as Python data structure, we re-use the `ShapedArray` class defined before(for types) and can represent the term syntax with a few Python structs. 

![Pasted image 20241017142150](https://github.com/user-attachments/assets/4cdbe1a3-0bf4-44fa-ad2b-4ca0c11bf1a8)

```python
# Class to hold abstract value as ShapedArray.
class Var:
    aval: ShapedArray
    def __init__(self, aval): self.aval = aval

# Class for holding value(both normal and abstract)
class Lit:
    val: Any
    aval: ShapedArray

    def __init__(self, val):
        self.aval = aval = raise_to_shaped(get_aval(val))
        self.val = np.array(val, aval.dtype)

# Atom is the building block for JaxPrs.
Atom = Union[Var, Lit]

# A JaxprEqn is basically a class holding the primitive, and the inputs and outputs associated with it. 
class JaxprEqn(NamedTuple):
    primitive: Primitive
    inputs: list[Atom]
    params: dict[str, Any]
    out_binders: list[Var]

# A JaxPr can hold multiple JaxprEqn 
class Jaxpr(NamedTuple):
    in_binders: list[Var]
    eqns: list[JaxprEqn]
    outs: list[Atom]

  
    def __hash__(self): return id(self)
    __eq__ = op.is_
  

def raise_to_shaped(aval):
    return ShapedArray(aval.shape, aval.dtype)
```

Type checking is very strict in JAX, which is crucial for speeding up computational workflows. Strict type checking allows JAX to perform type specialization, and optimize code for specific data types. For JaxPrs, type checking involves checking whether there are no unbound variables, and that variables are bound only once, and the equation of the type of primitive matches the type of output binders.  JaxPrs are platform-agnostic, so type checking ensures consistency across platforms. 

```python
class JaxprsType(NamedTuple):

    in_types: list[ShapedArray]
    out_types: list[ShapedArray]

    def __repr__(self):
        in_types = ", ".join(aval.str_short() for aval in self.in_types)
        out_types = ", ".join(aval.str_short() for aval in self.out_types)

        return f'({in_types}) -> ({out_types})'

# Typechcek for reasons mentioned above.
def typecheck_jaxpr(jaxpr):
    env: set[Var] = set()

    for v in jaxpr.in_binders:
        if v in env: raise TypeError
        env.add(v)

    for eqn in jaxpr.eqns:

        in_types = [typecheck_atom(env, x) for x in eqn.inputs]
        out_types = abstract_eval_rules[eqn.primitive](*in_types, **eqn.params)

        for out_binder, out_type in zip(eqn.out_binders, out_types):
            if not out_type == out_binder.aval: raise TypeError

        for out_binder in eqn.out_binders:
            if out_binder in env: raise TypeError
            env.add(out_binder)

        in_types = [v.aval for v in jaxpr.in_binders]
        out_types = [typecheck_atom(env, x) for x in jaxpr.outs]
        return JaxprsType(in_types, out_types)

def typecheck_atom(env, x):
    if isinstance(x, Var):
        if x not in env: raise TypeError("Unbound Variable")
        return x.aval

    elif isinstance(x, Lit):
        return raise_to_shaped(get_aval(x.val))
    
    else:
        assert False

# This is a simple JaxPr interpreter, with type checking.
def eval_jaxpr(jaxpr, args):
    env = {}

    def read(x):
        return env[x] if type(x) is Var else x.val

    def write(v, val):
        assert v not in env
        env[v] = val
        
    map(write, jaxpr.in_binders, args)

    for eqn in jaxpr.eqns:
        in_vals = map(read, eqn.inputs)
        outs = bind(eqn.primitive, *in_vals, **eqn.params) # Using bind makes this interpreter traceable too.
        map(write, eqn.out_binders, outs)
        
    return map(read, jaxpr.outs)

def jaxpr_as_fun(jaxpr):
    return lambda *args: eval_jaxpr(jaxpr, args)

```

Similarly to what we did with other interpreters, we'll now enable tracing for JaxPrs. There are two ways in which we can do this, and we'll start with what `jit` uses. But first, let us define some helper functions and then build up the `JaxPrTrace` and `JaxPrTracer`.  

The `JaxPrTrace` class implements a `new_arg` function to return a `Tracer` instance after adding it to the builder. The `get_or_make_tracer` method add a tracer to the builder, or if it doesn't exists(checked using the `id` of the Tracer instance). The `pure` and `lift` variables of the Tracer return the reference to this function. The `process_primitive` function is similar to the ones described before, with the only difference being the use of JaxPrs.  

```python
# Helper functions: Jaxprs with Tracing.

# Split a list and outputs the partitions.
def split_list(lst, n):
    assert 0 <= n <= len(lst)
    return lst[:n], lst[n:]

# Partition a list and return the components. 
def partition_list(bs, l):
    assert len(bs) == len(l)
    lists = lst1, lst2 = [], []
    for b, x in zip(bs, l):
        lists[b].append(x)
    return lst1, lst2

# Tracer, as mentioned contains the boxed-up values.
class JaxprTracer(Tracer):
    __slots__ = ['aval']
    aval: ShapedArray

    def __init__(self, trace, aval):
        self._trace = trace
        self.aval = aval
  
# Main JaxPrTrace class.
class JaxprTrace(Trace):

    def new_arg(self, aval):
        aval = raise_to_shaped(aval)
        tracer = self.builder.new_tracer(self, aval)
        self.builder.tracer_to_var[id(tracer)] = Var(aval)
        return tracer

    def get_or_make_const_tracer(self, val):
        tracer = self.builder.const_tracers.get(id(val))
        if tracer is None:
            tracer = self.builder.new_tracer(self, raise_to_shaped(get_aval(val)))
            self.builder.add_const(tracer, val)
        return tracer
    pure = lift = get_or_make_const_tracer

    def process_primitive(self, primitive, tracers, params):
        avals_in = [t.aval for t in tracers]
        avals_out = abstract_eval_rules[primitive](*avals_in, **params)
        out_tracers = [self.builder.new_tracer(self, a) for a  in avals_out]
        inputs = [self.builder.getvar(t) for t in tracers]
        outvars = [self.builder.add_var(t) for t in out_tracers]

        self.builder.add_eqn(JaxprEqn(primitive, inputs, params, outvars))
        return out_tracers

    @property
    def builder(self):
        return self.main.global_data

abstract_eval_rules = {}
```

`JaxPrBuilder` is the container we use to keeps track of the variables, constants and equations, - as the interpreter global data and will be referenced later as we build up the JaxPrs. The implementation is followed.

```python
# Container class to hold up data.
class JaxprBuilder:

    eqns: list[JaxprEqn]
    tracer_to_var: dict[int, Var]
    const_tracers: dict[int, JaxprTracer]
    constvals: dict[Var, Any]
    tracers: list[JaxprTracer]

    def __init__(self):
        self.eqns = []
        self.tracer_to_var = {}
        self.const_tracers = {}
        self.constvals = {}
        self.tracers = []
        
	# Add a new tracer with a given aval
    def new_tracer(self, trace, aval):
        tracer = JaxprTracer(trace, aval)
        self.tracers.append(tracer)
        return tracer
	## Other getter and setters method for the class, self explanatory.

    def add_eqn(self, eqn):
        self.eqns.append(eqn)

    def add_var(self, tracer):
        assert id(tracer) not in self.tracer_to_var
        var = self.tracer_to_var[id(tracer)] = Var(tracer.aval)
        return var

    def getvar(self, tracer):
        var = self.tracer_to_var.get(id(tracer))
        assert var is not None
        return var

    def add_const(self, tracer, val):
        var = self.add_var(tracer)
        self.const_tracers[id(val)] = tracer
        self.constvals[var] = val
        return var

    def build(self, in_tracers, out_tracers):
        constvars, constvals = unzip2(self.constvals.items())
        t2v = lambda t: self.tracer_to_var[id(t)]
        in_binders = constvars + [t2v(t) for t in in_tracers]
        out_vars = [t2v(t) for t in out_tracers]
        jaxpr = Jaxpr(in_binders, self.eqns, out_vars)
        typecheck_jaxpr(jaxpr) # important step!
        jaxpr, constvals = _inline_literals(jaxpr, constvals)
        
        return jaxpr, constvals

def _inline_literals(jaxpr, consts):
    const_binders, other_binders = split_list(jaxpr.in_binders, len(consts))
    scalars = [type(x) in jax_types and not get_aval(x).shape for x in consts]

    new_const_binders, lit_binders = partition_list(scalars, const_binders)
    new_consts, lit_vals = partition_list(scalars, consts)
    literals = dict(zip(lit_binders, map(Lit, lit_vals)))
    new_eqns = [JaxprEqn(eqn.primitive, [literals.get(x, x) for x in eqn.inputs], eqn.params, eqn.out_binders) for eqn in jaxpr.eqns]
    new_outs = [literals.get(x, x) for x in jaxpr.outs]
    new_jaxpr = Jaxpr(new_const_binders + other_binders, new_eqns, new_outs)
    typecheck_jaxpr(new_jaxpr)
    
    return new_jaxpr, new_consts
```

With the Tracer, Trace implementations in place, let us implement the `eval_rules` as we did for other cases as well. Most of these are very general, with the intention that these abstraction will be reused for other JaxPr-producing trace methods. 

```python
# Binop for Add, Multiply.
def binop_abstract_eval(x, y):
    if not isinstance(x, ShapedArray) or not isinstance(y, ShapedArray):
        raise TypeError
    if raise_to_shaped(x) != raise_to_shaped(y): raise TypeError
    return [ShapedArray(x.shape, x.dtype)]

abstract_eval_rules[add_p] = binop_abstract_eval
abstract_eval_rules[mul_p] = binop_abstract_eval

# Compare for less than, greater than.
def compare_abstract_eval(x, y):
    if not isinstance(x, ShapedArray) or not isinstance(y, ShapedArray):
        raise TypeError
    if x.shape != y.shape: raise TypeError
    return [ShapedArray(x.shape, np.dtype('bool'))]

abstract_eval_rules[greater_p] = compare_abstract_eval
abstract_eval_rules[less_p] = compare_abstract_eval

# Vectorized Op for Sin, Cosine and Negation.
def vectorized_unop_abstract_eval(x):
    return [ShapedArray(x.shape, x.dtype)]

abstract_eval_rules[sin_p] = vectorized_unop_abstract_eval
abstract_eval_rules[cos_p] = vectorized_unop_abstract_eval
abstract_eval_rules[neg_p] = vectorized_unop_abstract_eval

# Different eval for reduce_sum.
def reduce_sum_abstract_eval(x, *, axis):
    axis_ = set(axis)
    new_shape = [d for i,d in enumerate(x.shape) if i not in axis_]
    return [ShapedArray(tuple(new_shape), x.dtype)]

abstract_eval_rules[reduce_sum_p] = reduce_sum_abstract_eval

# One for broadcast as well.
def broadcast_abstract_eval(x, *, shape, axes):
    return [ShapedArray(tuple(shape), x.dtype)]

abstract_eval_rules[broadcast_p] = broadcast_abstract_eval
```

With all the things in place, we can kick off our Transformation API. There is however a really fundamental flaw in `make_jaxpr_v1`, which maybe deserves a blog post on it's own[^3] . In short, the input which were not boxed up in `JaxprTracer` instances ended up wasting memory, time dispatching and maybe even fragmenting memory.

This "omnistagging" issue ensures that JaxprTrace started by `make_jaxpr` is always applied. Conceptually, the dynamic trace is identical to stashing the current interpreter stack and starting a new one with the JaxprTrace at the bottom.  The new transformation API(`make_jaxpr`) uses the `dynamic_trace` global(see Part 1) for this reason.

```python
from functools import lru_cache

def make_jaxpr_v1(f, *avals_in):
    avals_in, in_tree = tree_flatten(avals_in)
    f, out_tree = flatten_fun(f, in_tree)
    builder = JaxprBuilder()

    with new_main(JaxprTrace, builder) as main:
        trace = JaxprTrace(main)
        tracers_in = [trace.new_arg(aval) for aval in avals_in]
        outs = f(*tracers_in)
        tracers_out = [full_raise(trace, out) for out in outs]
        jaxpr, consts = builder.build(tracers_in, tracers_out)
    return jaxpr, consts, out_tree()
  

# There is a limitations tho. This version can't stage out all the primitve opeations performed by the Python Callable.

@contextmanager
def new_dynamic(main):
    global dynamic_trace
    prev_dynamic_trace, dynamic_trace = dynamic_trace, main
    try:
        yield
    finally:
        dynamic_trace = prev_dynamic_trace


def make_jaxpr(f, *avals_in):
    avals_in, in_tree = tree_flatten(avals_in)
    f, out_tree = flatten_fun(f, in_tree)
    builder = JaxprBuilder()

    with new_main(JaxprTrace, builder) as main:
        with new_dynamic(main):
            trace = JaxprTrace(main)
            tracers_in = [trace.new_arg(aval) for aval in avals_in]
            outs = f(*tracers_in)
            tracers_out = [full_raise(trace, out) for  out in outs]
            jaxpr, consts = builder.build(tracers_in, tracers_out)
        
        return jaxpr, consts, out_tree()
```
### Part 3 - JIT

After converting the Pythonic functions into JaxPrs, the next step is taken up by JIT compiler. JIT, or Just-In-Time compiler analyzes the JaxPrs and identifies the specific operations needed, then the optimized machine code is generated for those operations. JIT only compiles the necessary code(and caches the compiled machine code for future use) - giving significant speedups for computationally expensive workflows.

Similar to JaxPrs and JVP, even JIT has a transformation like API transforming a Python function, but conceptually, JIT under the hood is a high-order primitive(basically, a high order primitive is parameterized by a function) rather than a transformation. Similar to a primitive, JIT take JaxPrs as input, returns a "transformed" function(in this case, an optimized version of that function) as output, and operates on functional level, transforming it's execution.

In order to handle high order primitives, JIT uses a staged processing approach, where we can just use `make_jaxpr` in the primitive wrapper to form JaxPrs up-front and skip the python function entirely[^4] - which is what we need to stage these computation to XLA(Accelerated Linear Algebra) for ML workflows. 

Since JIT is a high-level primitive, we need to give it transformation rules. When we evaluate any `xla_primitive` application, we stage out the computation to XLA by translating the JaxPrs into an XLA HLO program ; including transferring the argument values to the XLA device, executing the XLA program(and cache the results as per the shape and dtype signature), and transferring back the results.

![Pasted image 20241017142406](https://github.com/user-attachments/assets/fcc923ff-9234-4c06-af3d-e673f4351dbb)


```python
# This is the staged JIT wrapper, with computation done by XLA.
def jit(f):

    def f_jitted(*args):
        avals_in = [raise_to_shaped(get_aval(x)) for x in args]
        jaxpr, consts, out_tree = make_jaxpr(f, *avals_in)
        outs = bind(xla_call_p, *consts, *args, jaxpr=jaxpr, num_consts = len(consts))
        return tree_unflatten(out_tree, outs)
    
    return f_jitted

xla_call_p = Primitive('xla_call')

# Utility for XLA call.
class IDhashable:
    val: Any
    def __init__(self, val):
        self.val = val

    def __hash__(self):
        return id(self.val)

    def __eq__(self, other):
        return type(other) is IDhashable and id(self.val) == id(other.val)

from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc

xe = xc._xla
xops = xc._xla.ops

def xla_call_impl(*args, jaxpr, num_consts):
    consts, args = args[:num_consts], args[num_consts:]
    hashable_consts = tuple(map(IDhashable, consts))
    execute = xla_callable(IDhashable(jaxpr), hashable_consts)
    return execute(*args)

impl_rules[xla_call_p] = xla_call_impl

@lru_cache
def xla_callable(hashable_jaxpr, hashable_consts):
    jaxpr = hashable_jaxpr.val
    typecheck_jaxpr(jaxpr)

    consts = [x.val for x in hashable_consts]
    in_avals = [v.aval for v in jaxpr.in_binders[len(consts):]]

    c = xc.XlaBuilder('xla_call')
    xla_consts = _xla_consts(c, consts)
    xla_params = _xla_params(c, in_avals)
    outs = jaxpr_subcomp(c, jaxpr, xla_consts + xla_params)
    out = xops.Tuple(c, outs)
    compiled = xb.get_backend(None).compile(
        xc._xla.mlir.xla_computation_to_mlir_module(c.build(out)))
    return partial(execure_compiled, compiled, [v.aval for v in jaxpr.outs])

def _xla_consts(c, consts):
    unique_consts = {id(cnst): cnst for cnst in consts}
    xla_consts = {

        id_: xops.ConstantLiteral(c, cnst) for id_, cnst in unique_consts.items()

    }
    return [xla_consts[id(cnst)] for cnst in consts]

def _xla_params(c, avals_in):
    return [xops.Parameter(c, i, _xla_shape(a)) for i, a in enumerate(avals_in)]

def _xla_shape(aval):
    return xc.Shape.array_shape(xc.dtype_to_etype(aval.dtype), aval.shape)

```

Let us now define the transformations for `xla_call_p`, other than its evaluation rule.

```python
# JVP rule for XLA call.
def xla_call_jvp_rule(primals, tangents, *, jaxpr, num_consts):
  del num_consts
  new_jaxpr, new_consts = jvp_jaxpr(jaxpr)
  outs = bind(xla_call_p, *new_consts, *primals, *tangents, jaxpr=new_jaxpr,
              num_consts=len(new_consts))
  n = len(outs) // 2
  primals_out, tangents_out = outs[:n], outs[n:]
  return primals_out, tangents_out
jvp_rules[xla_call_p] = xla_call_jvp_rule

# JVP for the JaxPrs.

@lru_cache
def jvp_jaxpr(jaxpr: Jaxpr) -> tuple[Jaxpr, list[Any]]:
  def jvp_traceable(*primals_and_tangents):
    n = len(primals_and_tangents) // 2
    primals, tangents = primals_and_tangents[:n], primals_and_tangents[n:]
    return jvp(jaxpr_as_fun(jaxpr), primals, tangents)

  in_avals = [v.aval for v in jaxpr.in_binders]
  new_jaxpr, new_consts, _ = make_jaxpr(jvp_traceable, *in_avals, *in_avals)
  return new_jaxpr, new_consts

# VMAP rule for XLA call.
def xla_call_vmap_rule(axis_size, vals_in, dims_in, *, jaxpr, num_consts):
  del num_consts  # Unused
  new_jaxpr, new_consts = vmap_jaxpr(jaxpr, axis_size, tuple(dims_in))
  outs = bind(xla_call_p, *new_consts, *vals_in, jaxpr=new_jaxpr,
              num_consts=len(new_consts))
  return outs, [0] * len(outs)
vmap_rules[xla_call_p] = xla_call_vmap_rule

@lru_cache
def vmap_jaxpr(jaxpr: Jaxpr, axis_size: int, bdims_in: tuple[BatchAxis, ...]
               ) -> tuple[Jaxpr, list[Any]]:
  vmap_traceable = vmap(jaxpr_as_fun(jaxpr), tuple(bdims_in))
  in_avals = [unmapped_aval(axis_size, d, v.aval)
              for v, d in zip(jaxpr.in_binders, bdims_in)]
  new_jaxpr, new_consts, _ = make_jaxpr(vmap_traceable, *in_avals)
  return new_jaxpr, new_consts

def unmapped_aval(axis_size: int, batch_dim: BatchAxis, aval: ShapedArray
                  ) -> ShapedArray:
  if batch_dim is not_mapped:
    return aval
  else:
    shape = list(aval.shape)
    shape.insert(batch_dim, axis_size)
    return ShapedArray(tuple(shape), aval.dtype)


# Abstract Eval XLA call rule.
def xla_call_abstract_eval_rule(*in_types, jaxpr, num_consts):
  del num_consts  # Unused
  jaxpr_type = typecheck_jaxpr(jaxpr)
  if not all(t1 == t2 for t1, t2 in zip(jaxpr_type.in_types, in_types)):
    raise TypeError
  return jaxpr_type.out_types
abstract_eval_rules[xla_call_p] = xla_call_abstract_eval_rule


def xla_call_translation(c, in_avals, out_avals, in_vals, *, jaxpr, num_consts):
  del num_consts, out_avals
  # Calling jaxpr_subcomp directly would inline.
  with ir.InsertionPoint(c.module.body):
    @func.func(*(aval_to_ir_type(aval) for aval in in_avals))
    def inner_xla_call(*params):
      return jaxpr_subcomp(c, jaxpr, params)
    name = c.symbol_table.insert(inner_xla_call.func_op)
  return func.CallOp(inner_xla_call.func_op, in_vals).results
hlo_translations[xla_call_p] = xla_call_translation

```

With all the rules in place, we turn our attention to an important issue - memory persistence for arrays. After the XLA operation is done, we transferred the results back to CPU memory as `np.array`, but most of the time, we need to transfer these results back for the next operation. For that, we'll introduce an Array class to wrap up XLA buffers. 

```python
def handle_result(aval: ShapedArray, buf):  # noqa: F811
  return Array(aval, buf)

class Array:
  buf: Any
  aval: ShapedArray

  def __init__(self, aval, buf):
    self.aval = aval
    self.buf = buf

  dtype = property(lambda self: self.aval.dtype)
  shape = property(lambda self: self.aval.shape)
  ndim  = property(lambda self: self.aval.ndim)

  def __array__(self): return np.asarray(self.buf)
  def __repr__(self):  return repr(np.asarray(self.buf))
  def __str__(self):   return str(np.asarray(self.buf))

  _neg = staticmethod(neg)
  _add = staticmethod(add)
  _radd = staticmethod(add)
  _mul = staticmethod(mul)
  _rmul = staticmethod(mul)
  _gt = staticmethod(greater)
  _lt = staticmethod(less)
input_handlers[Array] = lambda x: x.buf

jax_types.add(Array)
```

With that, we implemented another core feature of JAX. Let's move on to the next part, where we implement some special autodiff functions - `linearize` and `vjp`, which have some caveats. 
### Part 4 - `linearize` and `vjp` 

Here's an diagram summarizing key points of linearize and VJP in JAX. Let us implement them!!

![Pasted image 20241017142629](https://github.com/user-attachments/assets/e14db39e-7f73-4147-832b-0e3d388f5540)

#### Implementing `linearize` JaxPr
Linearize computes the linear approximation of a function, and it operates in the tangent space. Therefore, we want to stage out the linear part of JVP computation to build a JaxPr from a JVP. To do this, we need to perform partial evaluation - to evaluate all the primal values as a tarce, but stage the tangent computations into a Jaxpr. Unlike the previous `make_jaxpr` functions, this approach stages out only those primitive binds with a dependence on tangent inputs.

```python
def split_half(lst: list[Any]) -> tuple[list[Any], list[Any]]:
  assert not len(lst) % 2
  return split_list(lst, len(lst) // 2)

def merge_lists(which: list[bool], l1: list[Any], l2: list[Any]) -> list[Any]:
  l1, l2 = iter(l1), iter(l2)
  out = [next(l2) if b else next(l1) for b in which]
  assert next(l1, None) is next(l2, None) is None
  return out

def linearize_flat(f, *primals_in):
  pvals_in = ([PartialVal.known(x) for x in primals_in] +
              [PartialVal.unknown(vspace(get_aval(x))) for x in primals_in])
  def f_jvp(*primals_tangents_in):
    primals_out, tangents_out = jvp(f, *split_half(primals_tangents_in))
    return [*primals_out, *tangents_out]
  jaxpr, pvals_out, consts = partial_eval_flat(f_jvp, pvals_in)
  primal_pvals, _ = split_half(pvals_out)
  assert all(pval.is_known for pval in primal_pvals)
  primals_out = [pval.const for pval in primal_pvals]
  f_lin = lambda *tangents: eval_jaxpr(jaxpr, [*consts, *tangents])
  return primals_out, f_lin

# This linearize function has JVp and partial evaluation combined.
def linearize(f, *primals_in):
  primals_in_flat, in_tree = tree_flatten(primals_in)
  f, out_tree = flatten_fun(f, in_tree)
  primals_out_flat, f_lin_flat = linearize_flat(f, *primals_in_flat)
  primals_out = tree_unflatten(out_tree(), primals_out_flat)

  def f_lin(*tangents_in):
    tangents_in_flat, in_tree2 = tree_flatten(tangents_in)
    if in_tree != in_tree2: raise TypeError
    tangents_out_flat = f_lin_flat(*tangents_in_flat)
    return tree_unflatten(out_tree(), tangents_out_flat)

  return primals_out, f_lin

def vspace(aval: ShapedArray) -> ShapedArray:
  return raise_to_shaped(aval)  # TODO handle integers?

```

As mentioned, in the `linearize`, there is JVP + general partial information transformation. The workflow is simple, turn a Python callable into outputs of two types - one where all the outputs can be computed from the known outputs, and a partial JaxPr which can only be performed after its required inputs are known[^5].   Think of partial evaluation as "unzipping" one computation into two(one for primal, and one for tangent jaxpr). We kind of only want to form a JaxPr for those operations whose operations must be delayed due to dependence on unknown inputs, which reduces unnecessary evaluations. 

For the reasons mentioned above, let us start our implementation by creating a `PartialVal` class. Partial evaluation will take a list of `PartialVal` representing inputs, and return a list of `PartialVal` outputs along with a jaxpr representing the delayed computation:

```python
class PartialVal(NamedTuple):
  aval: ShapedArray
  const: Any | None
 
  @classmethod
  def known(cls, val: Any):
    return PartialVal(get_aval(val), val)

  @classmethod
  def unknown(cls, aval: ShapedArray):
    return PartialVal(aval, None)

  # To check whether the inputs are known or unknown, so we distribute accordingly. 
  is_known   = property(lambda self: self.const is not None)
  is_unknown = property(lambda self: self.const is     None)

# The transformation API.
def partial_eval_flat(f: Callable, pvals_in: list[PartialVal]
                      ) -> tuple[Jaxpr, list[PartialVal], list[Any]]:
  with new_main(PartialEvalTrace) as main:
    trace = PartialEvalTrace(main)
    tracers_in = [trace.new_arg(pval) for pval in pvals_in]
    outs = f(*tracers_in)
    tracers_out = [full_raise(trace, out) for out in outs]
    pvals_out = [t.pval for t in tracers_out]
    unk_tracers_in  = [t for t in tracers_in  if t.pval.is_unknown]
    unk_tracers_out = [t for t in tracers_out if t.pval.is_unknown]
    jaxpr, consts = tracers_to_jaxpr(unk_tracers_in, unk_tracers_out)
  return jaxpr, pvals_out, consts

```

Now, we'll implement the `PartialEvalTrace` and `PartialEvalTracer`. The difference with the previous versions of the `Trace`, `Tracer` classes is that the interpreter will build JaxPrs on the fly and will keep track of data dependencies. In order to do so, they implement a Bipartite DAG(Directed Acyclic Graph) between the nodes of the `PartialEvalTracer` (representing staged out values) and `JaxprRecipe` nodes (representing formulas for how to compute some values from others). The reason to choose Bipartite graph structure is to simplify dependency management(modular updates and extensions)

These recipe's can be of several types - `JaxprEqnRecipe` (corresponding to `JaxPrEqn`'s primitive application), and constants, lambda binders.

```python

from weakref import ref, ReferenceType

# Wrapper for a lambda recipe.
class LambdaBindingRecipe(NamedTuple):
  pass

# Wrapper for a const recipe.
class ConstRecipe(NamedTuple):
  val: Any

# Wrapper for Jaxpr recipe
class JaxprEqnRecipe(NamedTuple):
  prim: Primitive
  tracers_in: list['PartialEvalTracer']
  params: dict[str, Any]
  avals_out: list[ShapedArray]
  tracer_refs_out: list['ReferenceType[PartialEvalTracer]']

JaxprRecipe = Union[LambdaBindingRecipe, ConstRecipe, JaxprEqnRecipe]

# Partial Eval Tracer - contains boxedup values.
class PartialEvalTracer(Tracer):
  pval: PartialVal
  recipe: JaxprRecipe | None

  def __init__(self, trace, pval, recipe):
    self._trace = trace
    self.pval = pval
    self.recipe = recipe

  aval = property(lambda self: self.pval.aval)

  def full_lower(self):
    if self.pval.is_known:
      return full_lower(self.pval.const)
    return self

```

With these implementations in place, let us now implement `PartialEvalTrace`. Each argument in it corresponds to a `LambdaBindingRecipe` leaf node, and each constant is a `ConstRecipe` leaf node holding a reference to the constant.

The implementation for `process_primitive` is also straightforward. If all inputs are known then we can bind the primitive to the known values(basically evaluate it in Python). On the other hand, if any inputs is unknown, then we stage out into a `JaxprEqnRecipe` representing the primitive application. All but the call to `xla_call_primitive` works on this logic(in the XLA primitive, we require recursive treatment)

```python

class PartialEvalTrace(Trace):
  def new_arg(self, pval: PartialVal) -> Any:
    return PartialEvalTracer(self, pval, LambdaBindingRecipe())

  def lift(self, val: Any) -> PartialEvalTracer:
    return PartialEvalTracer(self, PartialVal.known(val), None)
  pure = lift

  def instantiate_const(self, tracer: PartialEvalTracer) -> PartialEvalTracer:
    if tracer.pval.is_unknown:
      return tracer
    else:
      pval = PartialVal.unknown(raise_to_shaped(tracer.aval))
      return PartialEvalTracer(self, pval, ConstRecipe(tracer.pval.const))

  def process_primitive(self, primitive, tracers, params):
    if all(t.pval.is_known for t in tracers):
      return bind(primitive, *map(full_lower, tracers), **params)
    rule = partial_eval_rules.get(primitive)
    if rule: return rule(self, tracers, **params)
    tracers_in = [self.instantiate_const(t) for t in tracers]
    avals_in = [t.aval for t in tracers_in]
    avals_out = abstract_eval_rules[primitive](*avals_in, **params)
    tracers_out = [PartialEvalTracer(self, PartialVal.unknown(aval), None)
                   for aval in avals_out]
    eqn = JaxprEqnRecipe(primitive, tracers_in, params, avals_out,
                         map(ref, tracers_out))
    for t in tracers_out: t.recipe = eqn
    return tracers_out

partial_eval_rules = {}

```

Now, we can build graph representations of JaxPrs with `PartialEvalTrace` - we just need a mechanism to convert graph representation to standard JaxPr(corresponds to a topological sort of the graph).

```python

def tracers_to_jaxpr(tracers_in: list[PartialEvalTracer],
                     tracers_out: list[PartialEvalTracer]):
  tracer_to_var: dict[int, Var] = {id(t): Var(raise_to_shaped(t.aval))
                                   for t in tracers_in}
  constvar_to_val: dict[int, Any] = {}
  constid_to_var: dict[int, Var] = {}
  processed_eqns: set[int] = set()
  eqns: list[JaxprEqn] = []
  for t in toposort(tracers_out, tracer_parents):
    if isinstance(t.recipe, LambdaBindingRecipe):
      assert id(t) in set(map(id, tracers_in))
    elif isinstance(t.recipe, ConstRecipe):
      val = t.recipe.val
      var = constid_to_var.get(id(val))
      if var is None:
        aval = raise_to_shaped(get_aval(val))
        var = constid_to_var[id(val)] = Var(aval)
        constvar_to_val[var] = val
      tracer_to_var[id(t)] = var
    elif isinstance(t.recipe, JaxprEqnRecipe):
      if id(t.recipe) not in processed_eqns:
        eqns.append(recipe_to_eqn(tracer_to_var, t.recipe))
        processed_eqns.add(id(t.recipe))
    else:
      raise TypeError(t.recipe)

  constvars, constvals = unzip2(constvar_to_val.items())
  in_binders = constvars + [tracer_to_var[id(t)] for t in tracers_in]
  out_vars = [tracer_to_var[id(t)] for t in tracers_out]
  jaxpr = Jaxpr(in_binders, eqns, out_vars)
  typecheck_jaxpr(jaxpr)
  return jaxpr, constvals

def recipe_to_eqn(tracer_to_var: dict[int, Var], recipe: JaxprEqnRecipe
                  ) -> JaxprEqn:
  inputs = [tracer_to_var[id(t)] for t in recipe.tracers_in]
  out_binders = [Var(aval) for aval in recipe.avals_out]
  for t_ref, var in zip(recipe.tracer_refs_out, out_binders):
    if t_ref() is not None: tracer_to_var[id(t_ref())] = var
  return JaxprEqn(recipe.prim, inputs, recipe.params, out_binders)

def tracer_parents(t: PartialEvalTracer) -> list[PartialEvalTracer]:
  return t.recipe.tracers_in if isinstance(t.recipe, JaxprEqnRecipe) else []

## Toposort and stuff

def toposort(out_nodes: list[Any], parents: Callable[[Any], list[Any]]):
  if not out_nodes: return []
  out_nodes = remove_duplicates(out_nodes)

  child_counts = {}
  stack = list(out_nodes)
  while stack:
    node = stack.pop()
    if id(node) in child_counts:
      child_counts[id(node)] += 1
    else:
      child_counts[id(node)] = 1
      stack.extend(parents(node))
  for node in out_nodes:
    child_counts[id(node)] -= 1

  sorted_nodes = []
  childless_nodes = [node for node in out_nodes if not child_counts[id(node)]]
  while childless_nodes:
    node = childless_nodes.pop()
    sorted_nodes.append(node)
    for parent in parents(node):
      if child_counts[id(parent)] == 1:
        childless_nodes.append(parent)
      else:
        child_counts[id(parent)] -= 1

  sorted_nodes = sorted_nodes[::-1]
  check_toposort(sorted_nodes, parents)
  return sorted_nodes

def remove_duplicates(lst):
  seen = set()
  return [x for x in lst if id(x) not in seen and not seen.add(id(x))]

def check_toposort(nodes: list[Any], parents: Callable[[Any], list[Any]]):
  seen = set()
  for node in nodes:
    assert all(id(parent) in seen for parent in parents(node))
    seen.add(id(node))

```

Let us test it in action. We also need to implement the partial evaluation rule `xla_call_p` (to handle JIT and related functions). There are two rules to write, one for trace-time partial evaluation(`xla_call_partial_eval`), and one for partial evaluation of Jaxprs(`xla_call_peval_eqn`) 

```python
# Example usage.
y, sin_lin = linearize(sin, 3.)
print(y, sin(3.))
print(sin_lin(1.), cos(3.))

## Rules for XLA Primitive(trace time and Partial Eval)

def xla_call_partial_eval(trace, tracers, *, jaxpr, num_consts):
  del num_consts  # Unused
  in_unknowns = [not t.pval.is_known for t in tracers]
  jaxpr1, jaxpr2, out_unknowns, num_res = partial_eval_jaxpr(jaxpr, in_unknowns)
  known_tracers, unknown_tracers = partition_list(in_unknowns, tracers)
  known_vals = [t.pval.const for t in known_tracers]
  outs1_res = bind(xla_call_p, *known_vals, jaxpr=jaxpr1, num_consts=0)
  outs1, res = split_list(outs1_res, len(jaxpr1.outs) - num_res)
  res_tracers = [trace.instantiate_const(full_raise(trace, x)) for x in res]
  outs2 = [PartialEvalTracer(trace, PartialVal.unknown(v.aval), None)
           for v in jaxpr2.outs]
  eqn = JaxprEqnRecipe(xla_call_p, res_tracers + unknown_tracers,
                       dict(jaxpr=jaxpr2, num_consts=0),
                       [v.aval for v in jaxpr2.outs], map(ref, outs2))
  for t in outs2: t.recipe = eqn
  return merge_lists(out_unknowns, outs1, outs2)
partial_eval_rules[xla_call_p] = xla_call_partial_eval

def partial_eval_jaxpr(jaxpr: Jaxpr, in_unknowns: list[bool],
                       instantiate: list[bool] | None = None,
                       ) -> tuple[Jaxpr, Jaxpr, list[bool], int]:
  env: dict[Var, bool] = {}
  residuals: set[Var] = set()

  def read(x: Atom) -> bool:
    return type(x) is Var and env[x]

  def write(unk: bool, v: Var) -> None:
    env[v] = unk

  def new_res(x: Atom) -> Atom:
    if type(x) is Var: residuals.add(x)
    return x

  eqns1, eqns2 = [], []
  map(write, in_unknowns, jaxpr.in_binders)
  for eqn in jaxpr.eqns:
    unks_in = map(read, eqn.inputs)
    rule = partial_eval_jaxpr_rules.get(eqn.primitive)
    if rule:
      eqn1, eqn2, unks_out, res = rule(unks_in, eqn)
      eqns1.append(eqn1); eqns2.append(eqn2); residuals.update(res)
      map(write, unks_out, eqn.out_binders)
    elif any(unks_in):
      inputs = [v if unk else new_res(v) for unk, v in zip(unks_in, eqn.inputs)]
      eqns2.append(JaxprEqn(eqn.primitive, inputs, eqn.params, eqn.out_binders))
      map(partial(write, True), eqn.out_binders)
    else:
      eqns1.append(eqn)
      map(partial(write, False), eqn.out_binders)
  out_unknowns = map(read, jaxpr.outs)
  if instantiate is not None:
    for v, uk, inst in zip(jaxpr.outs, out_unknowns, instantiate):
      if inst and not uk: new_res(v)
    out_unknowns = map(op.or_, out_unknowns, instantiate)

  residuals, num_res = list(residuals), len(residuals)
  assert all(type(v) is Var for v in residuals), residuals

  ins1, ins2 = partition_list(in_unknowns, jaxpr.in_binders)
  outs1, outs2 = partition_list(out_unknowns, jaxpr.outs)

  jaxpr1 = Jaxpr(ins1, eqns1, outs1 + residuals)
  jaxpr2 = Jaxpr(residuals + ins2, eqns2, outs2)
  typecheck_partial_eval_jaxpr(jaxpr, in_unknowns, out_unknowns, jaxpr1, jaxpr2)

  return jaxpr1, jaxpr2, out_unknowns, num_res

def typecheck_partial_eval_jaxpr(jaxpr, unks_in, unks_out, jaxpr1, jaxpr2):
  jaxprty = typecheck_jaxpr(jaxpr)    # (a1,  a2) -> (b1, b2 )
  jaxpr1ty = typecheck_jaxpr(jaxpr1)  #  a1       -> (b1, res)
  jaxpr2ty = typecheck_jaxpr(jaxpr2)  # (res, a2) -> b2

  a1, a2 = partition_list(unks_in, jaxprty.in_types)
  b1, b2 = partition_list(unks_out, jaxprty.out_types)
  b1_, res = split_list(jaxpr1ty.out_types, len(b1))
  res_, a2_ = split_list(jaxpr2ty.in_types, len(res))
  b2_ = jaxpr2ty.out_types

  if jaxpr1ty.in_types != a1: raise TypeError
  if jaxpr2ty.out_types != b2: raise TypeError
  if b1 != b1_: raise TypeError
  if res != res_: raise TypeError
  if a2 != a2_: raise TypeError
  if b2 != b2_: raise TypeError

partial_eval_jaxpr_rules = {}

def xla_call_peval_eqn(unks_in: list[bool], eqn: JaxprEqn,
                       ) -> tuple[JaxprEqn, JaxprEqn, list[bool], list[Var]]:
  jaxpr = eqn.params['jaxpr']
  jaxpr1, jaxpr2, unks_out, num_res = partial_eval_jaxpr(jaxpr, unks_in)
  ins1, ins2 = partition_list(unks_in, eqn.inputs)
  out_binders1, out_binders2 = partition_list(unks_out, eqn.out_binders)
  residuals = [Var(v.aval) for v in jaxpr2.in_binders[:num_res]]
  eqn1 = JaxprEqn(xla_call_p, ins1, dict(jaxpr=jaxpr1, num_consts=0),
                  out_binders1 + residuals)
  eqn2 = JaxprEqn(xla_call_p, residuals + ins2,
                  dict(jaxpr=jaxpr2, num_consts=0), out_binders2)
  return eqn1, eqn2, unks_out, residuals
partial_eval_jaxpr_rules[xla_call_p] = xla_call_peval_eqn

```

Now, we can compose `linearize` and `jit` .
```python

#Example usage.

@jit
def f(x):
  y = sin(x) * 2.
  z = - y + x
  return z

y, f_lin = linearize(f, 3.)
y_dot = f_lin(1.)
print(y, y_dot)

```

#### Implementing `vjp` and `grad`

The `vjp` transformation is very similar to linearize, with the only difference being the fact that  we transpose the linear part of the computation before returning it, so our implementation is pretty straightforward. Also, since we have the linear computation as a JaxPr, we can implement the transpose transformation as a JaxPr interpreter. The use of `UndefPrimal` instance is to indicate which arguments we want to transpose(and with respect to what). We also register this as a pytree node as that gives us a handy way to prune these placeholders out of argument lists.

```python
# VJP implementation
def vjp(f, x):
  y, f_lin = linearize(f, x)
  f_vjp = lambda y_bar: transpose(f_lin)(y_bar)
  return y, f_vjp

# Transpose transformation

def vjp_flat(f, *primals_in):
  pvals_in = ([PartialVal.known(x) for x in primals_in] +
              [PartialVal.unknown(vspace(get_aval(x))) for x in primals_in])
  primal_pvals_in, tangent_pvals_in = split_half(pvals_in)
  def f_jvp(*primals_tangents_in):
    primals_out, tangents_out = jvp(f, *split_half(primals_tangents_in))
    return [*primals_out, *tangents_out]
  jaxpr, pvals_out, consts = partial_eval_flat(f_jvp, pvals_in)  # linearize
  primal_pvals, _ = split_half(pvals_out)
  assert all(pval.is_known for pval in primal_pvals)
  primals_out = [pval.const for pval in primal_pvals]
  transpose_inputs = consts + [UndefPrimal(p.aval) for p in tangent_pvals_in]
  f_vjp = lambda *cts: eval_jaxpr_transposed(jaxpr, transpose_inputs, cts)
  return primals_out, f_vjp

def vjp(f, *primals_in):
  primals_in_flat, in_tree = tree_flatten(primals_in)
  f, out_tree = flatten_fun(f, in_tree)
  primals_out_flat, f_vjp_flat = vjp_flat(f, *primals_in_flat)
  primals_out = tree_unflatten(out_tree(), primals_out_flat)

  def f_vjp(*cotangents_out):
    cotangents_out_flat, _ = tree_flatten(cotangents_out)
    cotangents_in_flat = f_vjp_flat(*cotangents_out_flat)
    return tree_unflatten(in_tree, cotangents_in_flat)

  return primals_out, f_vjp

class UndefPrimal(NamedTuple):
  aval: ShapedArray

register_pytree_node(UndefPrimal,
                     lambda u: (u.aval, ()),
                     lambda aval, _: UndefPrimal(aval))

```

Next, we can write `eval_jaxpr_transposed`, along with the transpose rules for all the primitives(which can be linear in at least one argument).

```python

# NB: the analogous function in JAX is called 'backward_pass'
def eval_jaxpr_transposed(jaxpr: Jaxpr, args: list[Any], cotangents: list[Any]
                          ) -> list[Any]:
  primal_env: dict[Var, Any] = {}
  ct_env: dict[Var, Any] = {}

  def read_primal(x: Atom) -> Any:
    return primal_env.get(x, UndefPrimal(x.aval)) if type(x) is Var else x.val

  def write_primal(v: Var, val: Any) -> None:
    if type(val) is not UndefPrimal:
      primal_env[v] = val

  def read_cotangent(v: Var) -> Any:
    return ct_env.pop(v, np.zeros(v.aval.shape, v.aval.dtype))

  def write_cotangent(x: Atom, val: Any):
    if type(x) is Var and val is not None:
      ct_env[x] = add(ct_env[x], val) if x in ct_env else val

  map(write_primal, jaxpr.in_binders, args)
  map(write_cotangent, jaxpr.outs, cotangents)
  for eqn in jaxpr.eqns[::-1]:
    primals_in = map(read_primal, eqn.inputs)
    cts_in = map(read_cotangent, eqn.out_binders)
    rule = transpose_rules[eqn.primitive]
    cts_out = rule(cts_in, *primals_in, **eqn.params)
    map(write_cotangent, eqn.inputs, cts_out)

  return [read_cotangent(v) for v, x in zip(jaxpr.in_binders, args)
          if type(x) is UndefPrimal]

transpose_rules = {}

# Rules

def mul_transpose_rule(cts, x, y):
  z_bar, = cts
  assert (type(x) is UndefPrimal) ^ (type(y) is UndefPrimal)
  return [mul(z_bar, y), None] if type(x) is UndefPrimal else [None, mul(x, z_bar)]
transpose_rules[mul_p] = mul_transpose_rule

def neg_transpose_rule(cts, x):
  ybar, = cts
  assert type(x) is UndefPrimal
  return [neg(ybar)]
transpose_rules[neg_p] = neg_transpose_rule

def add_transpose_rule(cts, x, y):
  z_bar, = cts
  return [z_bar, z_bar]
transpose_rules[add_p] = add_transpose_rule

def reduce_sum_transpose_rule(cts, x, *, axis):
  y_bar, = cts
  return [broadcast(y_bar, x.aval.shape, axis)]
transpose_rules[reduce_sum_p] = reduce_sum_transpose_rule

def xla_call_transpose_rule(cts, *invals, jaxpr, num_consts):
  del num_consts  # Unused
  undef_primals = [type(x) is UndefPrimal for x in invals]
  transposed_jaxpr, new_consts = transpose_jaxpr(jaxpr, tuple(undef_primals))
  residuals, _ = partition_list(undef_primals, invals)
  outs = bind(xla_call_p, *new_consts, *residuals, *cts,
              jaxpr=transposed_jaxpr, num_consts=len(new_consts))
  outs = iter(outs)
  return [next(outs) if undef else None for undef in undef_primals]
transpose_rules[xla_call_p] = xla_call_transpose_rule

@lru_cache
def transpose_jaxpr(jaxpr: Jaxpr, undef_primals: tuple[bool, ...]
                    ) -> tuple[Jaxpr, list[Any]]:
  avals_in, avals_out = typecheck_jaxpr(jaxpr)
  traceable = partial(eval_jaxpr_transposed, jaxpr)
  args = [UndefPrimal(a) if u else a for a, u in zip(avals_in, undef_primals)]
  trans_jaxpr, consts, _ = make_jaxpr(traceable, tuple(args), tuple(avals_out))
  typecheck_jaxpr(trans_jaxpr)
  return trans_jaxpr, consts

```

Now that we can linearize and transpose, we can finally write `grad`.

```python
# Grad implementation.
def grad(f):
  def gradfun(x, *xs):
    y, f_vjp = vjp(f, x, *xs)
    if np.shape(y) != (): raise TypeError
    x_bar, *_ = f_vjp(np.ones(np.shape(y), np.result_type(y)))
    return x_bar
  return gradfun

# Example usage.
def f(x):
  y = sin(x) * 2.
  z = - y + x
  return z

print(grad(f)(3.))

```

Finally, we are done with our implementation!!! Give yourself a pat on the back, you now have your version of JAX, in Python, spelled out completely. I'll be maintaining a repository for this particular blog, and will update things as I learn more about this amazing library. Would like to thank Lucas Beyer for replying to my tweet to motivate me to understand more about this framework.

### Resources
1. [Autodiadax](https://jax.readthedocs.io/en/latest/autodidax.html) - Implementing the JAX Core from scratch.
2. [PyTorch is Dead, Long Live JAX](https://neel04.github.io/my-website/blog/pytorch_rant/) - Neel Gupta.
3. [Is JAX better than PyTorch](https://www.reddit.com/r/learnmachinelearning/comments/16vgfed/is_jax_a_better_choice_to_focus_on_over_pytorch/) - Reddit Discussion.
4. [JAX - The Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) - JAX Docs


[^1]: A better and more detailed explanation(without abstracting any math) goes something like this(according to JAX Docs):
	>To answer that, first think about how you could use a JVP to build a full Jacobian matrix. If we apply a JVP to a one-hot tangent vector, it reveals one column of the Jacobian matrix, corresponding to the nonzero entry we fed in. So we can build a full Jacobian one column at a time, and to get each column costs about the same as one function evaluation. That will be efficient for functions with “tall” Jacobians, but inefficient for “wide” Jacobians.

	>If you’re doing gradient-based optimization in machine learning, you probably want to minimize a loss function from parameters in $\mathbb{R}^N$ to a scalar loss value in $\mathbb{R}$. That means the Jacobian of this function is a very wide matrix: $\partial{f(x)} \in \mathbb{R}^{1 \times n}$, which we often identify with the Gradient vector $\nabla f(x) \in \mathbb{R}^n$. Building that matrix one column at a time, with each call taking a similar number of FLOPs to evaluate the original function, sure seems inefficient! In particular, for training neural networks, where f is a training loss function and n can be in the millions or billions, this approach just won’t scale. To do better for functions like this, we just need to use reverse-mode autodiff.

[^2]: For a refresher on autodiff, refer [this](https://news.ycombinator.com/item?id=37256903) Hackernews post with some really awesome explanations by really awesome people(and possible pitfalls). 

[^3]: This is often referred to as the "omnistaging" issue in the JAX-ML repo. Even if I try, I can't explain in the detail this [PR](https://github.com/jax-ml/jax/pull/3370) is described. Highly recommend to read it.

[^4]: As per the official documentation, this is what happens in staged processing. 
	> In this case, `make_jaxpr` puts its `JaxprTrace` at the top of the interpreter stack, and no transformations lower in the stack, which might enter via closed-over Tracers, are applied to the Python callable as we trace it. (Transformations applied within the Python callable are applied as usual, being added to the stack above the JaxprTrace.) Instead, the transformations lower in the stack are later applied to the call primitive, and the call primitive’s rules must then transform the jaxpr itself. Because we trace to a jaxpr up-front, this approach can’t support data-dependent Python control flow, but it is more straightforward to implement. We refer to this kind of higher-order primitive as an “initial-style higher-order primitive”, and say that its jaxpr-processing transformation rules are “initial-style transformation rules.”

[^5]: This transformation is tricky to summarize in a type signature. If we assume the input function’s type signature is `(a1, a2) -> (b1, b2)`, where `a1` and `a2` represent the known and unknown inputs, respectively, and where `b1` only has a data dependency on `a1` while `b2` has some data dependency on `a2`, then we might write: `partial_eval : ((a1, a2) -> (b1, b2)) -> a1 -> exists r. (b1, r, (r, a2) -> b2)`

---

## Arrakis - A toolkit to conduct, track and visualize mechanistic interpretability experiments.

- **Project Name :** Arrakis.
- **Project Description :** Arrakis is a library to conduct, track and visualize mechanistic interpretability experiments.
- **Project Home :** [Github Repo](https://github.com/yash-srivastava19/arrakis)
- **Project Documentation :** [Read the Docs](https://arrakis-mi.readthedocs.io/en/latest/README.html)
- **PyPI Home :** [PyPi Home](https://pypi.org/project/arrakis-mi/)

## Introduction
_"The greatest enemy of knowledge is not ignorance, it is the illusion of knowledge."_ 
                                                                                 -Daniel J. Boorstin
                                                                                 
 Understanding how we think is a question that has perplexed us for a long time. There have been countless theories, many thought experiments, and a lot of experiments to try to unravel the mysteries of the brain. With time, we have become more aware of our brain's working, and in my honest, and a little biased opinion, Artificial Intelligence has come the closest to model our mystery organ.

This is one of the reasons why Interpretability as a field makes a lot sense to me. It tries to unravel the inner working of one of the most successful proxy of human brain - Large Language Model. Mechanistic Interpretability is on the approach in AI alignment to reverse engineer neural networks and understand the inner workings.

Although this field is really exciting(and challenging), researchers have made quite a significant progress in coming up with hypothesis and conducting experiments to prove their validity. Heavyweights like Anthropic, Google Deepmind, EleutherAI, Leap Labs and many open source organizations have been pushing the boundaries of knowledge in this field, and many researchers are pivoting to Interpretability to advance this field.

## What and Why?
Arrakis is one such tool which I made for the community to provide a means for researchers working in this field to quickly iterate on their ideas and run experiments. Arrakis is a complete suite to conduct MI experiments. It is still very much in it's infancy, and I hope to evolve this project with the support of the community.

## Key Idea
The speed at which a researchers innovate is limited by its iteration speed. The more you work on experiments, the more you'll realize that the biggest biggest bottleneck for any project is iteration speed. Arrakis is made so that this doesn't happen. The core principle behind Arrakis is decomposability. Arrakis provides 10+ plug and play tools(plus the ability to make your own) to do common experiments in MI. Other than that, there are several other features such as version control, model profiling and logging provided as add-ons. This makes experimentation really flexible, and at the same time, reduces the iteration time bottleneck. Everything in Arrakis is made in this plug and play fashion.

## Walkthrough
Let us look at how Arrakis works in practice. After the installation, here are the steps to make black box models transparent!
### Create a `HookedAutoModel` from Config
`HookedAutoModel` is a wrapper around the HUggingface PreTrainedModel, and the only difference between them is a single decorator on the forward function. Everything just works out of the box - and the functionality can be removed without affeccting the model just by removing the decorator. First, create a `HookedConfig` for the model you want to support with the required parameters. Then, create a `HookedAutoModel` from the config. As of now, these models are supported - gpt2, gpt-neo, gpt-neox, llama,gemma,phi3,qwen2, mistral, stable-lm

```python
from arrakis.src.core_arrakis.activation_cache import *

config = HookedAutoConfig(name="llama", 
    vocab_size=50256, 
    hidden_size=8, 
    intermediate_size=2, 
    num_hidden_layers=4, 
    num_attention_heads=4,
    num_key_value_heads=4)

model = HookedAutoModel(config)
```
### Setup Interpretability Bench

`IntepretabilityBench` is the workspace where researchers can conduct experiments. At it’s core, the whole purpose of Arrakis is to conduct MI experiment, so it is made keeping accessibility in mind. Just derive from the `BaseInterpretabilityBench` and instantiate an object(`exp` in this case). This object provides a lot of function out-of the box based on the “tool” you want to use for the experiment, and have access to the functions that the tool provides. You can also create your own tool(read about that [here](https://arrakis-mi.readthedocs.io/en/latest/README.html#extending-arrakis) )

```python
from arrakis.src.core_arrakis.base_bench import BaseInterpretabilityBench

class MIExperiment(BaseInterpretabilityBench):
    def __init__(self, model, save_dir="experiments"):
        super().__init__(model, save_dir)
        self.tools.update({"custom": CustomFunction(model)})

exp = MIExperiment(model)
```

Apart from access to MI tools, the object also provides you a convinient way to log your experiments. To log your experiments, just decorate the function you are working with `@exp.log_experiment`, and that is pretty much it. The function creates a local version control on the contents of the function, and stores it locally. You can run many things in parallel, and the version control helps you keep track of it.
```python
# Step1: Create a function where you can do operations on the model.

@exp.log_experiment   # This is pretty much it. This will log the experiment.
def attention_experiment():
    print("This is a placeholder for the experiment. Use as is.")
    return 4

# Step 2: Then, you run the function, get results. This starts the experiment.
attention_experiment()

# Step 3: Then, we will look at some of the things that logs keep a track of
l = exp.list_versions("attention_experiment")  # This gives the hash of the content of the experiment.
print("This is the version hash of the experiment: ", l)

# Step 4: You can also get the content of the experiment from the saved json.
print(exp.get_version("attention_experiment", l[0])['source'])  # This gives the content of the experiment.

# Apart from these tools, there are also `@exp.profile_model`(to profile how # much resources the model is using) and `@exp.test_hypothesis`(to test hypothesis). Support of more tools will be added as I get more feedback from the community.
```

### Create you experiments

By default, Arrakis provides a lot of Anthropic’s Interpretability experiments(Monosemanticity, Residual Decomposition, Read Write Analysis and a lot more. These are provided as tools, so in your experiments, you can plug and play with them and conduct your experiments. Here’s an example of how you can do that.

```python
# Making functions for Arrakis to use is pretty easy. Let's look it in action.

# Step 1: Create a function where you can do operations on the model. Think of all the tools you might need for it.
# Step 2: Use the @exp.use_tools decorator on it, with additional arg of the tool.
# Step 3: The extra argument gives you access to the function. Done.

@exp.use_tools("write_read")  # use the `exp.use_tools()` decorator.
def read_write_analysis(read_layer_idx, write_layer_idx, src_idx, write_read=None):  # pass an additional argument.
    # Multi-hop attention (write-read)

    # use the extra argument as a tool.
    write_heads = write_read.identify_write_heads(read_layer_idx)  
    read_heads = write_read.identify_read_heads(write_layer_idx, dim_idx=src_idx) 

    return {
        "write_heads": write_heads, 
        "read_heads": read_heads
    }

print(read_write_analysis(0, 1, 0)) # Perfecto!

```

### Visualize the Results
Generating plots is Arrakis is also plug and play, just add the decorator and plots are generated by default. Read more about the graphing docs [here](https://arrakis-mi.readthedocs.io/en/latest/README.html#)

```python
from arrakis.src.graph.base_graph import *

# Step 1: Create a function where you can want to draw plot.
# Step2: Use the @exp.plot_results decorator on it(set the plotting lib), with additional arg of the plot spec. Pass input_ids here as well(have to think on this)
# Step3: The extra argument gives you access to the fig. Done.

exp.set_plotting_lib(MatplotlibWrapper) # Set the plotting library.

@exp.plot_results(PlotSpec(plot_type = "attention", data_keys = "h.1.attn.c_attn"), input_ids=input_ids) # use the `exp.plot_results()` decorator.
def attention_heatmap(fig=None): # pass an additional argument.
    return fig

attention_heatmap() # Done.
plt.show()
```

These are three upper level classes in Arrakis. One is the `InterpretabilityBench` where you conduct experiments, the second is the `core_arrakis` where I’ve implemented some common tests for Transformer based model and the third is the `Graphing`.

### Resources
- Transformers Circuits Thread - https://transformer-circuits.pub 

---

## On High Agency and Work Ethic
*Better beware of notions like genius and inspiration; they are a sort of magic wand and should be used sparingly by anybody who wants to see things clearly.* 
                                                                                                                                          José Ortega y Gasset

One of the traits of highly successful and intelligent people is something called high agency and relentless work ethic. If you have been under the impression that all of it is talent, that is simply not true, I’m pretty sure most of the time, it is having high agency and work ethic. 

If you are aware about the life of Terry Tao, you can quickly judge that whatever he has achieved is purely based on talent, but sorry to break the bubble, it is simply not true. In his own words, “Progress is obtained naturally and cumulatively as a consequence of hard work, directed by intuition, literature, and a bit of luck”. Having talent is a gift, not an excuse. More so, it is equally important on how one develops and nurtures their talent.

Hard Work is a talent, Discipline is also a talent. Talents are not always latent which are waiting for some trigger to just spring into your life - that’s what happens in movies, and in reality it is simply not true. If you dive deeply into the lives of people who are even remotely successful, you can deduce most of the time what made them who they are today, is work ethic and being high agency.

High agency people bend reality to their will. When presented with a problem, a high agency person deals with it head on. They take responsibility and ownership of the problem, and usually get to the core of the problem and try solving it - this shows how much you care about the problem, and to what limit you can pursue it. They don’t limit themselves, they figure shit out, and usually this is a huge differentiator. If you do everything, you will win.

Work ethic is something that fuels high agency. If you want to have a take away from this blog, it should be that every(and any) quality can be cultivated - all it requires is effort. Do whatever it takes, and be in the driver seat of your own life - take your destiny in your own hands. Read books, take inspiration from other people. In the end, love what you do purely, and agency will just follow. Only condition is that you have to be ruthless in implementation and do high quality work.  

Speaking from personal experience, I wasn’t someone who used to have a work ethic, as I was told from a really early age that I was smart. I was writing computer games before I knew how to use a pen and read encyclopedias about things I didn't even know existed. School was easy, as with very little effort, I was among the top students. With time, I got to know the value of work ethic, and just how important it is to have it. I worked a lot on this particular area in my life, and now I’m in a much better place. I know my capabilities, and can improve on it endlessly.

People are obsessed with prodigies and often believe having such identity can set them apart. They swallow the pill that “they are not smart” and often don’t pursue(or quickly give up) what they want because of this. It takes time, patience, resilience, a deep sense of affection for the craft and most importantly - hard work. Don’t aim to be a prodigy, be Sisyphus :) 

---

## Leviathan - Let's modify and improve the Transformer model(from scratch).
Transformers have been the go-to architecture for modern deep learning stack from the moment they were introduced. They are quite powerful, and many successful architectures have been made from them which have been SOTA in their respective domain.

Tackling on changing the architecture of Transformers is a easy problem, and resources such as “[Illustrated Transformers](https://jalammar.github.io/illustrated-transformer/)” have helped a lot deeply understand the intricacies of it. It is one thing to read the paper and do the math, and another to debug like a child. Fortunately, this was a problem that I’ve been thinking a lot about in the past, the implementation was a different beast. All that was left was running some experiments and comparing it to the transformer baseline.

I’ve tried to implement this model called “Leviathan” which is a modified version of Transformers that uses correlation score(an analogy taken from signal processing). The implementation can be found [here](https://github.com/yash-srivastava19/attention-free-revolution), and here’s my reasoning on why I think it performs similar to Transformers.

**Why Correlation ?**
I read somewhere that self-attention(simple scaled dot product attention) can be seen as a Graph Neural Network, where each token in the input sequence is a mode, and edges denote the relationship between each token, and that attention layers is a directed acyclic graph - which makes sense as different context gives different meanings to how different tokens are connected.

If we think of tokens as signals, attention can be used to capture long range dependencies in signals, and correlation is great when there is delay in the signals. Correlation, or to be more general, Cross Correlation is a more general way to find similarity between signals. Try thinking of dot product as basically just cross correlation with zero lag.

Suppose instead of a vanilla dot product, we used cross correlation - which ultimately uses a sliding window product, and now due to the “lag” in the tokens, there are effectively more nodes in the graph, which basically allows for more ways in which signals(tokens) can be connected. Having more nodes means we can learn rich features, as there are now more ways in which a bunch of tokens interact.

Architectural Design 
I wanted to have the correlation metric to be a drop in replacement to the scaled dot product attention. I tried to implement the algorithm using `scipy` signals module which looks something like this : 

![image](https://github.com/yash-srivastava19/yash-srivastava19.github.io/assets/85068689/17f94687-e93b-462f-9234-e71d07aab002)

Instead of having Transformer architectures, I named my architecture with correlation scores as Leviathan. Leviathan language models are very similar to GPT2 in terms of implementation. 

**Experimentation Results**
Although I’m very much limited by the availability of GPU, (only having access to free tier GPUs in Colaboratory)  I had to experiment with small models, and therefore the results might be underwhelming when compared to conventional approaches. However, I’ve kept track of some model statistics using Weights and Biases that can be found [here](https://api.wandb.ai/links/ysrivastava82/xt0v4om9).(this contains the negative log likelihood error, and we’ll do one for bits per char as well - which can be found [here](https://wandb.ai/ysrivastava82/uncategorized/reports/Leviathan-BPC-Characteristics---Vmlldzo2Nzg3ODQz?accessToken=afn77rhm3vg8euliqccj7s5scoog1doe2sfnry8t6l6rk8w18wi9ai075dt956v9)  !) As far as general results go, for a small Leviathan model of around ~50K parameters on the tiny shakespeare dataset, the model logged validation loss from around 4 to 2.7 after 7500 epochs for a simple model with 2  heads and 512 context length(from 3.6 to 1.5 BPC in the same model, but for 6500 epochs(In the WANDB report, I accidently put the negative BPC, and realized only after complete training )). The training notebook can be found [here](https://colab.research.google.com/drive/12IRA4AOlria3n2gd1SkAMnq3kC_QqYoN?usp=sharing). 

**What more do we need ?**
I did this project as I wanted to understand the attention mechanism in detail, and what better way to do it than dismantling it piece by piece and understanding each component. This project was also received very well within the github community, as it managed to get 8 stars and several discussions for which I’m really glad. 

I want to continue to do more experimentd in this project as it was something that I really enjoyed(and also, a problem that really frustrated me(but in a good way)). I have thought of making a sparse Mixture of Experts language model based on Leviathan architectures (which can be found in the “future” folder in the git repository for the project). 

Apart from that, I’ve also thought of making a separate architecture using Involution scores - which is basically a mix between attention and convolution. Involution is more effective and efficient than CNNs, and is much simpler than self-attention.

---

## Design for Generality - Dated(06/04/2024)
Designing is a complicated process, not a complex process. What I mean particularly is that the latter is analogous to difficult, while the former means it just has a lot of moving parts. The basic task of designer is to reduce the entropy of moving parts and make sense of it, without losing much generality. There are many examples you can think of, I would like to go with Transformers.

### Learning design principles with Transformers

When working on big projects, it is often advised to be as general as possible - I for once disagree to some extent with this statement. I believe you need to be general in limit of infinite exploration, that is, you need to be general both in the starting and end, but in different amounts. These amounts are defined by the scope and requirements of the project.
This is the advantage of working with a cracked team - you get to bear the benefits of the design process and it becomes an inspiration for many. Transformer architecture captures this pretty well.
Residual stream is basically a stream connecting the inputs and outputs of the entire system. Any operation that happens in the entire architecture branches from the residual stream, and "reads" and "writes" to it only. Designing in this way not only reduces the entropy drastically, but also allows to rapidly prototype different parts. 
Most confusion or complications stems from the fact that it is very unclear what that "part" does in the entire system. This gives rise to question such as where does it go? what are the inputs, what are the outputs, what are the connections with different parts et. al. Entropy reduction is not a loss of generality, but rather a tedious process to understand what works together and what not. It fells like a loss of generality, but I can assure that it is basically a lossless compression technique(like a .zip file) , so that we can better attend to the information at hand.

---

## Footnotes
