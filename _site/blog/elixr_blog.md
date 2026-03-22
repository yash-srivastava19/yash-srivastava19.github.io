Note: I had just finished Karpathy's micrograd video and understood, for the first time, exactly how backprop works. Then I started wondering: the math works over reals because derivatives are real. What happens if the values are complex? That question took a weekend to start answering, and I'm still not sure I have it right.

# Elixr: What if PyTorch Used Complex Numbers?

- **Project Home:** [github.com/yash-srivastava19/Elixir](https://github.com/yash-srivastava19/Elixir)
- **Language:** Python

---

## The starting point: micrograd

If you haven't watched Karpathy's [micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0), go watch it. It's one of the best explanations of how neural network training works at the mechanical level. The core idea: every mathematical operation in a forward pass can be represented as a node in a computation graph, and you can automatically compute gradients by traversing that graph in reverse and applying the chain rule.

Karpathy's `micrograd` does this for scalar real values. The `Value` class wraps a number, tracks which operations produced it, and stores a `_backward` closure that knows how to compute the local gradient contribution:

```python
class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set()

    def __mul__(self, other):
        out = Value(self.data * other.data)
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
```

Build up a forward pass, call `backward()` on the output, and every node accumulates its gradient automatically. That's it. That's automatic differentiation.

I understood this. I felt good about it. Then the question arrived: **what if `data` wasn't a real number?**

---

## Why complex-valued networks

Complex numbers show up in several places in deep learning - signal processing, MRI reconstruction, radar systems, some quantum computing simulations. The standard approach is to decompose a complex number into its real and imaginary parts and treat them as two separate real-valued channels. It works, but it throws away the algebraic structure.

The alternative is to build a network that operates on complex numbers natively. For that you need:

1. Complex arithmetic with gradient tracking
2. A notion of "derivative" that makes sense for complex-valued functions
3. Activation functions that either respect complex structure or split into real/imaginary parts

The second point is where it gets genuinely hard, and where Elixr is doing something unusual.

---

## Wirtinger calculus: derivatives that work for complex functions

Real calculus defines the derivative as `lim (f(x+h) - f(x)) / h`. For complex functions, there's a complication. A complex function `f(z)` is only differentiable in the full complex sense (holomorphic) if it satisfies the Cauchy-Riemann equations. Most neural network operations, including ReLU, aren't holomorphic.

The standard workaround is **Wirtinger calculus**, which defines two partial derivatives for a function `f(z, z*)`:

```
dL/dz    (treating z* as constant)
dL/dz*   (treating z as constant)
```

For gradient descent on a real-valued loss, what you actually want is `dL/dz*`, the Wirtinger conjugate derivative. PyTorch implements exactly this for its complex tensor support. Elixr takes a more direct route: it wraps a `Complex` class carrying real and imaginary components, tracks both through the computation graph, and computes gradients for each component separately.

```python
class Complex:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __mul__(self, other):
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        real = self.real * other.real - self.imag * other.imag
        imag = self.real * other.imag + self.imag * other.real
        return Complex(real, imag)

    def modulus(self):
        return (self.real**2 + self.imag**2) ** 0.5

    def conjugate(self):
        return Complex(self.real, -self.imag)
```

The `ComplexValue` class wraps this with autograd tracking:

```python
class ComplexValue:
    def __init__(self, complex_num):
        self.data = complex_num
        self.grad = Complex(0.0, 0.0)
        self._backward = lambda: None
        self._prev = set()
```

Each operation stores a closure that computes the gradient contribution and accumulates it into `.grad.real` and `.grad.imag`. The backward pass walks the computation graph in reverse topological order, calling each closure.

---

## The honest question: does the gradient make sense?

There's an open issue on the repo titled "Does the Complex Gradient makes sense?" It's labelled `help wanted, question`. I'm asking sincerely.

The mechanics work - you can run a forward pass, call backward, and get numbers out. But whether those numbers are correct in the Wirtinger sense, and whether descending along them actually minimizes a loss, is something I haven't proven. The gradient computation is intuitive (track real and imaginary parts separately through the chain rule) but intuition and correctness aren't the same thing in complex analysis.

The architecture is there. The math needs more rigour before I'd trust it for anything beyond experimentation. If you work on complex-valued networks and want to tell me I've gotten the gradients wrong, or right, I genuinely want to know.

---

## What the library looks like

```
engine.py         - ScalerValue: real-valued autograd (micrograd-style)
complex_engine.py - ComplexValue: complex-valued autograd
nn.py             - Neuron, Layer, MLP built on ScalerValue
trace_graph.py    - Graphviz visualization of the computation graph
```

The graph visualization is one of the more useful parts. After a forward pass, you can render the entire computation graph as a PNG and see exactly what operations are connected to what, and which gradients flow back through which paths:

```python
from trace_graph import draw_dot

# after a forward pass
dot = draw_dot(loss)
dot.render('grad_graph', format='png')
```

This helped me understand where gradient computation was going wrong more than any amount of print statements.

---

## Where this leads

The reason this is interesting beyond being a learning project: if you can do autograd over complex numbers, the natural next step is quaternions - four-component hypercomplex numbers that are particularly useful for representing 3D rotations. A quaternion-valued network that trains end-to-end on 3D spatial data, with gradients flowing through quaternion arithmetic natively, would be a genuinely unusual thing. No decomposition into real channels, no loss of rotational structure.

That's the direction. The complex number layer is the foundation that needs to be correct first.

---

## Try it

```bash
git clone https://github.com/yash-srivastava19/Elixir
cd Elixir
```

The code is small enough to read in an afternoon. If the Wirtinger gradient question is something you have an answer to, or if you want to help build the quaternion extension, the repo is MIT licensed and PRs are open.
