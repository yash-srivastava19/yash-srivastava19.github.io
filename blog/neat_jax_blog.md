---
layout: post
title: "Shoving NEAT into JAX: When Evolving Topologies Meet Static Computation Graphs"
---

Note: This started as a passion project. It remains a passion project. I have made peace with this.

# Shoving NEAT into JAX: When Evolving Topologies Meet Static Computation Graphs

- **Project Home:** [github.com/yash-srivastava19/NEAT-JAX](https://github.com/yash-srivastava19/NEAT-JAX)

---

NEAT is from 2002. JAX is from 2018. Making them talk to each other taught me more about both than reading any paper did.

---

## What is NEAT, and why should you care?

**NEAT** — Neuroevolution of Augmenting Topologies — is a genetic algorithm for evolving neural networks, introduced by Stanley and Miikkulainen in 2002. What makes it interesting, and what makes it hard, is that it doesn't just evolve *weights*. It evolves *topology*. The structure of the network itself — how many nodes, how they're connected — changes over generations.

The life cycle of a NEAT population looks like this:

```
Generation 0:
  Population of minimal networks (input → output, no hidden nodes)

Each generation:
  1. Evaluate fitness of every network on the task
  2. Group networks into species by structural similarity
  3. Select survivors, weighted by fitness within each species
  4. Reproduce: crossover two parents to produce offspring
  5. Mutate offspring: add a node, add a connection, perturb a weight
  6. Repeat

Over time:
  Networks grow more complex, accumulating structure that helps them survive
```

The "augmenting topologies" part is the key innovation. Networks *start* simple — just inputs wired directly to outputs — and grow structure only when that structure earns its keep. Innovations are protected by **speciation**: networks that look structurally similar compete with each other, giving new mutations time to develop before they have to fight against well-optimized mature networks.

This is elegant. It's also the source of every headache I ran into.

---

## What is EvoJAX, and why did I want to use it?

[EvoJAX](https://github.com/google/evojax), from Tang, Tian, and Ha at Google Brain, is a hardware-accelerated neuroevolution toolkit built on JAX. The pitch is straightforward: if your neuroevolution algorithm can be expressed in JAX, you get TPU and GPU acceleration essentially for free. Evolution strategies, CMA-ES, population-based training — all of it running in parallel across accelerators.

The canonical benchmark task in the EvoJAX repo is **neural slime volleyball**: a two-player physics game where each side controls a "slime" that must hit a ball over a net. The observation space is simple (ball position and velocity, self position), the action space is small (move left, right, jump), but the task requires reactive, competitive behavior that makes it a surprisingly rich benchmark for evolved controllers. If you haven't seen the GIF in the EvoJAX README of two slimes evolved entirely by gradient-free methods vollying back and forth, go look at it — it's a better argument for neuroevolution than any paper abstract.

EvoJAX works beautifully for fixed-topology neuroevolution. You define a network architecture once. The entire population shares that architecture with different weights. You vectorize the fitness evaluation across the population. Everything stays in JAX arrays. Everything is JIT-compilable. It's fast.

NEAT's topology mutation breaks all of this.

---

## The fundamental incompatibility

If you haven't used JAX before, I wrote about it [here](/blog/jax_blog). The short version: JAX is built around **functional transformations** applied to arrays with **static shapes**. When you call `jax.jit` on a function, JAX traces through it once to build a computation graph, then compiles that graph to XLA. The shapes of every intermediate tensor must be knowable at compile time.

NEAT's genome is a dynamic data structure. Here's what a minimal genome looks like in pure Python:

```python
# A NEAT genome: Python dicts with variable-length lists
genome = {
    "nodes": [
        {"id": 0, "type": "input"},
        {"id": 1, "type": "output"},
        # nodes get ADDED here during mutation
    ],
    "connections": [
        {"in": 0, "out": 1, "weight": 0.7, "enabled": True, "innov": 1},
        # connections get ADDED here during mutation
    ]
}
```

And here's what NEAT's genetic operators look like conceptually:

```python
# Standard NEAT mutation: add a node
def mutate_add_node(genome):
    # Pick a connection, disable it, insert a new node in the middle
    conn = random.choice(genome["connections"])
    conn["enabled"] = False
    new_node = {"id": new_id(), "type": "hidden"}
    genome["nodes"].append(new_node)           # <-- list grows
    genome["connections"].append(new_conn_1)   # <-- list grows
    genome["connections"].append(new_conn_2)   # <-- list grows
    return genome

# Standard NEAT crossover: combine two parent genomes
def crossover(parent1, parent2):
    child_connections = []
    for innov in all_innovation_numbers(parent1, parent2):
        if innov in parent1 and innov in parent2:
            child_connections.append(random.choice([parent1[innov], parent2[innov]]))
        elif innov in parent1:
            child_connections.append(parent1[innov])  # disjoint/excess gene
    # child may have different number of nodes than either parent
    return build_genome(child_connections)
```

Now try to put this inside JAX. The moment you `jax.jit` a forward-pass function that accepts a genome, JAX needs to know: how many nodes? how many connections? Those are the array shapes. But those shapes are *the thing that's evolving*. Generation 1 genomes have 2 nodes. Generation 50 genomes might have 15. You cannot trace a function that changes its own tensor shapes.

This is what the actual crash looks like when you try to naively mix Python genome evolution with JAX array evaluation:

```python
@jax.jit
def evaluate_genome(genome_arrays, obs):
    # JAX expects fixed shapes here.
    # If genome_arrays["weights"] has shape (2, 3) for one individual
    # and shape (5, 7) for another, JAX cannot compile this.
    weights = genome_arrays["weights"]   # shape: ???
    hidden = jax.nn.relu(obs @ weights)
    return hidden

# This works fine for individual 0 (2 nodes, 3 connections)
evaluate_genome(to_jax(population[0]), obs)

# This triggers a recompilation — shape changed
# With large populations this becomes catastrophically slow
evaluate_genome(to_jax(population[1]), obs)  # different topology
```

In the NEAT-JAX repo, `ind.py` and `neat.py` hit exactly this wall. The genome lives as Python lists (because they need to grow). The forward pass wants JAX arrays (for speed). Every time you convert between them with different population members, you're either recompiling or lying to JAX about the shapes.

---

## Workarounds I tried

### 1. Max-topology padding

The cleanest workaround is to define an upper bound on network size upfront, and pad every genome to that size. A genome with 3 hidden nodes in a world where the max is 20 just has 17 "disabled" nodes that contribute nothing to the forward pass.

```python
MAX_NODES = 32
MAX_CONNECTIONS = 128

def genome_to_fixed_arrays(genome):
    # Weight matrix: always (MAX_NODES, MAX_NODES)
    W = jnp.zeros((MAX_NODES, MAX_NODES))
    # Mask: 1.0 where connection is enabled, 0.0 otherwise
    mask = jnp.zeros((MAX_NODES, MAX_NODES))

    for conn in genome["connections"]:
        if conn["enabled"]:
            W = W.at[conn["in"], conn["out"]].set(conn["weight"])
            mask = mask.at[conn["in"], conn["out"]].set(1.0)

    return {"weights": W, "mask": mask}

@jax.jit
def forward(arrays, obs):
    W = arrays["weights"] * arrays["mask"]  # zero out disabled connections
    # Fixed shape (MAX_NODES,) throughout — JAX is happy
    activations = jnp.zeros(MAX_NODES)
    activations = activations.at[:obs.shape[0]].set(obs)
    # Simplified: one-pass propagation
    activations = jax.nn.relu(activations @ W)
    return activations[-1]  # output node
```

This actually works. The shapes are static. `jax.jit` compiles once. The mask suppresses disabled connections at runtime without changing the graph structure at compile time.

The cost: you're paying for `MAX_NODES x MAX_NODES` memory for every individual in your population, even the tiny ones. With a population of 500 and `MAX_NODES = 32`, that's 500 × 32 × 32 = 512,000 floats for the weight matrices alone. That's fine. The harder problem is crossover — matching parents by innovation number when the genome is a dense matrix rather than a list of (innovation_number, weight) pairs requires careful bookkeeping, and the natural NEAT crossover logic doesn't map cleanly to array operations.

### 2. Python-side evolution, JAX-side evaluation

Keep genomes as Python dicts entirely. Only convert to JAX for the forward pass evaluation:

```python
def evaluate_population(population, task_obs):
    fitnesses = []
    for genome in population:
        arrays = genome_to_fixed_arrays(genome)  # Python → JAX
        # jit-compiled forward pass, but different for each genome size
        fitness = run_episode(arrays, task_obs)
        fitnesses.append(float(fitness))         # JAX → Python
    return fitnesses
```

This respects both constraints: genomes grow freely in Python, evaluations run on accelerator. The problem is the Python↔JAX boundary. Every `genome_to_fixed_arrays` call is a data transfer. Every `float(fitness)` brings a result back. If your population is 500 and you're running 100 generations, that's 50,000 round trips. The accelerator spends most of its time waiting for Python to hand it the next genome. You're not getting the speedup you came for.

### 3. Fixed topology, variable connection masks

The simplest retreat: give up on node addition entirely. Allow NEAT to evolve *which connections exist* (via enable/disable masks) but not *how many nodes there are*. The topology space is the set of possible edge patterns on a fixed node graph, not the set of possible graphs.

```python
# Fixed graph: input_nodes + max_hidden + output_nodes, all pre-allocated
# Evolution only touches the binary connection mask
def mutate_connection(genome, rng):
    i, j = random.choice(rng, valid_connection_pairs)
    genome["mask"] = genome["mask"].at[i, j].set(
        1.0 - genome["mask"][i, j]  # toggle
    )
    return genome
```

This is JIT-friendly and fast. It's also not really NEAT anymore — the whole point of NEAT is that structure grows from nothing. This is closer to a topology-search version of a fixed network, which is interesting but different.

---

## Current status: honest accounting

The training notebook (`neat_jax.ipynb`) exists. The max-topology padding approach gets far enough to run experiments — populations evolve, fitness improves, slimes learn to jump. The architecture is there.

But there's a known bug that blocks the full NEAT topology evolution loop: **mixing Python lists and JAX arrays in the genome representation causes shape mismatches that break JIT compilation mid-run**. This happens specifically in the crossover step, where child genomes can end up with a connection count that differs from what the JIT-compiled function was traced with, even after padding, because the padding logic has a case where innovation numbers don't align as expected.

This is documented in the README. I have not fixed it. Here's why it's hard:

The real fix isn't a patch to the crossover code. It's a different genome representation altogether. The Python list of `(innovation_number, weight, enabled)` tuples is the natural structure for NEAT, and it's fundamentally at odds with JAX's requirement for fixed-shape arrays. You can work around this with padding and masks, but you're always fighting the representation. The deeper solution is probably something like **learnable topology masks** — where the topology is itself a continuous variable that can be optimized or evolved in a JAX-friendly way — rather than explicit graph mutations over discrete structures.

That's a research problem, not an engineering one. People are working on it. Some relevant directions: differentiable graph neural architecture search, continuous relaxations of discrete graph structures, and the line of work on neural architecture search via gradient methods. None of them are NEAT, but they're trying to solve the adjacent problem: topology as something you optimize rather than enumerate.

---

## What this collision taught me

**JAX's constraints are load-bearing.** The requirement for static shapes isn't an arbitrary restriction — it's what makes XLA compilation possible, which is what makes JAX fast on TPUs. When you fight it, you're not finding a loophole; you're opting out of the thing that makes JAX worth using. The right response is to redesign your representation to fit JAX's model, not to bolt a workaround onto the data structure you already have.

**NEAT was designed for CPUs in 2002.** This isn't a criticism of the algorithm — it's genuinely elegant for what it does. But "add a node" is a Python list append. "Add a connection" is another append. The entire algorithm assumes mutable, variable-length data structures because that's what you have in sequential, single-threaded computation on a CPU. Retrofitting it for hardware-accelerated parallel evaluation is a genuine research problem, not just an engineering challenge.

**The "eager Python, accelerated JAX" split is a real pattern.** It's not unique to NEAT. PyTorch Lightning separates training loop logic (Python) from compute-heavy operations (tensor ops). TensorFlow's `tf.function` has the same static-shape requirement as JAX's `jit`. The architectural pattern of "do the dynamic coordination in Python, do the compute in the accelerated framework" is something you encounter anywhere you're trying to apply modern ML infrastructure to algorithms with dynamic structure. Understanding where that boundary is, and how expensive crossing it is, matters.

**18 GitHub stars.** Some PRs have come in. People are interested in this problem. The tension between evolutionary algorithms (which are inherently dynamic and population-based) and hardware-accelerated ML frameworks (which want fixed, vectorizable computation) is real, and NEAT-JAX hits it about as directly as possible. If you've thought about this problem — if you have a cleaner genome representation, a better way to express topology evolution in JAX, or a pointer to work I should know about — I genuinely want to hear it.

---

## References

- Stanley, K.O. & Miikkulainen, R. (2002). *Evolving Neural Networks through Augmenting Topologies.* Evolutionary Computation, 10(2), 99–127.
- Tang, Y., Tian, Y., & Ha, D. (2022). *EvoJAX: Hardware-Accelerated Neuroevolution.* [arXiv:2202.05008](https://arxiv.org/abs/2202.05008)
- JAX documentation: [Ahead-of-time lowering and compilation](https://jax.readthedocs.io/en/latest/aot.html) — specifically the section on why static shapes are required for `jit`

---

If you know a cleaner way to represent dynamic topologies in JAX, I genuinely want to know. PRs welcome, my inbox is open: [ysrivastava82@gmail.com](mailto:ysrivastava82@gmail.com)

**Note:** If you find this kind of work interesting and your organization does something similar, consider hiring me? I'm on the job market and would love to [chat](mailto:ysrivastava82@gmail.com).
