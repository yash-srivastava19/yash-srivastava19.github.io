Note: Mechanistic interpretability is getting genuinely out of hand, and I mean that in the best possible way. Every few months someone invents a new tool for peering inside neural networks, and the toolbox is now large enough that you can start combining them in weird ways. This post is about one such combination. It is a proof of concept, not a breakthrough. But I think the methodology is interesting enough to write up. If you work on interpretability, I'd love to [chat](mailto:ysrivastava82@gmail.com).

# Using MCTS to Navigate Sparse Autoencoder Feature Space

- **Project Home:** [Github Repo](https://github.com/yash-srivastava19/deeprobe)

---

## The Idea That Eventually Made Sense

Deeprobe did not start as "MCTS on SAE features." It started as a vague conviction that search algorithms and representation learning should be combinable in some interesting way, and then spent a while being wrong about exactly how.

The thinking evolved roughly like this:

**Step 1: LLMs + MCTS.** The initial intuition was broad — LLMs have a huge output space, MCTS is good at navigating huge spaces, so maybe you can use MCTS to steer generation. Too broad. This is basically just beam search with extra steps, and there is already a lot of work in that direction.

**Step 2: GANs + MCTS.** GANs have a latent space. MCTS explores spaces. Could you use MCTS to navigate the GAN latent space toward some target? The idea is not crazy but it is vague in a way that makes it hard to pin down what "reward" even means. Abandoned.

**Step 3: A paper about molecules.** I came across work on generating molecules using autoencoders combined with tree search — the intuition being that a compressed, continuous latent space is much easier for a search algorithm to simulate over than the raw combinatorial space of molecular graphs. The autoencoder does the compression; the search does the navigation. The pieces clicked.

**Step 4: The inversion.** The molecule paper uses a *compressed* latent space — smaller, denser, more tractable. But what about going the other direction? Sparse Autoencoders in the mechanistic interpretability world do the opposite: they *expand* a dense activation space into a sparser, larger, more interpretable one. GPT-2 has 768 neurons per layer. A SAE trained on that layer might have 16,000+ features. You trade compactness for interpretability.

The question that made Deeprobe make sense: **can you use MCTS to navigate that expanded space and find which features actually matter for a specific task?**

---

## Some Background You Need

### Why Neurons Are Annoying

Neural network neurons are polysemantic. A single neuron in GPT-2 activates for "the word 'cat'", "something about databases", "tokens near commas", and seventeen other things simultaneously. This is not a bug — it is how the model fits more information than it has neurons — but it makes interpretability hard, because you cannot point at a neuron and say "this neuron represents X."

Sparse Autoencoders were developed (by Anthropic and others) as a way to decompose these polysemantic activations into a larger set of near-monosemantic features. The core idea: train an autoencoder with a sparsity penalty on the hidden layer, and it learns to represent each activation as a sparse combination of a larger dictionary of features. Each feature tends to correspond to one coherent concept.

The cost is the expanded space. 768 dimensions becomes 16,000+. The space is much more interpretable but also much larger.

### The IOI Task

Indirect Object Identification (IOI) is one of the classic benchmarks in mechanistic interpretability. The setup:

> "When John and Mary went to the store, John gave a drink to ___"

A well-functioning language model should complete this with "Mary." It needs to track that "Mary" is the indirect object — the one who was not already mentioned as the subject of the second clause. Wang et al. (2022) identified the specific circuit of attention heads in GPT-2 that implements this behavior. It is one of the clearest examples we have of a human-interpretable algorithm being executed inside a transformer.

Deeprobe's question: can we find the same signal — the fact that "Mary" is the relevant entity — by probing SAE features with MCTS, rather than by doing attention head analysis?

---

## Why MCTS Instead of Gradients

The obvious alternative to MCTS for finding important features is gradient-based attribution: compute how much each feature's activation contributes to the output, take the gradient, done. So why bother with MCTS?

A few reasons.

**Gradients are local.** Gradient-based attribution tells you how much a small perturbation to a feature changes the output. It does not tell you anything about interactions between features, or about features that matter in combination but not individually. MCTS, by maintaining a tree of states, can reason about sequences of feature combinations. It has lookahead.

**Gradients assume a smooth landscape.** In a sparse feature space with 16,000+ dimensions, the landscape can be extremely non-smooth. Features interact in complex, non-linear ways. A local gradient might point in a direction that looks promising but leads nowhere globally.

**MCTS explores.** The UCB1 exploration bonus in MCTS explicitly encourages trying things that have not been tried much yet. In a large feature space, this matters — there is a real risk that greedy attribution methods fixate on obvious features and miss important ones that only activate in specific contexts.

The tradeoff is that MCTS is expensive. Every node expansion requires evaluating the reward function, and the branching factor in a 16,000-feature space is enormous. This is the central tension of Deeprobe, and I will come back to it.

---

## How It Works

Think of it this way. Imagine you are trying to find a specific book in a library. The original library has 768 shelves — dense, polysemantic, each shelf holding books from completely different genres. Someone then reorganizes the library: now it has 16,000 shelves, each dedicated to one specific topic. The organization is much better. But the library is also much bigger, and you do not have a complete catalog. MCTS is the search strategy you use to navigate it. It helps. But the search space has gotten much harder to cover exhaustively.

Here is how the MCTS loop works over SAE features:

```python
# Pseudocode: MCTS over SAE feature space for IOI

# State: set of currently active SAE features
# Actions: probe a specific token's features (John, Mary, or control)
# Reward: cosine similarity of probed feature activation to "correct" answer (Mary)

def mcts_sae(sae_activations, n_iterations):
    root = Node(state=initial_feature_set(sae_activations))

    for _ in range(n_iterations):
        # 1. Selection: walk the tree using UCB1
        node = select(root)  # UCB1: score = Q/N + c * sqrt(log(N_parent) / N)

        # 2. Expansion: if node is not terminal, expand one child
        if not node.is_terminal():
            node = expand(node, sae_activations)

        # 3. Simulation: rollout from expanded node to estimate value
        reward = simulate(node, sae_activations)
        # reward = cosine_similarity(probed_feature_activation, mary_token_embedding)

        # 4. Backpropagation: update visit counts and Q-values up the tree
        # (This is MCTS tree backprop — NOT gradient backprop through the SAE)
        backpropagate(node, reward)

    # Return the action (feature probe) with highest visit count from root
    return best_action(root)
```

The key distinction worth emphasizing: "backpropagation" here is MCTS tree backpropagation — updating visit counts and average rewards up the path from the evaluated node to the root. This has nothing to do with gradient backpropagation through the neural network. The SAE weights are frozen throughout; we are using MCTS as a search policy over the feature space, not training anything.

The state at each node is the current set of active features. The actions are which token's features to probe — in the IOI task, the natural actions are: probe features associated with "John" (the repeated name), probe features associated with "Mary" (the indirect object), or probe features associated with a control token. The reward is a similarity score: how well does the activated feature align with the indirect object, "Mary"?

---

## What Actually Happened

Getting this working required going through TransformerLens and SAELens and wiring up the pieces. The basic pipeline:

1. Load GPT-2-small via TransformerLens
2. Run the IOI template through the model, capture activations at target layers
3. Pass activations through the SAE (from SAELens) to get feature representations
4. Run MCTS over the feature space using the similarity reward
5. Inspect which features MCTS converges on

There was a bug in `tokenize_and_concatenate` in TransformerLens that caused problems with small datasets — it makes assumptions about minimum dataset size that break when you are working with a handful of IOI template strings. I patched it locally to get things moving.

**The honest results:** Basic MCTS on SAE activations does work for the IOI task, in the sense that it identifies features that correlate with the indirect object. Features that fire strongly for "Mary" in the IOI context do bubble up through the search. This is encouraging.

But there are real caveats.

**The combinatorial explosion is serious.** With 16,000+ features and a meaningful branching factor at each node, MCTS cannot go very deep in any reasonable number of iterations. The coverage is thin. You are sampling a small portion of the space and hoping you got lucky with the features you happened to explore. This is not inherently fatal — MCTS is designed to handle large branching factors — but it means the results are noisy and the confidence in any particular feature is limited.

**The reward function is too simple.** Cosine similarity of a feature activation to a token embedding is a reasonable proxy but it is noisy. Features that fire for many different things will score well on the similarity metric even if they are not specifically tracking indirect objects. The MCTS tree ends up spending exploration budget on features that look good superficially but are not specifically about the IOI task. This is where the current approach falls short most badly.

---

## The Reward Modeling Problem

In retrospect, this is the interesting unsolved problem. MCTS is a well-understood algorithm. SAEs are a mature tool. The part that is genuinely hard is specifying what "finding the right feature" means in a way that is precise enough for MCTS to navigate toward.

In the molecule generation work that inspired this, the reward is relatively clear: does this molecule have the desired properties? You can score a molecule. In IOI, the ground truth from Wang et al. is at the level of attention heads — specific heads at specific layers implement the indirect object tracking. Translating that into a feature-level reward signal that MCTS can use effectively is not obvious.

Some directions that might be worth exploring:

**Causal interventions as reward.** Rather than using similarity as the reward, run an activation patching experiment: replace the feature activation with a counterfactual and see how much the model's output changes. This is more grounded but much more expensive to compute per node.

**Contrastive reward.** Score features not on their absolute similarity to "Mary" but on the *difference* in their activation between IOI inputs where "Mary" is correct versus inputs where a different name is correct. This should be more specific.

**Learned reward models.** Train a small classifier on known IOI-relevant and IOI-irrelevant features, then use that as the MCTS reward function. This requires labeled data but could be much more precise.

The methodology of MCTS over SAE features is sound in principle. But it is only as good as the reward signal you give it, and that is where the real work is.

---

## Tools and References

This experiment used:

- **TransformerLens** — for model loading, hook points, and activation capture
- **SAELens** — for pre-trained SAEs on GPT-2
- Wang, K., Variengien, A., et al. (2022). *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small.* The benchmark that defined the IOI task.
- Segler, M. H. S., et al. (2018). *Planning Chemical Syntheses With Deep Neural Networks and Symbolic AI.* The inspiration for combining latent space representations with tree search.

---

## What I Think Is True

Deeprobe works as a proof of concept. The idea of using MCTS to navigate SAE feature space is coherent and implementable. The IOI task is a reasonable test bed. The results are encouraging without being conclusive.

The interesting part is not "MCTS found the indirect object features" — the interesting part is that this is a new class of interpretability method that does not require gradient access, operates globally rather than locally, and scales to the expanded feature spaces that SAEs produce. Whether it scales *well* is a different question, and the answer right now is: not yet, because the reward modeling is too crude.

If you want to look at the code, it is at [https://github.com/yash-srivastava19/deeprobe](https://github.com/yash-srivastava19/deeprobe). It is research code — proceed accordingly.

---

I think the reward modeling problem is where the real work is. If you have ideas on better reward signals for interpretability tasks, I want to talk. Reach me at [ysrivastava82@gmail.com](mailto:ysrivastava82@gmail.com).
