Note: I grew up switching between Hindi and English mid-sentence without thinking about it. One afternoon I started wondering if GPT-2, trained only on English, had any idea what was happening when you dropped a Hindi word in. Two days later I had a GitHub repo. Worth it.

# Do LLMs Have a Secret Interlingua? Testing GPT-2's Hinglish Comprehension with Sparse Autoencoders

- **Project Home:** [Github Repo](https://github.com/yash-srivastava19/sae-macaronic-analysis)

---

## The Question That Wouldn't Leave Me Alone

Something I've always been curious about: when bilingual speakers code-switch - dropping Hindi words mid-English sentence, the way you do when you're talking to your family - does a model like GPT-2 have *any* idea what's happening semantically? Or does it just see an unknown token and shrug?

The specific form of this question that grabbed me was: **does GPT-2 fire the same internal features for "gaana" (Hindi for "song") as it does for "song," when both appear in otherwise identical sentences?**

If the answer is yes - even partially - that's genuinely surprising. GPT-2-small was not trained on Hindi. It was trained on English web text. There is no reason it *should* generalize. But language models are strange things, and I've learned to not assume I know what they know.

This is a mechanistic interpretability experiment. The method: Sparse Autoencoders (SAEs). The dataset: 8 matched English/Hinglish sentence pairs, differing by exactly one word. The result: partial, inconclusive, and worth talking about.

---

## Some Background: What Are SAEs?

If you've heard of polysemantic neurons, you know the problem. A single neuron in a language model might activate for "banana," "yellow things," "tropical fruits," and "currency slang" all at once. The neuron isn't confused - it's just overloaded. Models are smaller than the conceptual space they need to represent, so features share neurons through a principle called *superposition*.

**Sparse Autoencoders** are a technique to untangle this. Think of them as a decoder that forces the model to "show its work." You take the activations from some internal layer of the model, project them into a much higher-dimensional space, impose a sparsity constraint (most features should be zero for any given input), and then reconstruct the original activations. What you're left with is a set of *monosemantic* features - each one tends to correspond to a single, interpretable concept.

The analogy I like: imagine a crowded room where everyone is whispering different things simultaneously (polysemantic neurons). An SAE is like a microphone array that can isolate each voice (monosemantic features). You can now ask: which voices are active, and how loudly?

For this experiment, I used two SAEs from the [SAELens](https://github.com/jbloomAus/SAELens) library:
- **JBloom's residual stream SAE** for GPT-2-small
- **Tom McGrath's MLP SAE** for GPT-2-small

Both are trained on GPT-2-small's internal activations. The goal: for each sentence pair, look at which SAE features fire on the English sentence versus the Hinglish sentence, and see how much they overlap.

---

## Why Residual Stream and MLP? (And Not Attention?)

A brief digression that I think is worth making explicit.

Mechanistic interpretability has a rich tradition of studying attention heads - the [Indirect Object Identification (IOI)](https://arxiv.org/abs/2211.00593) paper is a beautiful example of this, showing that specific attention heads in GPT-2 implement a precise syntactic operation (copying the indirect object token). That work targets *structural* computation: which token attends to which other token.

My task is different. I'm not asking about token-level manipulation or syntactic structure. I'm asking about **semantic meaning**: does the model represent "song" and "gaana" similarly? That kind of question lives in the residual stream and MLP layers, which are where semantic content accumulates and gets transformed. The residual stream at a given layer is a running summary of "what does the model know about this sequence so far." MLPs are where a lot of the key-value memory lives - they look up facts and store associations.

So I deliberately restricted my investigation to those two components. If there's a cross-lingual semantic circuit in GPT-2, I'd expect to find traces of it there.

---

## The Experiment

### The Prompt Pairs

Eight sentence pairs. Each pair is identical except one word is replaced with its Hindi equivalent. Here they are:

| # | English Prompt | Hinglish Prompt | Swapped Word |
|---|----------------|-----------------|--------------|
| 1 | She loves to sing Bollywood **songs** | She loves to sing Bollywood **gaane** | songs → gaane |
| 2 | He prefers to eat spicy Indian **food** | He prefers to eat spicy Indian **khana** | food → khana |
| 3 | She enjoys shopping for traditional **dresses** | She enjoys shopping for traditional **kapde** | dresses → kapde |
| 4 | He wants to travel to exotic **locations** | He wants to travel to exotic **jagah** | locations → jagah |
| 5 | She enjoys reading mystery **novels** | She enjoys reading mystery **kahaaniyan** | novels → kahaaniyan |
| 6 | They are planning a trip to the **mountains** | They are planning a trip to **pahad** | mountains → pahad |
| 7 | He is a fan of Bollywood **movies** | He is a fan of Bollywood **filme** | movies → filme |
| 8 | They are attending a **wedding** next month | They are attending a **shaadi** next month | wedding → shaadi |

The prompts are all culturally consistent - they're about things Hinglish speakers would naturally discuss. The Hindi words are common, high-frequency terms. If GPT-2 has any cross-lingual generalization at all, these pairs give it the best chance to show up.

### What I Measured

For each prompt, I ran the sentence through GPT-2-small, hooked into the residual stream and MLP activations at each layer, and passed those activations through the SAE decoder. The SAE outputs a set of (feature index, activation value) pairs - the features that "fired" for that input.

I then compared:
- Which features fired for the English prompt
- Which features fired for the Hinglish prompt
- Whether the same features fire at similar magnitudes (feature overlap)

The visualization files are named to make this tractable. A file named `blocks.2.hook_resid_pre_9_eng_2_vis.html` means: layer 2, residual stream hook, top-9 features, English version of prompt 2. The `hing` variants give you the same but for Hinglish. You can see them all on the [repo](https://github.com/yash-srivastava19/sae-macaronic-analysis) or via the Dropbox links there.

---

## What I Found

**The honest answer: partial evidence, not a clean result.**

Some layers show overlapping feature activations for matched word pairs - the same SAE features light up for "songs" and "gaane." Others show almost no overlap. The pattern is not uniform across all prompt pairs or all layers.

A few things that stood out:

**The residual stream tends to preserve more semantic overlap than MLP layers.** This makes a certain kind of sense. The residual stream accumulates a broad, distributed representation of meaning. MLPs are more "lookup table"-like - they're accessing stored associations, and those associations were learned from English training data. If the residual stream is catching something general about the token's context, MLP layers would be less likely to generalize to tokens they've never explicitly seen.

**Some Hindi words did better than others.** "Shaadi" (wedding) and "khana" (food) showed more feature overlap in certain layers than "kahaaniyan" (novels) or "jagah" (locations). My guess: "shaadi" and "khana" appear in some form in English text - as loanwords, in cultural context, in food blogs. GPT-2 may have seen them. "Kahaaniyan" is a more morphologically complex Hindi form (it's a plural) and almost certainly never appeared in GPT-2's training corpus.

**The overlap is never perfect.** Even in the most favorable cases, there are features that fire for one but not the other. This is expected - the tokens are phonologically and orthographically completely different. The question was whether *any* semantic signal bleeds through, and in some cases, it does appear to.

---

## What This Might Mean (And What It Definitely Doesn't Mean)

The **interlingua hypothesis** is a concept from classic NLP translation research: the idea that a translation system might develop an intermediate, language-neutral representation - a kind of mental Esperanto that both source and target languages pass through. Early machine translation researchers hoped to build this explicitly. Modern neural MT systems seem to develop something like it implicitly, at least partially.

The question I was probing: does GPT-2, trained monolingually, develop *anything like this* for Hindi words embedded in English context?

The partial overlap I observed is consistent with a weak version of this. The model isn't doing nothing with "gaane" - it's not just seeing a random out-of-vocabulary string. Some features that encode "song-like" context are firing. But it's not robust, and it's not uniform.

**What this definitely doesn't mean:** GPT-2 "understands Hindi." It doesn't. It might be picking up on contextual cues - "Bollywood" right before the Hindi word, the general semantic frame of the sentence - rather than the Hindi word itself. The SAE features that overlap could be encoding the *context* (Bollywood-related content) rather than the *meaning of the Hindi word*. I can't cleanly separate these in my current setup.

---

## Limitations (And I'm Being Genuinely Honest Here)

I want to be upfront about what this experiment is and isn't:

1. **GPT-2 is a tiny model.** GPT-2-small has 117M parameters. Modern LLMs are orders of magnitude larger. Whatever cross-lingual generalization exists in GPT-2 is a lower bound on what might exist in bigger models, not an upper bound.

2. **8 prompts is a very small sample.** I can describe patterns, but I cannot do statistical significance testing on 8 data points. This is exploratory.

3. **No baseline for "random overlap."** I didn't measure how much feature overlap you'd expect from two completely unrelated English word substitutions (e.g., swapping "songs" for "books"). Without that baseline, I can't rigorously claim the Hindi-English overlap is above chance.

4. **SAE quality matters.** The SAEs I used are trained on GPT-2-small and are good, but they're not perfect decompositions. Some features may be poorly defined or conflate multiple concepts.

5. **Contextual confounding.** As I mentioned above - the model might be responding to context ("Bollywood") rather than the Hindi word itself. This is the hardest confound to rule out without ablation experiments.

This isn't a paper. It's an experiment I ran because the question was interesting and I wanted to see what would happen. I think the result is suggestive enough to be worth writing up, even if it doesn't settle anything.

---

## What I'd Do Next

If I were to take this further (and I'm genuinely tempted):

- **Run ablations**: mask out the surrounding context and see if feature overlap disappears. If "Bollywood" is doing all the work, removing it should kill the overlap.
- **Try larger models**: GPT-2 is a worst-case scenario for cross-lingual generalization. A model that has seen more multilingual data would be far more interesting.
- **More prompt pairs**: 8 is not enough. You'd want at least 50-100 pairs to make statistical claims.
- **Controlled vocabulary**: select Hindi words that definitely *don't* appear in English text, to rule out the possibility that GPT-2 has simply seen them.
- **Compare with a multilingual model**: run the same experiment on mBERT or BLOOM and see if the feature overlap is qualitatively stronger.

The machinery for all of this exists. SAELens makes it genuinely accessible. The main bottleneck is time.

---

## Closing Thoughts

I started this project because I was nerd-sniped by a simple question, and I ended it having learned a lot more about mechanistic interpretability tooling, macaronic language, and the weird edge cases of what monolingual models do or don't generalize.

The result is "partial evidence, inconclusive" - which is an honest result. GPT-2 wasn't trained on Hindi. The fact that *anything* interesting happens when you feed it Hinglish is, to me, worth noting. It suggests that semantic representations in these models are more contextually grounded and less token-specific than a naive view would suggest.

If you work on multilingual interpretability - especially SAE-based approaches to cross-lingual semantics - I'd genuinely love to collaborate or just compare notes. Hit me up at [ysrivastava82@gmail.com](mailto:ysrivastava82@gmail.com). And if your organization does this kind of research, consider that I'm on the job market and would love to [chat](mailto:ysrivastava82@gmail.com).

The full code, SAE feature visualizations, and Dropbox links for all HTML charts are at the [GitHub repo](https://github.com/yash-srivastava19/sae-macaronic-analysis). Go poke around.
