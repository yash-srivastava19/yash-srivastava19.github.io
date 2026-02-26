Note: I was watching Karpathy's minBPE video for the third time, late at night. The question hit me mid-rewatch: what if the corpus you feed BPE isn't plaintext? I thought it'd be a quick experiment. It became a PyPI package. Karpathy is responsible for a lot of my late nights.

# What Happens When You Run BPE on Encrypted Text?

Repo: [yash-srivastava19/sec_bpe](https://github.com/yash-srivastava19/sec_bpe)

## The Origin Story

I was going through Karpathy's [minBPE video](https://www.youtube.com/watch?v=zduSFxRajkE) for probably the third time - because that's the kind of person I am - and something clicked differently this time. Not about BPE itself, but about the *corpus* being fed into it. Everyone just... takes the plaintext corpus for granted. You have text, you run BPE, you get a vocabulary. But what if the text wasn't text?

Around the same time, I was reading the [Random BPE paper](https://arxiv.org/abs/2311.01480), which argues that BPE's greedy merge order isn't the sacred, irreplaceable thing we treat it as. You can randomize the merge decisions and still get vocabularies that perform comparably. That planted a seed: if the *order* of merges doesn't matter that much, what about the *distribution* of the corpus itself? BPE is fundamentally just a frequency analysis over byte pairs. What if we messed with the byte distribution before handing it off?

Then I was reading about classical ciphers for no particular reason (as one does), and I landed on the Playfair cipher. Playfair operates on *digraphs* - letter pairs. It encrypts two letters at a time, together. And I thought: BPE also operates on pairs. It's literally called *Byte Pair* Encoding. It merges the most frequent *pairs* of bytes. What if those pairs were ciphertext pairs?

That's the whole idea. Encrypt first, then BPE.

## BPE in 30 Seconds

If you don't know BPE, watch [Karpathy's video](https://www.youtube.com/watch?v=zduSFxRajkE) first. Seriously. I'm not going to do it better justice than he does.

The short version: BPE starts with a vocabulary of individual bytes (or characters). It then iteratively finds the most frequent adjacent pair in the corpus and merges them into a new token. Repeat until you hit your target vocabulary size. The result is a vocabulary that naturally captures common subwords - "ing", "er", "the", "un" - because those sequences appear together so frequently that they get merged early.

The merge *order* matters because high-frequency pairs get merged first, and those merged tokens then become candidates for further merges. You end up with a vocabulary heavily biased toward the statistical regularities of your training language.

That bias is exactly what I wanted to disrupt.

## Why ChaCha20 Specifically?

The obvious question when you hear "encrypt then BPE" is: why not just use AES? Or any block cipher?

Block ciphers have fixed block sizes. AES operates on 128-bit (16-byte) blocks. If your plaintext isn't a multiple of 16 bytes, you need padding - which means the ciphertext is *longer* than the plaintext. BPE relies on the ciphertext being a faithful byte-for-byte stand-in for the original. Padding breaks that assumption completely.

Stream ciphers solve this elegantly. A stream cipher generates a keystream and XORs it byte-by-byte with the plaintext. Output length equals input length. Always. No padding, no length mismatch.

ChaCha20 is the stream cipher to use in 2024 for a few reasons:

- **Length-preserving by design.** Every byte of plaintext maps to exactly one byte of ciphertext.
- **Fast in software.** AES is fast *when you have hardware AES-NI instructions* (most modern CPUs do). But ChaCha20 is faster in pure software implementations - no special hardware required. This matters if you're running on VMs or embedded environments.
- **Modern and battle-tested.** ChaCha20 is used in TLS 1.3. It's not a toy cipher. It was designed by Daniel Bernstein and has survived serious cryptanalysis.
- **Semantically secure.** Given a random nonce and key, the output is computationally indistinguishable from random bytes. This is the property we actually care about for this experiment.

That last point is key. ChaCha20 doesn't just obscure the text - it makes the output *look like uniform random noise*. Which has interesting implications for BPE, as we'll get to.

## The Core Algorithm

The full SecBPE pipeline looks like this:

```python
from Crypto.Cipher import ChaCha20
import os

def encrypt_corpus(plaintext: bytes, key: bytes, nonce: bytes) -> bytes:
    cipher = ChaCha20.new(key=key, nonce=nonce)
    return cipher.encrypt(plaintext)

def sec_bpe_train(plaintext_corpus: bytes, vocab_size: int, key: bytes, nonce: bytes):
    # Step 1: Encrypt the corpus
    proxy_corpus = encrypt_corpus(plaintext_corpus, key, nonce)

    # Step 2: Run standard BPE on the ciphertext
    # proxy_corpus is the same length as plaintext_corpus
    # BPE sees random-looking bytes and builds a vocabulary from them
    merges, vocab = run_bpe(proxy_corpus, vocab_size)

    # The resulting vocab contains ciphertext tokens
    # Without the key, they mean nothing
    return merges, vocab, key, nonce

def tokenize(text: bytes, merges: dict, vocab: dict, key: bytes, nonce: bytes):
    # At inference time: encrypt the input first
    encrypted_input = encrypt_corpus(text, key, nonce)
    # Then tokenize using the proxy vocabulary
    return apply_merges(encrypted_input, merges)
```

The key insight: the plaintext vocabulary is never materialized anywhere in the model or its artifacts. The vocabulary tokens are ciphertext blobs. If you ship this model to someone without the key, they cannot reconstruct what subwords the model actually learned. The vocabulary is a map of encrypted patterns.

Decoding works the same way in reverse - model output tokens are detokenized using the proxy vocabulary, then decrypted with the key to recover plaintext.

Install it from PyPI:

```bash
pip install sec-bpe
```

## The Counterintuitive Hypothesis

Here's where it gets interesting. This isn't just "encryption for privacy." There's a hypothesis baked in about tokenization quality.

ChaCha20 produces output that is statistically close to uniform random. In standard BPE, common English subwords completely dominate the early merges. The word "the" gets merged very early. "ing", "er", "un", "re" - all of these high-frequency patterns get captured fast, while rare byte sequences almost never appear in the final vocabulary.

The result is a vocabulary that's heavily skewed toward a small set of very common patterns. If you feed the model text that doesn't look like its training distribution - code, mathematical notation, names in non-English scripts - the tokenizer has to work harder, splitting things into more tokens than necessary because the rare patterns were never merged.

Encrypted BPE changes this. When byte pairs are uniformly distributed, there's no dominant pair to merge first. The frequency distribution over pairs is flat. This means:

- Merges happen more *evenly* across the byte space
- Rare byte sequences get a fighting chance of being merged
- The vocabulary covers the byte space more uniformly

Think of it like packing a suitcase. Standard BPE on English is like packing a suitcase with unevenly sized clothes - you end up with big bulky sweaters taking up most of the space, and socks stuffed into gaps. BPE on encrypted text is like packing uniformly sized blocks - they stack more evenly, with less wasted space.

Whether this actually produces a *better* tokenizer is an empirical question. But the entropy argument suggests it could produce a more robust one for out-of-distribution inputs, since the vocabulary doesn't have the same English-biased skew.

I ran experiments comparing standard BPE vs SecBPE on vocabulary overlap, merge frequency distributions, and tokenization of held-out text. The merge frequency histogram for SecBPE is noticeably flatter - fewer "superstar" tokens that appear hundreds of times more often than the rest. Whether that translates to downstream model quality improvement is still an open question.

## Honest Assessment of the Tradeoffs

I'm not going to pretend this is a complete solution to anything. Here are the real problems:

**Computational overhead.** Every inference pass requires an encryption step before tokenization. For batch inference at scale, this adds up. ChaCha20 is fast, but it's not free.

**Key management.** The model is completely useless without the key. If you lose the key, you cannot tokenize new inputs or decode outputs. Key rotation also becomes a problem - you'd need to retrain the tokenizer with the new key. This is a real operational burden.

**Interpretability is gone.** This is the big one. The tokens are ciphertext strings. You cannot look at your vocabulary and understand what the model learned. Debugging why a model is producing bad outputs for certain inputs becomes much harder when you can't inspect what the tokenizer is doing. Probing experiments, vocabulary analysis, all the standard interpretability tooling - it doesn't work anymore without decrypting first.

One partial workaround I'm thinking about: a semantic-preserving hashing approach, where instead of encryption, you apply a deterministic hash that maps semantically similar subwords to similar codes. This would preserve some interpretability (you can still compare token distances) while obscuring the actual vocabulary. It's a different threat model, but it partially addresses the interpretability problem. That's for a future experiment.

**The security model needs scrutiny.** ChaCha20 is semantically secure, but if an adversary can see many (plaintext, ciphertext) pairs going through your tokenizer, they might learn something about the vocabulary mapping. The security guarantees hold under standard cryptographic assumptions, but applying crypto in an ML pipeline creates new attack surfaces that haven't been thoroughly analyzed.

## References

- Karpathy's minBPE: [youtube.com/watch?v=zduSFxRajkE](https://www.youtube.com/watch?v=zduSFxRajkE) and [github.com/karpathy/minbpe](https://github.com/karpathy/minbpe)
- Random BPE paper: [arxiv.org/abs/2311.01480](https://arxiv.org/abs/2311.01480)
- ChaCha20 spec: [RFC 8439](https://datatracker.ietf.org/doc/html/rfc8439)

## What's Next

This is a WIP. There's a list of experiments I want to run:

- Downstream LM performance comparison (standard vocab vs proxy vocab on identical architectures)
- SecBPE on multilingual corpora - does uniform byte distribution help cross-lingual tokenization?
- The semantic-preserving hash alternative
- Formal analysis of the information-theoretic security properties

If this sounds interesting to you, raise a PR or [mail me](mailto:ysrivastava82@gmail.com). The more eyes on this, the better.
