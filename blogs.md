# Technical Blogs
Some of my learnings about the stuff in tech. Personal, but very helpful for others as well.

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


## Design for Generality - Dated(06/04/2024)
Designing is a complicated process, not a complex process. What I mean particularly is that the latter is analogous to difficult, while the former means it just has a lot of moving parts. The basic task of designer is to reduce the entropy of moving parts and make sense of it, without losing much generality. There are many examples you can think of, I would like to go with Transformers.

### Learning design principles with Transformers

When working on big projects, it is often advised to be as general as possible - I for once disagree to some extent with this statement. I believe you need to be general in limit of infinite exploration, that is, you need to be general both in the starting and end, but in different amounts. These amounts are defined by the scope and requirements of the project.
This is the advantage of working with a cracked team - you get to bear the benefits of the design process and it becomes an inspiration for many. Transformer architecture captures this pretty well.
Residual stream is basically a stream connecting the inputs and outputs of the entire system. Any operation that happens in the entire architecture branches from the residual stream, and "reads" and "writes" to it only. Designing in this way not only reduces the entropy drastically, but also allows to rapidly prototype different parts. 
Most confusion or complications stems from the fact that it is very unclear what that "part" does in the entire system. This gives rise to question such as where does it go? what are the inputs, what are the outputs, what are the connections with different parts et. al. Entropy reduction is not a loss of generality, but rather a tedious process to understand what works together and what not. It fells like a loss of generality, but I can assure that it is basically a lossless compression technique(like a .zip file) , so that we can better attend to the information at hand.
