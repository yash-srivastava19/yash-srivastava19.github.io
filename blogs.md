# Technical Blogs
Some of my learnings about the stuff in tech. Personal, but very helpful for others as well.

## Arrakis - A toolkit to conduct, track and visualize mechanistic interpretability experiments.

**Project Name :** Arrakis.
**Project Description :** Arrakis is a library to conduct, track and visualize mechanistic interpretability experiments.
**Project Home :** https://github.com/yash-srivastava19/arrakis
**Project Documentation :** https://arrakis-mi.readthedocs.io/en/latest/README.html
**PyPI Home :** https://pypi.org/project/arrakis-mi/
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
