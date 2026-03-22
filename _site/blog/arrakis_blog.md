Note: Arrakis is a work which is really personal and pivotal for me. I worked on it solo during my free time while simultaneously working a co-op and uni. Buildspace was really pivotal in providing the push.

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
`HookedAutoModel` is a wrapper around the HUggingface PreTrainedModel, and the only difference between them is a single decorator on the forward function. Everything just works out of the box - and the functionality can be removed without affeccting the model just by removing the decorator. First, create a `HookedConfig` for the model you want to support with the required parameters. Then, create a `HookedAutoModel` from the config. As of now, these models are supported - gpt2, gpt-neo, gpt-neox, llama,gemma,phi3,qwen2, mistral, stable-lm

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

`IntepretabilityBench` is the workspace where researchers can conduct experiments. At it’s core, the whole purpose of Arrakis is to conduct MI experiment, so it is made keeping accessibility in mind. Just derive from the `BaseInterpretabilityBench` and instantiate an object(`exp` in this case). This object provides a lot of function out-of the box based on the “tool” you want to use for the experiment, and have access to the functions that the tool provides. You can also create your own tool(read about that [here](https://arrakis-mi.readthedocs.io/en/latest/README.html#extending-arrakis) )

```python
from arrakis.src.core_arrakis.base_bench import BaseInterpretabilityBench

class MIExperiment(BaseInterpretabilityBench):
    def __init__(self, model, save_dir="experiments"):
        super().__init__(model, save_dir)
        self.tools.update({"custom": CustomFunction(model)})

exp = MIExperiment(model)
```

Apart from access to MI tools, the object also provides you a convinient way to log your experiments. To log your experiments, just decorate the function you are working with `@exp.log_experiment`, and that is pretty much it. The function creates a local version control on the contents of the function, and stores it locally. You can run many things in parallel, and the version control helps you keep track of it.
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

# Apart from these tools, there are also `@exp.profile_model`(to profile how # much resources the model is using) and `@exp.test_hypothesis`(to test hypothesis). Support of more tools will be added as I get more feedback from the community.
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
Generating plots is Arrakis is also plug and play, just add the decorator and plots are generated by default. Read more about the graphing docs [here](https://arrakis-mi.readthedocs.io/en/latest/README.html#)

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

These are three upper level classes in Arrakis. One is the `InterpretabilityBench` where you conduct experiments, the second is the `core_arrakis` where I’ve implemented some common tests for Transformer based model and the third is the `Graphing`.

### Resources
- Transformers Circuits Thread - https://transformer-circuits.pub 
