Note: This is a life lesson which in someway or other you'll learn. I'm pretty sure.

## Design for Generality - Dated(06/04/2024)
Designing is a complicated process, not a complex process. What I mean particularly is that the latter is analogous to difficult, while the former means it just has a lot of moving parts. The basic task of designer is to reduce the entropy of moving parts and make sense of it, without losing much generality. There are many examples you can think of, I would like to go with Transformers.

### Learning design principles with Transformers

When working on big projects, it is often advised to be as general as possible - I for once disagree to some extent with this statement. I believe you need to be general in limit of infinite exploration, that is, you need to be general both in the starting and end, but in different amounts. These amounts are defined by the scope and requirements of the project.
This is the advantage of working with a cracked team - you get to bear the benefits of the design process and it becomes an inspiration for many. Transformer architecture captures this pretty well.
Residual stream is basically a stream connecting the inputs and outputs of the entire system. Any operation that happens in the entire architecture branches from the residual stream, and "reads" and "writes" to it only. Designing in this way not only reduces the entropy drastically, but also allows to rapidly prototype different parts. 
Most confusion or complications stems from the fact that it is very unclear what that "part" does in the entire system. This gives rise to question such as where does it go? what are the inputs, what are the outputs, what are the connections with different parts et. al. Entropy reduction is not a loss of generality, but rather a tedious process to understand what works together and what not. It fells like a loss of generality, but I can assure that it is basically a lossless compression technique(like a .zip file) , so that we can better attend to the information at hand.
