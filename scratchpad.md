# Scratchpad

This is my scratchpad, where I'll try to include stuff that I'm currently working on and why. This might serve as a log for the initial days of the project. If you want to work together on any of this, or have ideas to look into, please hit me up at my [mail](mailto:ysrivastava82@gmail.com)

## SDPA Optimization Package.
I've been looking into ways to optimize SDPA with less memory overhead. I'm trying to understand how can we use less space in the KV Cache of SDPA to boost performance. For decoder only models, KV cache can give faster matrix multiplication times, but the ever growing size for KV cache is a problem. I've identified 3 axes on which we can reduce the memory footprint of models calculating SDPA, which are: precision, batch_size/seq_len, and KV_cache size. I think optimizing SDPA on these 3 axes and finding the Goldilocks zone is a good starting point. For starting, I'm running benchmarking tests with fused attention functions and quantization(and testing with Flex Attention too). The idea is to make a all bateries included pytorch package that optimizes SDPA. I need to understand the tradeoffs between these techniques, but I need some sanity check on whether I'm going in the right direction. If you have any ideas, hit me up.

![image](https://github.com/user-attachments/assets/e73f7cd3-8114-45c8-a8b4-b7fde28ed837)

## GNUMake, but with a lot of batteries included
I've been thinking about this project from a long time. I'm a big advocate of using tools to speedup your workflow, and the closest something that has come to it is GNUMake. What I'm looking for is also similar, but with a lot of additional features to enhance my workflow(which I think is more data centric). I actually want to make a tool that is more general purpose, and many other people can benefit from it. I've been working on it in my free time, and as on now, I think it is better to build something on top of Luigi, which actually solves a lot of my problems.

## Handshake, but opensource
I feel many application could benefit if they were built in public. During my time at my undergraduate, my college didn't have a career portal which actually gave me a lot of information overload. I think it is the responsibility of students to build anything that their uni doesn't have, but no one took initiative(and nor did I). I want to have a portal where students and potential employers can share information without too much overhead. For the initial version, it should have alumni connect page, calendar, and a way to upload and score resume. I don't think I have time to devote to this project, but if someone is interested, they can build it.

## Monthly Infinite Media Canvas
I'll soon graduate and the idea to connect with my friends even when they are away from is something I want to invest in. I've been thinking of an application, where a group gets a collaborative infinte canvas for each month, and they can put text, audio, video, images and anything. They have a month to collect all their memories(and what is going on in their life), and in this way, you can catch up to them without having to worry about spending time on "yet another social media site". I think I can spend some time on it, but I don't know where to look for inspiration.

## LSP
I just want to build a LSP and see it in action.
