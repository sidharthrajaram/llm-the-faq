# The FAQs of LLM Training
2/27/2025 - Sidharth Rajaram

### **What are: FLOP, FLOPs, FLOPS, FLOP-days?**
- **FLOP**: Floating Point Operation
- **FLOPs**: Floating Point Operations (plural)
- **FLOPS**: Floating Point Operations per Second (with a capital S)
- **FLOP-days**: Floating Point Operations per Day (also known as FLOPs per day)
### **How many GPUs are needed to train a 405B parameter LLM like Llama 3?** (in other words, why are companies buying a gazillion GPUs)
*Assumptions:*
- Assuming NVIDIA H100 has peak performance of ~1979 TFLOPS (assume an average of 1000 TFLOPS during the course of actual training due to variable utilization %, etc.). We're assuming that only H100s were used for pre-training.
- A token is ~4 bytes
- a forwards and backwards pass of a single token is ~6 FLOPs

Llama 3 405B was pre-trained on 15.6 trillion tokens. With that, we can estimate the total compute that was required to train the model:
```math
6 \times (15.6 \times 10^{12}) \times (405 \times 10^{9}) = 3.79 \times 10^{25} \; FLOPs
```
Now let's calculate how much compute a single H100 can deliver over the course of one day:
```math
Single\;H100\;during\;training:\; 1000 \times 10^{12} \; FLOPS
```
```math
Over\;a\;day\:(86400\;seconds):\; 86400 \times (1000 \times 10^{12}) = 8.64\times10^{19}\;FLOPdays
```
Remember that FLOPdays is the same as FLOPs per day. We can now calculate how many days it would take to perform the total compute of Llama 3 405B pre-training with a single H100:
```math
\frac{3.79\times10^{25}\;FLOPs}{8.64\times10^{19}\;FLOPs\;per\;day}=438657\;days=1200\;years
```
Interesting. Okay, but **what if we want our model to finish training *before* the next millennium?** 

We're going to need more GPUs. See Figure 1.

Figure 1:

<img width="567" alt="Screenshot 2025-02-28 at 12 28 13 PM" src="https://github.com/user-attachments/assets/a4a322b7-7eee-4d57-9774-baeb4587a9d3" />


What if we could deliver 16,000 H100s to this training task? 
```math
\frac{3.79\times10^{25}\;FLOPs}{16000\times(8.64\times10^{19}\;FLOPs\;per\;day)}=27\;days
```
That's more like it. 

### **How much GPU memory is required to train a 405B parameter model? What about an N-parameter model?**
Based on https://huggingface.co/blog/dpo_vlm

To calculate the memory requirement of a N-parameter model alone is fairly straightforward. Assuming the model has N parameters, where each parameter is of precision P (# of bytes, i.e. P=4 for float32)
```math
Model~alone:~N\times P
```
This is equivalent to the calculation to determine roughly how much memory is needed to load the trained model for *inference*. 

However, during pre-training, there's other data structures being maintained in GPU memory as well: (1) reference model, (2) gradients, and (3) optimizer states.

```math
Reference~model:~N\times P
```
```math
Gradients:~N\times P
```
```math
Optimizer~states:~S\times N\times P
```
```math
S = states~per~parameter
```
For Llama 3 405B (N = 405 x 10^9, P = 4), which uses the AdamW optimizer (2 states per parameter), this sums up to:
```math
(405 \times 10^{9}) \times 4
```
```math
+~(405 \times 10^{9}) \times 4
```
```math
+~(405 \times 10^{9}) \times 4
```
```math
+~2 \times (405 \times 10^{9}) \times 4
```
```math
=~8.1 \times 10^{12}~bytes = 8.1~TB~of~GPU~memory
```
That's kind of a lot, considering that a single H100 has about 80 GB of GPU memory. However, we already know from the above calculations that doing anything of this scale with a single GPU is not the best idea. Let's continue with the assumption that we have 16,000 H100s (roughly 3 orders of magnitude more GPU memory than what is required). 

**Note**: As a rough, top of mind metric, you can go with: "20GB of GPU memory per 1 billion parameters".

### **Where does the model actually go?** **How does it get trained across these gazillion GPUs?**

That's great that we have a lot of GPUs. We address the compute time issue (a month versus 1200 years) and we address the GPU memory capacity issue. So how does the model actually get emplaced on these 16,000 GPUs and trained? These questions are the heart of **Model Parallelism**.
#### Model Parallelism:
The model’s layers and weight matrices are **sharded** among the 16,000 GPUs so that each GPU stores and processes a portion of the overall model. As we calculated above, it simply would not be possible to store a replica of the model on each GPU. There are two main approaches to this sharding action: (1) _Pipeline parallelism_ – dividing the model by layers (depth-wise), and (2) _Tensor parallelism_ – splitting the computations _within_ a layer across GPUs. Oftentimes these approaches are combined as well. 

Ultimately, this giant 405B parameter LLM might be **spread over dozens or hundreds of GPUs** via these model parallelism techniques, and then **many such replicas** run in parallel (data parallel) to utilize *all* 16k GPUs.  During training, **gradients are synchronized** both *within* each model replica and *across* all replicas, so the model updates as if it were one giant model being trained on the combined data from all GPUs.

For a deeper dive, check out this great blog post by Jeremy Jordan: https://www.jeremyjordan.me/distributed-training/

### **How large is the training data? How is the sharded and replicated model actually trained on this data?**

LLMs are next-token generators. They are very good at receiving a bunch of input tokens (sentences, images, amino acid sequences, etc.) and then generating highly plausible tokens back (more sentences, images, protein geometries, etc.). Accordingly, they are trained on a ton of tokens. Llama 3 405B was trained on 15.6 trillion tokens. A token is roughly 4 bytes, so that makes 60ish TB of training tokens. 

Considering we are talking about a massive training cluster that has 16,000 GPUs and massive data infrastructure, 60 TB seems kinda small [1] right? Recall our earlier calculation about training time. Ultimately, 15.6T tokens is costly in terms of *compute*, not storage. 

During each step of training, a **global batch** of training tokens is to be processed by the training cluster. This might be something like 16M total tokens per training step. This global batch of tokens would be evenly split across all 16,000 GPUs, so we can say each GPU (thereby a shard of the model) processes a *distinct* subset 1000 tokens during any given training step. An "epoch" represents one full pass through the entire training data, where every training token passed through one GPU.

But what about multiple passes over the same tokens? Since the model is split across 16,000 GPUs, does a particular training sample need to be "seen" by multiple (or all) shards of the model? **Not really**, and here's why:

1. Due to gradient synchronization, the "learnings" from one subset of the training tokens that passed through *one* model shard on *one* GPU can be propagated very effectively to the singular "common" model. 
2. Furthermore, with datasets so large, models are becoming capable of learning quite well **without** much (or any) repetition of tokens (i.e. passing the same token through multiple shards of the model on multiple GPUs). 

This is what enables GPUs to linearly speed up training – 16,000 GPUs can process 16,000 times more samples per step than a single GPU and we can be confident that the whole model is still getting updated correctly. See Figure 1.


[1]: 60 TB is the size of the training *tokens* which is a highly compressed representation of the original training data which can consists of petabytes of images, video, text, etc. (i.e. "trained on the entire internet").

**Note**: For more details on what goes into the formation of the training dataset, the pre-processing workload that leads to training tokens, and the common practice of **caching the training tokens on node-local NVMe from shared object store**, check out Glenn Lockwood's post: https://blog.glennklockwood.com/2025/02/llm-training-without-parallel-file.html


