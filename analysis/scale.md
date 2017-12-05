# Data Parallel Scalability
Training a model on an amazon-review-sized dataset is a significant time investment. In order to improve our ability to iterate on experiments we found it imperative early on to investigate the scalability of data parallelism in PyTorch. The model was trained on Tesla V100's (volta), Tesla P100's (pascal), and VCA 6000's (maxwell), with a batch size of 32 per gpu, in order to benchmark wall-time processing speed (in characters/second) against OpenAI's reported speed of 12.5k characters/second. Four of our pascall-class gpus achieved a combined speed of 13.4k characters/second.

![scaling graph](../figures/both_scalability.png "(Distributed) Data Parallelism Scalability")

Our investigation unfortunately held up fears from the PyTorch community about the abysmal performance of data parallelism with recurrent networks. **Pytorch's vanilla DataParallel wrapper produces little to no speedup for recurrent architectures** due to issues related to the python Global Interpreter Lock (GIL). 

DistributedDataParallel was fortunately not plagued by the same lock problems. DistributedDataParallel starts a separate process for each model replica and communicates gradient updates directly via all-reduce calls on the gpu for minimal latency. This allows for relatively linear scaling accounting for redundant data processing being done in each process, which will be adressed in future updates.

Additionally, we tried setting `torch.backends.cuda.benchmark=True`, but it provided no observable benefit to scalability either, and in fact slightly hurt performance due to its additional overhead. 

We have yet to analyze gpu utilization in depth beyond wall-time training performance.

## PyTorch + GIL
All the parallel python threads involved in PyTorch's DataParallelism use the same c-backed data structure to run tensor ops and aggregate gradients with. In order to ensure thread-safe execution the python threads compete for a global lock needed to interpret python commands that run (GPU) operations on tensors in the computation graph. This is problematic for recurrent architectures, where there are numerous of the same operation to call, each requiring only a little computation (as opposed to the heftier operations characteristic of image-processing models). In recurrent architectures no speedup will be observed as each thread spends a significant portion of time trying to interpret the same recurrent function calls numerous times while either passing this lock around or waiting for it.

-----

[<- Reproducing Results](./reproduction.md) | [Open Questions ->](./questions.md)