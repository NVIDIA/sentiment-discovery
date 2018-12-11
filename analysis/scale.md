# Data Parallel Scalability
Training a model on an amazon-review-sized dataset is a significant time investment. In order to improve our ability to iterate on experiments we found it imperative early on to investigate the scalability of distributed data parallelism in PyTorch. Multiplicative LSTM models were trained on DGX-1Vs with NVLINK and 8 Tesla V100's per node. To best utilize the V100s' tensorcores, training was performed with a batch size of 256 per gpu and [mixed precision training](./reproduction.md#fp16-training). The gradient all reduce between GPUs was performed in fp16 to take advantage of the NCCL2 library.

For our data parallel implementation we utilize a DistributedDataParallel model and 1 process per gpu in order to avoid common problems associated with GIL locking and multiprocessing. We investigated distributed training on up to 128 gpus across 16 DGX1-V nodes. We additionally analyzed the effect on throughput scaling of different model sizes with infiniband (200GB/s) and ethernet (10GB/s) interconnects between the nodes.

![scaling graph](../figures/distributed_scalability.png "(Distributed) Data Parallelism Scalability")

We find that with this system setup we are able to achieve relatively linear speedup relative to the number of available GPUs. Infiniband interconnects allow for greater throughput scaling with multiple nodes than ethernet allows for. Additionally, we find that throughput scaling with Infiniband is better with larger models (8k mLSTM model) since the increased communication overhead is less than the increase in compute intensity for these larger mLSTM models. We expect that this trend would continue similarly to other models.

## PyTorch + GIL
In order to ensure thread-safe execution the python threads compete for a global lock needed to interpret python commands. This is problematic for running multiple processes within a single python processes (such as data loading process+multiple parralel computational graph executions), and can negatively impact the performance of data parallel training implementations. To circumvent this we utilize a distributed data parallel training paradigm with 1 python process per gpu so that each computational unit has full access to the python interpreter.

-----

[<- Reproducing Results](./reproduction.md) | [Open Questions ->](./questions.md)