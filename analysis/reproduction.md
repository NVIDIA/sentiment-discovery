# Reproducing Results
Our reproduction achieved comparable results in both unsupervised reconstruction and sentiment transfer. This is denoted by solid lines for our model and dashed lines for OpenAI's model. We used weights sourced from OpenAI to manually verify performance on our instance of the data.

![reproduction results graph](../figures/reproduction_graph.png "Reproduction metrics")

## Training 
Contrary to results in the OpenAI work the validation reconstruction loss is lower than the training loss. 

We hypothesize that this is due to selecting shards before shuffling the dataset, causing similar samples to end up in the same shard. Even though one shard amounts to `80 million/1002 ~ 80000` reviews, it's still not a representative slice (\< .1%) of the dataset, especially when some categories such as books dominate ~10% of total sample reviews. 

The validation loss could then be low because of one of two (or both) rationales:
 * A non-representative set of data samples that the model excels at. 
 * Similar samples benefit from the hidden state persisted from earlier samples. 

More investigation is required however, to fully interpret this phenomena.

### Training Set Up
It took several cycles of trial and error to come up with a result comparable to the original. Some things were not entirely apparent from the paper, key model details were often hidden in one line, and took several tries to get right. Other minutia were found out independently. We've included what we found to work well.
 * **Model**: 4096-d mLSTM, 64-d embedding, 256-d output. (we also trained a similarly parameterized lstm)
 * **Weight Norm**: applied only to lstm parameters (hidden->hidden/gate weights), not embedding or output. 
 * **Optimizer**: Adam
 * **Learning Rate**: 5e-4 per batch of 128. Linear Learning rate decay to 0 over course of epoch.
 * **Gradient Clipping**: We occassionally ran into problems with destabilizing gradient explosions. Therfore, we clipped our gradients to a maximum of `1.`.
 * **Data set**: Aggressively Deduplicated Amazon Review dataset with 1000/1/1 train/test/validation shards. Each of the three sets are internally shuffled. Samples in a shard are concatenated together so as to persist state across gradient updates.
 * **State Persistence**: The hidden state is persisted across all samples and reset at the start of a shard.
 * **Batch per gpu**: 128 (instead of OpenAI's 32).
 * **Hardware**: 8 volta-class gpus (instead of OpenAI's 4 pascal)
 * **Learning Rate Scaling**: We take queues from recent work in training imagenet at scale and leverage [FAIR's (Goyal et. al 2017)](https://arxiv.org/pdf/1706.02677.pdf) linear scaling rule. To account for our 4x batch size increase and 2x gpu increase we used a learning rate of `5e-4 * 8 -> 4e-3`.
 * **Processing Speed**: With our hardware and batch size we achieved a processing speed (wall-time) of 76k characters/second compared to OpenAI's 12.5k ch/s
 * **Processing Time**: It took approximately 5 days to train on the million samples of the paper and 6.5 days to train on a full epoch of the amazon dataset.

## Transfer
We chose to reproduce transfer results with the binary Stanford Sentiment Treebank as opposed to the IMDB dataset because of its smaller size, and faster turnaround time for experiments.

Data samples are featurized by running a forward pass of the model over the data and extracting the cell (not hidden) state from the last token. 

These features are used for training a logistic regression model (via sklearn/PyTorch) against the samples' associated labels. We use classification performance on a hold out validation set to select an L1 regularization hyperparameter. 

The feature with the largest L1 penalty is then treated as a "sentiment neuron" to classify the samples and analyze difference in performance.

**It should be noted that the our model learned a negatively correlated sentiment feature**

------

[<- Why Unsupervised Language Modeling?](./unsupervised.md) | [Data Parallel Scalability ->](./scale.md)