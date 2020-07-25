# Open Questions
The best research work often leaves the reader with more questions than it answers. Here's some of the questions we thought of while reproducing this work.
 * Waiting for long training runs to end is frustrating, and impacts research and development. How can we further improve the speed and scalability of our system?
 * Sentiment is not just black or white, positive or negative, it's much more nuanced and a result of multiple factors. What are the other significant features our regression model is focusing on? What other discriminative language tasks would these features do well on?
   * More generally, what other tasks could unsupervised language modeling at scale be used for? Discriminative or not.
 * The performance of the model on sentiment transfer seems to exhibit uncertainty dependent on the most recently modeled samples from the unsupervised task. Due to this apparent dependence on data, we'd ideally like to fine tune our model to a subset of data that is relevant to the task being transferred to. With this in mind, given a specific subset of data, how can we fine-tune our language model to the specified data?
 * Our model learned sentiment because it was a relevant piece of information for reconstruction of product reviews, not because it was told to learn sentiment. In our case we even found that it learned negative sentiment, contrary to OpenAI's model. In an ideal scenario we should have the ability to supervise/force our model to learn about a specific characteristic, such as author frustration. Given a target transfer task, how can performance on that task be used to regularize what the model learns about language? 
 * Most importantly, can the above be achieved without adversely impacting performance? Will the model retain its understanding of general language properties and its ability to generalize and transfer well?  

-----

[<- Data Parallel Scalability](./scale.md)
