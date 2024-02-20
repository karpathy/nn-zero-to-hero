
## Neural Networks: Zero to Hero

A course on neural networks that starts all the way at the basics. The course is a series of YouTube videos where we code and train neural networks together. The Jupyter notebooks we build in the videos are then captured here inside the [lectures](lectures/) directory. Every lecture also has a set of exercises included in the video description. (This may grow into something more respectable).

---

**Lecture 1: The spelled-out intro to neural networks and backpropagation: building micrograd**

Backpropagation and training of neural networks. Assumes basic knowledge of Python and a vague recollection of calculus from high school.

- [YouTube video lecture](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Jupyter notebook files](lectures/micrograd)
- [micrograd Github repo](https://github.com/karpathy/micrograd)

---

**Lecture 2: The spelled-out intro to language modeling: building makemore**

We implement a bigram character-level language model, which we will further complexify in followup videos into a modern Transformer language model, like GPT. In this video, the focus is on (1) introducing torch.Tensor and its subtleties and use in efficiently evaluating neural networks and (2) the overall framework of language modeling that includes model training, sampling, and the evaluation of a loss (e.g. the negative log likelihood for classification).

- [YouTube video lecture](https://www.youtube.com/watch?v=PaCmpygFfXo)
- [Jupyter notebook files](lectures/makemore/makemore_part1_bigrams.ipynb)
- [makemore Github repo](https://github.com/karpathy/makemore)

---

**Lecture 3: Building makemore Part 2: MLP**

We implement a multilayer perceptron (MLP) character-level language model. In this video we also introduce many basics of machine learning (e.g. model training, learning rate tuning, hyperparameters, evaluation, train/dev/test splits, under/overfitting, etc.).

- [YouTube video lecture](https://youtu.be/TCH_1BHY58I)
- [Jupyter notebook files](lectures/makemore/makemore_part2_mlp.ipynb)
- [makemore Github repo](https://github.com/karpathy/makemore)

---

**Lecture 4: Building makemore Part 3: Activations & Gradients, BatchNorm**

We dive into some of the internals of MLPs with multiple layers and scrutinize the statistics of the forward pass activations, backward pass gradients, and some of the pitfalls when they are improperly scaled. We also look at the typical diagnostic tools and visualizations you'd want to use to understand the health of your deep network. We learn why training deep neural nets can be fragile and introduce the first modern innovation that made doing so much easier: Batch Normalization. Residual connections and the Adam optimizer remain notable todos for later video.

- [YouTube video lecture](https://youtu.be/P6sfmUTpUmc)
- [Jupyter notebook files](lectures/makemore/makemore_part3_bn.ipynb)
- [makemore Github repo](https://github.com/karpathy/makemore)

---

**Lecture 5: Building makemore Part 4: Becoming a Backprop Ninja**

We take the 2-layer MLP (with BatchNorm) from the previous video and backpropagate through it manually without using PyTorch autograd's loss.backward(). That is, we backprop through the cross entropy loss, 2nd linear layer, tanh, batchnorm, 1st linear layer, and the embedding table. Along the way, we get an intuitive understanding about how gradients flow backwards through the compute graph and on the level of efficient Tensors, not just individual scalars like in micrograd. This helps build competence and intuition around how neural nets are optimized and sets you up to more confidently innovate on and debug modern neural networks.

I recommend you work through the exercise yourself but work with it in tandem and whenever you are stuck unpause the video and see me give away the answer. This video is not super intended to be simply watched. The exercise is [here as a Google Colab](https://colab.research.google.com/drive/1WV2oi2fh9XXyldh02wupFQX0wh5ZC-z-?usp=sharing). Good luck :)

- [YouTube video lecture](https://youtu.be/q8SA3rM6ckI)
- [Jupyter notebook files](lectures/makemore/makemore_part4_backprop.ipynb)
- [makemore Github repo](https://github.com/karpathy/makemore)

---

**Lecture 6: Building makemore Part 5: Building WaveNet**

We take the 2-layer MLP from previous video and make it deeper with a tree-like structure, arriving at a convolutional neural network architecture similar to the WaveNet (2016) from DeepMind. In the WaveNet paper, the same hierarchical architecture is implemented more efficiently using causal dilated convolutions (not yet covered). Along the way we get a better sense of torch.nn and what it is and how it works under the hood, and what a typical deep learning development process looks like (a lot of reading of documentation, keeping track of multidimensional tensor shapes, moving between jupyter notebooks and repository code, ...).

- [YouTube video lecture](https://youtu.be/t3YJ5hKiMQ0)
- [Jupyter notebook files](lectures/makemore/makemore_part5_cnn1.ipynb)

---


**Lecture 7: Let's build GPT: from scratch, in code, spelled out.**

We build a Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3. We talk about connections to ChatGPT, which has taken the world by storm. We watch GitHub Copilot, itself a GPT, help us write a GPT (meta :D!) . I recommend people watch the earlier makemore videos to get comfortable with the autoregressive language modeling framework and basics of tensors and PyTorch nn, which we take for granted in this video.

- [YouTube video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY). For all other links see the video description.

---

**Lecture 8: Let's build the GPT Tokenizer**

The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizers are a completely separate stage of the LLM pipeline: they have their own training sets, training algorithms (Byte Pair Encoding), and after training implement two fundamental functions: encode() from strings to tokens, and decode() back from tokens to strings. In this lecture we build from scratch the Tokenizer used in the GPT series from OpenAI. In the process, we will see that a lot of weird behaviors and problems of LLMs actually trace back to tokenization. We'll go through a number of these issues, discuss why tokenization is at fault, and why someone out there ideally finds a way to delete this stage entirely.

- [YouTube video lecture](https://www.youtube.com/watch?v=zduSFxRajkE)
- [minBPE code](https://github.com/karpathy/minbpe)
- [Google Colab](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing)

---

Ongoing...

**License**

MIT