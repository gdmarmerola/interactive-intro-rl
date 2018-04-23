# interactive-intro-rl

## An Interactive Introduction to Reinforcement Learning

In this repository, we extend the code presented for the first of Big Data's open seminar series, *An Interactive Introduction to Reinforcement Learning*, presented at 2017-11-21. In this talk, we introduce various algorithms for solving the exploration/exploitation trade-off in reinforcement learning. We present four scenarios: the Multi-Armed Bandit (Bernoulli) setting, the Contextual Bandit setting, the Mushroom Bandit, and Bayesian Optimization of hyperparameters. The algorithms and concepts are presented with mathematical concepts, code, and interactive animations conveyed using Jupyter Notebooks.

## What you'll learn

### The Multi-Armed Bandit problem

We present the Multi-Armed Bandit problem, the simplest setting of reinforcement learning, which is perfect for illustrating the exploration/exploitation trade-off. Suppose that you face a row of slot machines (bandits) on a casino. Each one of the **K** machines has a fixed probability **theta_k** of providing a binary reward to you. You have to decide which machines to play, how many times to play each machine and in which order to play them, in order to maximize your cumulative reward. 

If after a few plays you bet everything on the best machine so far, you risk getting stuck with a suboptimal strategy, as you lack sufficient information to discard the others. If you explore too much, you risk wasting many of your plays gathering information instead of making profits. 

We will show many algorithms to address this dilemma, highlighting a very promising heuristic, which despite being very old (1933) was rediscovered recently showing state-of-the-art results: **Thompson Sampling**. To illustrate the algorithms, we show interactive animations like [this one](https://github.com/bigdatabr/interactive-intro-rl/blob/master/thompson_sampling_mab.mp4), to show the algorithm choices over time along with the posterior distributions for the bandit probabilities **theta_k**.


### The Contextual Bandit problem

We also present the Contextual Bandit problem, which is almost the same as the Multi-Armed Bandit problem, but the bandit probabillities 
**theta_k** are actually a function of some exogenous variables:

**theta_k(x) = f(x)**

To solve this problem, we implement two algorithms: (a) an e-greedy strategy using a regular logistic regression for modeling **theta_k(x)** and (b) Thompson Sampling with the Online Logistic Regression by Chapelle & Li on their 2011 [paper](https://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling) "An Empirical Evaluation of Thompson Sampling". The Online Logistic Regression allows for uncertainty in its coefficients, such that we can have not only a point estimate for **theta_k(x)**, but a distribution. The fitting process of the algorithm is studied with [this animation](https://github.com/bigdatabr/interactive-intro-rl/blob/master/thompson_sampling_olr.mp4). We also show animations like the ones shown in the first section.

### More Contextual Bandits: the Mushroom Bandit

In this demonstration, we use the Mushroom dataset from UCI in order to illustrate what a real-life contextual bandit might look like. The dataset contains many features describing edible or non-edible mushrooms. We set up a game in which an agent has to choose, at each round, the top **k** most edible mushrooms amongst all of the remaining mushrooms. The agent which eats all the edible mushrooms first wins. In order to quickly learn the patterns of edible mushrooms, efficient exploration is needed. We implement Bootstrapped Neural Networks inspired by [Osband et. al. (2016)](https://arxiv.org/abs/1602.04621) and experiment with sampling trees from a Random Forest in order to get uncertainty estimates. We play the game with these techniques and compare them to their greedy counterparts (regular Neural Network and RF).

### Bayesian Optimization of hyperparamters

Hyperparameter optimization (Algorithm Configuration, or even, Automated Machine Learning) is one of the toughest problems in ML and has received increased attention in recent years. In this tutorial, we frame the hyperparameter optimization problem as a reinforcement learning problem. Given this formulation, our reward will be the chosen model's score in the validation set, and our action will be the choice of a hyperparameter set. We choose Gaussian Processes to perform a regression from the hyperparameters to the validation score, which allows us to get a distribution over functions, and therefore, apply Thompson Sampling. 

We first show a simple example which illustrates the bigger problem: optimizting the non-convex [Ackley's function](https://en.wikipedia.org/wiki/Ackley_function), without using gradient information. Then, we build a validation framework and optimize the regularization parameter of a linear model with L1 regularization (LASSO). 



## Required libraries

We use Python 3.6.2. Most of the requirements are satistified by installing the Anaconda Python distribution. The following libraries are required:

* **Jupyter Notebook** (to make use of IPython functions)
* **numpy** and **scipy** for calculations, distibutons and optimization
* **matplotlib** and **seaborn** for visualizations
* **pandas** to manipulate data
* **sklearn** to run Logistic Regression and Random Forests
* **tqdm** to time simulation executions
* **tensorflow** and **keras**: for neural networks

## Running the Notebooks

The Notebooks are self-contained. We encorage you to clone the repository and run the Notebooks, changing parameters as you like. Any feedback is deeply appreciated.

