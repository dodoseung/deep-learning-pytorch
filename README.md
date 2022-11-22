# Deep Leanring Pytorch

This project is the set of repositories written by Seungwon Do including a various deep learning algorithms using Pytorch.

## Reinforcement Learning

### Model-free Reinforcment Learning
1. [Deep Q Network (DQN)](https://github.com/dodoseung/dqn-deep-q-network-pytorch)
	* Implementation
		* [Numeric state and discrete action](https://github.com/dodoseung/dqn-deep-q-network-pytorch/blob/main/dqn.py)
		* [Image state and discrete action](https://github.com/dodoseung/dqn-deep-q-network-pytorch/blob/main/dqn_image_input.py)
	* Reference paper
		* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602v1)
2. [Vanilla Policy Gradient (VPG)](https://github.com/dodoseung/vpg-vanilla-policy-gradient-pytorch)
	* Implementation
		* [Numeric state and discrete action](https://github.com/dodoseung/vpg-vanilla-policy-gradient-pytorch/blob/main/vpg.py)
3. [Actor Critic (AC)](https://github.com/dodoseung/ac-actor-critic-pytorch)
	* Implementation
		* [Numeric state and discrete action](https://github.com/dodoseung/ac-actor-critic-pytorch/blob/main/ac.py)
4. [Advantage Actor Critic (A2C)](https://github.com/dodoseung/a2c-advantage-actor-critic-pytorch)
	* Implementation
		* [Numeric state, discrete action, and training per step](https://github.com/dodoseung/a2c-advantage-actor-critic-pytorch/blob/main/a2c.py)
		* [Numeric state, discrete action, and training per episode](https://github.com/dodoseung/a2c-advantage-actor-critic-pytorch/blob/main/a2c_per_epi.py)
	* Reference paper
		* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783v2)
5. [Asynchronous Advantage Actor Critic (A3C)](https://github.com/dodoseung/a3c-asynchronous-advantage-actor-critic-pytorch)
	* Implementation
		* [Numeric state and discrete action](https://github.com/dodoseung/a3c-asynchronous-advantage-actor-critic-pytorch/blob/main/a3c.py)
	* Reference paper
		* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783v2)
6. [Deep Deterministic Policy Gradient (DDPG)](https://github.com/dodoseung/ddpg-deep-deterministic-policy-gradient-pytorch)
	* Implementation
		* [Numeric state and continous action](https://github.com/dodoseung/ddpg-deep-deterministic-policy-gradient-pytorch/blob/main/ddpg.py)
	* Reference paper
		* [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971v6)
7. [Trust Region Policy Optimization (TRPO)](https://github.com/dodoseung/trpo-trust-region-policy-optimization-pytorch)
	* Implementation
		* [Numeric state and discrete action](https://github.com/dodoseung/trpo-trust-region-policy-optimization-pytorch/blob/main/trpo_discrete.py)
		* [Numeric state and continous action](https://github.com/dodoseung/trpo-trust-region-policy-optimization-pytorch/blob/main/trpo_continous.py)
	* Reference paper
		* [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477v5)
7. [Proximal Policy Optimization (PPO)](https://github.com/dodoseung/ppo-proximal-policy-optimization-pytorch)
	* Implementation
		* [Numeric state, discrete action, and gae](https://github.com/dodoseung/ppo-proximal-policy-optimization-pytorch/blob/main/ppo_discrete_gae.py)
		* [Numeric state, continous action, and gae](https://github.com/dodoseung/ppo-proximal-policy-optimization-pytorch/blob/main/ppo_continous_gae.py)
	* Reference paper
		* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347v2)
7. [Twin Delayed Deep Deterministic Policy Gradient (TD3)](https://github.com/dodoseung/td3-twin-delayed-deep-deterministic-policy-gradient-pytorch)
	* Implementation
		* [Numeric state and deterministic continous action](https://github.com/dodoseung/td3-twin-delayed-deep-deterministic-policy-gradient-pytorch/blob/main/td3.py)
	* Reference paper
		* [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477v3)

- [SAC](https://github.com/dodoseung/sac-soft-actor-critic-pytorch) - Soft Actor Critic | [Paper1](https://arxiv.org/pdf/1801.01290v2.pdf) [Paper2](https://arxiv.org/abs/1812.05905) [Paper3](https://arxiv.org/abs/1910.07207)
<!--### Model-based Reinforcement Learning
To be implemented...
### Partially Observable Markov Decision Process
To be implemented...
### Inverse Reinforcement Learning
To be implemented...
### Multi-agent Reinforcement Learning
To be implemented...
### Meta Reinforcement Learning
To be implemented...
### Hierarchical Reinforcement Learning
To be implemented...
### Distributed Reinforcement Learning
To be implemented...
### Exploration
To be implemented-->

## Computer Vision
### Auto Encoder
- [AE](https://github.com/dodoseung/auto-encoder-pytorch) - Auto Encoder
### Super Resolution
- [REDNet](https://github.com/dodoseung/vpg-vanilla-policy-gradient-pytorch) - Residual Encoder Decoder Network | [Paper](https://arxiv.org/abs/1603.09056)
### Image Classification
- [WaveMix-Lite](https://github.com/dodoseung/wavemix-lite-pytorch) - Resource-efficient Token Mixing for Images using 2D Discrete Wavelet Transform | [Paper](https://arxiv.org/abs/2205.14375)

## Natural Language Processing
### Pre-trained Language Model
- [Transformer](https://github.com/dodoseung/transformer-pytorch) - Transformer | [Paper](https://arxiv.org/abs/1706.03762)
- [GPT-2](https://github.com/dodoseung/gpt2-generative-pre-training-2-pytorch) - Generative Pre-Training | [Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
