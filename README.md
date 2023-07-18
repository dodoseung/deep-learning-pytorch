# Deep Learning Pytorch

This project is a set of repositories written by Seungwon Do including various deep learning algorithms using Pytorch.

## Reinforcement Learning

### Model-free Reinforcement Learning

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
4. [Advantage Actor-Critic (A2C)](https://github.com/dodoseung/a2c-advantage-actor-critic-pytorch)
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
		* [Numeric state, continuous action, and deterministic action](https://github.com/dodoseung/ddpg-deep-deterministic-policy-gradient-pytorch/blob/main/ddpg.py)
	* Reference paper
		* [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971v6)
7. [Trust Region Policy Optimization (TRPO)](https://github.com/dodoseung/trpo-trust-region-policy-optimization-pytorch)
	* Implementation
		* [Numeric state and discrete action](https://github.com/dodoseung/trpo-trust-region-policy-optimization-pytorch/blob/main/trpo_discrete.py)
		* [Numeric state, continuous action, and stochastic action](https://github.com/dodoseung/trpo-trust-region-policy-optimization-pytorch/blob/main/trpo_continous.py)
	* Reference paper
		* [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477v5)
7. [Proximal Policy Optimization (PPO)](https://github.com/dodoseung/ppo-proximal-policy-optimization-pytorch)
	* Implementation
		* [Numeric state, discrete action, and gae](https://github.com/dodoseung/ppo-proximal-policy-optimization-pytorch/blob/main/ppo_discrete_gae.py)
		* [Numeric state, continuous action, stochastic action, and gae](https://github.com/dodoseung/ppo-proximal-policy-optimization-pytorch/blob/main/ppo_continous_gae.py)
	* Reference paper
		* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347v2)
7. [Twin Delayed Deep Deterministic Policy Gradient (TD3)](https://github.com/dodoseung/td3-twin-delayed-deep-deterministic-policy-gradient-pytorch)
	* Implementation
		* [Numeric state, continuous action, and deterministic action](https://github.com/dodoseung/td3-twin-delayed-deep-deterministic-policy-gradient-pytorch/blob/main/td3.py)
	* Reference paper
		* [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477v3)
8. [Soft Actor-Critic (SAC)](https://github.com/dodoseung/sac-soft-actor-critic-pytorch)
	* Implementation
		* [Numeric state, discrete action, and entropy](https://github.com/dodoseung/sac-soft-actor-critic-pytorch/blob/main/sac_discrete_entropy.py)
		* [Numeric state, continuous action, and stochastic action](https://github.com/dodoseung/sac-soft-actor-critic-pytorch/blob/main/sac_continous_stochastic.py)
		* [Numeric state, continuous action, stochastic action, and entropy](https://github.com/dodoseung/sac-soft-actor-critic-pytorch/blob/main/sac_continous_stochastic_entropy.py)
	* Reference paper
		* [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290v2.pdf)
		* [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
		* [Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/abs/1910.07207)

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
To be implemented

1. []()
	* Implementation
		* []()
		* []()
	* Reference paper
		* []()
-->

## Computer Vision
### Image Classification and Semantic Segmentation
1. [WaveMix-Lite](https://github.com/dodoseung/wavemix-lite-pytorch)
	* Implementation
		* [WaveMix-Lite Model](https://github.com/dodoseung/wavemix-lite-pytorch/blob/master/wavemix_lite.py)
		* [Image classification and CIFAR10 dataset](https://github.com/dodoseung/wavemix-lite-pytorch/blob/master/wavemix_lite_cifar10_image_classification.py)
		* [Image classification and Places-365 dataset](https://github.com/dodoseung/wavemix-lite-pytorch/blob/master/wavemix_lite_places365_image_classification.py)
   		* [Semantic segmentation and Cityscapes dataset](https://github.com/dodoseung/wavemix-lite-pytorch/blob/master/wavemix_lite_cityscapes_semantic_segmentatiopn.py)
	* Reference paper
		* [WaveMix-Lite: A Resource-efficient Neural Network for Image Analysis](https://arxiv.org/abs/2205.14375)
2. [U-Net](https://github.com/dodoseung/unet-pytorch)
	* Implementation
		* [U-Net Model](https://github.com/dodoseung/unet-pytorch/blob/main/unet.py)
   		* [Semantic segmentation and Cityscapes dataset](https://github.com/dodoseung/unet-pytorch/blob/main/unet_cityscapes_semantic_segmentatiopn.py)
	* Reference paper
		* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
### Generative Model
1. [Auto Encoder (AE)](https://github.com/dodoseung/auto-encoder-pytorch)
	* Implementation
		* [Denoise](https://github.com/dodoseung/auto-encoder-pytorch/blob/master/auto_encoder.py)
2. [Variational Auto Encoder (VAE)](https://github.com/dodoseung/vae-variational-auto-encoder-pytorch)
	* Implementation
		* [VAE Model](https://github.com/dodoseung/vae-variational-auto-encoder-pytorch/blob/main/variational_auto_encoder.py)
		* [Image generation and EMNIST dataset](https://github.com/dodoseung/vae-variational-auto-encoder-pytorch/blob/main/variational_auto_encoder_emnist_image_generation.py)
	* Reference paper
		* [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114v10)
3. [Residual Encoder-Decoder Network (REDNet)](https://github.com/dodoseung/rednet-residual-encoder-decoder-network-pytorch)
	* Implementation
		* [Super resolution](https://github.com/dodoseung/rednet-residual-encoder-decoder-network-pytorch/blob/master/rednet.py)
	* Reference paper
		* [Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections](https://arxiv.org/abs/1603.09056)

## Natural Language Processing
### Pre-trained Language Model
1. [Transformer](https://github.com/dodoseung/transformer-pytorch)
	* Implementation
		* [Transformer Model](https://github.com/dodoseung/transformer-pytorch/blob/master/transformer.py)
		* [Translator DE to EN](https://github.com/dodoseung/transformer-pytorch/blob/master/translator_de_to_en.py)
	* Reference paper
		* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
1. [Generative Pre-Training 2 (GPT-2)](https://github.com/dodoseung/gpt2-generative-pre-training-2-pytorch)
	* Implementation
		* [GPT2 Model](https://github.com/dodoseung/gpt2-generative-pre-training-2-pytorch/blob/master/gpt2.py)
	* Reference paper
		* [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

