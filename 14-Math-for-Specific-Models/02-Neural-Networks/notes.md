[<- Linear Models](../01-Linear-Models/notes.md) | [Home](../../README.md) | [Probabilistic Models ->](../03-Probabilistic-Models/notes.md)

---

# Neural Networks

Neural networks learn nonlinear feature maps and train them end to end with backpropagation. They are linear algebra, nonlinear activations, loss functions, and chain-rule gradients stacked into a trainable program.

## Overview

A feed-forward network is a composition:

$$
f_\theta(x)=f_L(f_{L-1}(\cdots f_1(x))).
$$

Each layer usually computes an affine map followed by a nonlinearity:

$$
h_{\ell+1}=\phi(W_\ell h_\ell+b_\ell).
$$

Backpropagation computes gradients of the loss with respect to every parameter by reusing intermediate derivatives from the output back to the input.

## Prerequisites

- Linear models and matrix multiplication
- Chain rule and gradients
- Cross-entropy and least-squares loss
- Basic optimization vocabulary

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates forward passes, backprop gradients, activations, initialization, gradient descent, Adam, normalization, dropout, and diagnostics. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for neural-network computations and debugging. |

## Learning Objectives

After this section, you should be able to:

- Explain why nonlinear activations are necessary.
- Compute a forward pass through a small MLP.
- Derive backprop gradients for affine layers and activations.
- Compare sigmoid, tanh, ReLU, GELU, and gated activations.
- Explain Xavier and He initialization.
- Implement SGD, momentum, Adam intuition, and gradient clipping.
- Explain BatchNorm, LayerNorm, dropout, and weight decay.
- Diagnose activation scale, gradient norms, overfitting, and optimization failure.

## Table of Contents

1. [From Linear Models to Neural Networks](#1-from-linear-models-to-neural-networks)
   - 1.1 [Learned feature map](#11-learned-feature-map)
   - 1.2 [Linear head](#12-linear-head)
   - 1.3 [Composition](#13-composition)
   - 1.4 [Nonlinearity](#14-nonlinearity)
   - 1.5 [Representation learning](#15-representation-learning)
2. [Forward Pass](#2-forward-pass)
   - 2.1 [Affine layer](#21-affine-layer)
   - 2.2 [Activation](#22-activation)
   - 2.3 [MLP layer stack](#23-mlp-layer-stack)
   - 2.4 [Batch vectorization](#24-batch-vectorization)
   - 2.5 [Output logits](#25-output-logits)
3. [Loss Functions](#3-loss-functions)
   - 3.1 [Regression MSE](#31-regression-mse)
   - 3.2 [Binary cross-entropy](#32-binary-crossentropy)
   - 3.3 [Softmax cross-entropy](#33-softmax-crossentropy)
   - 3.4 [Regularized objective](#34-regularized-objective)
   - 3.5 [Empirical risk](#35-empirical-risk)
4. [Backpropagation](#4-backpropagation)
   - 4.1 [Chain rule](#41-chain-rule)
   - 4.2 [Layer gradient](#42-layer-gradient)
   - 4.3 [Activation derivative](#43-activation-derivative)
   - 4.4 [Reverse accumulation](#44-reverse-accumulation)
   - 4.5 [Gradient check](#45-gradient-check)
5. [Activations](#5-activations)
   - 5.1 [Sigmoid](#51-sigmoid)
   - 5.2 [Tanh](#52-tanh)
   - 5.3 [ReLU](#53-relu)
   - 5.4 [GELU](#54-gelu)
   - 5.5 [SwiGLU](#55-swiglu)
6. [Initialization and Signal Propagation](#6-initialization-and-signal-propagation)
   - 6.1 [Variance propagation](#61-variance-propagation)
   - 6.2 [Xavier initialization](#62-xavier-initialization)
   - 6.3 [He initialization](#63-he-initialization)
   - 6.4 [Symmetry breaking](#64-symmetry-breaking)
   - 6.5 [Depth instability](#65-depth-instability)
7. [Optimization](#7-optimization)
   - 7.1 [SGD](#71-sgd)
   - 7.2 [Momentum](#72-momentum)
   - 7.3 [Adam](#73-adam)
   - 7.4 [Learning-rate schedule](#74-learningrate-schedule)
   - 7.5 [Gradient clipping](#75-gradient-clipping)
8. [Normalization and Regularization](#8-normalization-and-regularization)
   - 8.1 [Batch normalization](#81-batch-normalization)
   - 8.2 [Layer normalization](#82-layer-normalization)
   - 8.3 [Dropout](#83-dropout)
   - 8.4 [Weight decay](#84-weight-decay)
   - 8.5 [Early stopping](#85-early-stopping)
9. [Expressivity and Generalization](#9-expressivity-and-generalization)
   - 9.1 [Universal approximation](#91-universal-approximation)
   - 9.2 [Depth efficiency](#92-depth-efficiency)
   - 9.3 [Overparameterization](#93-overparameterization)
   - 9.4 [Double descent](#94-double-descent)
   - 9.5 [Inductive bias](#95-inductive-bias)
10. [Diagnostics](#10-diagnostics)
   - 10.1 [Shape checks](#101-shape-checks)
   - 10.2 [Activation statistics](#102-activation-statistics)
   - 10.3 [Gradient norms](#103-gradient-norms)
   - 10.4 [Train validation curves](#104-train-validation-curves)
   - 10.5 [Ablations](#105-ablations)

---

## Shape Map

```text
input batch:        X       shape (B, d_in)
layer weights:      W_l     shape (d_out, d_in)
pre-activation:     Z_l     shape (B, d_out)
activation:         H_l     shape (B, d_out)
logits:             Z       shape (B, classes)
```

## 1. From Linear Models to Neural Networks

This part studies from linear models to neural networks as trainable representation learning. Keep track of forward values, backward gradients, scale, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Learned feature map](#1-learned-feature-map) | make features trainable instead of fixed | $h=f_\theta(x)$ |
| [Linear head](#1-linear-head) | final prediction is often linear on learned features | $\hat y=W h+b$ |
| [Composition](#1-composition) | depth composes many simple maps | $f=f_L\circ\cdots\circ f_1$ |
| [Nonlinearity](#1-nonlinearity) | without nonlinear activations, stacked linear layers collapse to one linear map | $W_2W_1x$ |
| [Representation learning](#1-representation-learning) | hidden layers learn intermediate coordinates useful for the task | $h_\ell$ |

### 1.1 Learned feature map

**Main idea.** Make features trainable instead of fixed.

Core relation:

$$h=f_\theta(x)$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 1.2 Linear head

**Main idea.** Final prediction is often linear on learned features.

Core relation:

$$\hat y=W h+b$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 1.3 Composition

**Main idea.** Depth composes many simple maps.

Core relation:

$$f=f_L\circ\cdots\circ f_1$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 1.4 Nonlinearity

**Main idea.** Without nonlinear activations, stacked linear layers collapse to one linear map.

Core relation:

$$W_2W_1x$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 1.5 Representation learning

**Main idea.** Hidden layers learn intermediate coordinates useful for the task.

Core relation:

$$h_\ell$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
## 2. Forward Pass

This part studies forward pass as trainable representation learning. Keep track of forward values, backward gradients, scale, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Affine layer](#2-affine-layer) | matrix multiply plus bias | $z=W x+b$ |
| [Activation](#2-activation) | apply elementwise nonlinearity | $a=\phi(z)$ |
| [MLP layer stack](#2-mlp-layer-stack) | repeat affine and nonlinear transformations | $h_{\ell+1}=\phi(W_\ell h_\ell+b_\ell)$ |
| [Batch vectorization](#2-batch-vectorization) | process many examples together | $H_{\ell+1}=\phi(H_\ell W_\ell^\top+\mathbf{1}b_\ell^\top)$ |
| [Output logits](#2-output-logits) | classification networks usually produce logits before softmax | $z_K=W_Kh_K+b_K$ |

### 2.1 Affine layer

**Main idea.** Matrix multiply plus bias.

Core relation:

$$z=W x+b$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 2.2 Activation

**Main idea.** Apply elementwise nonlinearity.

Core relation:

$$a=\phi(z)$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 2.3 MLP layer stack

**Main idea.** Repeat affine and nonlinear transformations.

Core relation:

$$h_{\ell+1}=\phi(W_\ell h_\ell+b_\ell)$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 2.4 Batch vectorization

**Main idea.** Process many examples together.

Core relation:

$$H_{\ell+1}=\phi(H_\ell W_\ell^\top+\mathbf{1}b_\ell^\top)$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 2.5 Output logits

**Main idea.** Classification networks usually produce logits before softmax.

Core relation:

$$z_K=W_Kh_K+b_K$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
## 3. Loss Functions

This part studies loss functions as trainable representation learning. Keep track of forward values, backward gradients, scale, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Regression MSE](#3-regression-mse) | penalize squared prediction error | $L=\frac{1}{n}\sum_i\Vert \hat y_i-y_i\Vert^2$ |
| [Binary cross-entropy](#3-binary-crossentropy) | Bernoulli negative log likelihood | $-\left[y\log p+(1-y)\log(1-p)\right]$ |
| [Softmax cross-entropy](#3-softmax-crossentropy) | multi-class negative log likelihood | $L=-\log p_y$ |
| [Regularized objective](#3-regularized-objective) | add parameter penalties or other constraints | $L_\mathrm{total}=L+\lambda R(\theta)$ |
| [Empirical risk](#3-empirical-risk) | training minimizes average loss over data | $\hat R(\theta)=n^{-1}\sum_i\ell(f_\theta(x_i),y_i)$ |

### 3.1 Regression MSE

**Main idea.** Penalize squared prediction error.

Core relation:

$$L=\frac{1}{n}\sum_i\Vert \hat y_i-y_i\Vert^2$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 3.2 Binary cross-entropy

**Main idea.** Bernoulli negative log likelihood.

Core relation:

$$-\left[y\log p+(1-y)\log(1-p)\right]$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 3.3 Softmax cross-entropy

**Main idea.** Multi-class negative log likelihood.

Core relation:

$$L=-\log p_y$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 3.4 Regularized objective

**Main idea.** Add parameter penalties or other constraints.

Core relation:

$$L_\mathrm{total}=L+\lambda R(\theta)$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 3.5 Empirical risk

**Main idea.** Training minimizes average loss over data.

Core relation:

$$\hat R(\theta)=n^{-1}\sum_i\ell(f_\theta(x_i),y_i)$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
## 4. Backpropagation

This part studies backpropagation as trainable representation learning. Keep track of forward values, backward gradients, scale, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Chain rule](#4-chain-rule) | gradients flow backward through composed functions | $\partial L/\partial x=(\partial z/\partial x)^\top\partial L/\partial z$ |
| [Layer gradient](#4-layer-gradient) | weight gradient is outer product of upstream error and input | $\partial L/\partial W=\delta x^\top$ |
| [Activation derivative](#4-activation-derivative) | nonlinearity gates gradient flow | $\delta_z=\delta_a\odot\phi'(z)$ |
| [Reverse accumulation](#4-reverse-accumulation) | store intermediate activations and traverse backward | $\delta_L\rightarrow\delta_{L-1}\rightarrow\cdots$ |
| [Gradient check](#4-gradient-check) | finite differences verify backprop implementation | $\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}$ |

### 4.1 Chain rule

**Main idea.** Gradients flow backward through composed functions.

Core relation:

$$\partial L/\partial x=(\partial z/\partial x)^\top\partial L/\partial z$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** Backprop is the chain rule organized so every parameter receives credit efficiently.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 4.2 Layer gradient

**Main idea.** Weight gradient is outer product of upstream error and input.

Core relation:

$$\partial L/\partial W=\delta x^\top$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 4.3 Activation derivative

**Main idea.** Nonlinearity gates gradient flow.

Core relation:

$$\delta_z=\delta_a\odot\phi'(z)$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 4.4 Reverse accumulation

**Main idea.** Store intermediate activations and traverse backward.

Core relation:

$$\delta_L\rightarrow\delta_{L-1}\rightarrow\cdots$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 4.5 Gradient check

**Main idea.** Finite differences verify backprop implementation.

Core relation:

$$\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
## 5. Activations

This part studies activations as trainable representation learning. Keep track of forward values, backward gradients, scale, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Sigmoid](#5-sigmoid) | squash to (0,1) but can saturate | $\sigma(z)=1/(1+e^{-z})$ |
| [Tanh](#5-tanh) | zero-centered saturating activation | $\tanh z$ |
| [ReLU](#5-relu) | simple piecewise linear activation | $\max(0,z)$ |
| [GELU](#5-gelu) | smooth activation used in many transformers | $x\Phi(x)$ |
| [SwiGLU](#5-swiglu) | gated activation used in modern LLM MLPs | $(xW_1)\odot \mathrm{swish}(xW_2)$ |

### 5.1 Sigmoid

**Main idea.** Squash to (0,1) but can saturate.

Core relation:

$$\sigma(z)=1/(1+e^{-z})$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 5.2 Tanh

**Main idea.** Zero-centered saturating activation.

Core relation:

$$\tanh z$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 5.3 ReLU

**Main idea.** Simple piecewise linear activation.

Core relation:

$$\max(0,z)$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** ReLU made deep feed-forward networks easier to optimize than saturating activations in many settings.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 5.4 GELU

**Main idea.** Smooth activation used in many transformers.

Core relation:

$$x\Phi(x)$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 5.5 SwiGLU

**Main idea.** Gated activation used in modern llm mlps.

Core relation:

$$(xW_1)\odot \mathrm{swish}(xW_2)$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
## 6. Initialization and Signal Propagation

This part studies initialization and signal propagation as trainable representation learning. Keep track of forward values, backward gradients, scale, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Variance propagation](#6-variance-propagation) | activation variance should not explode or vanish across layers | $\mathrm{Var}(h_\ell)$ |
| [Xavier initialization](#6-xavier-initialization) | balance fan-in and fan-out for tanh-like activations | $\mathrm{Var}(W)=2/(n_{in}+n_{out})$ |
| [He initialization](#6-he-initialization) | scale for ReLU-like activations | $\mathrm{Var}(W)=2/n_{in}$ |
| [Symmetry breaking](#6-symmetry-breaking) | random weights let units learn different features | $W_i\ne W_j$ |
| [Depth instability](#6-depth-instability) | bad initialization makes gradients vanish or explode | $\prod_\ell J_\ell$ |

### 6.1 Variance propagation

**Main idea.** Activation variance should not explode or vanish across layers.

Core relation:

$$\mathrm{Var}(h_\ell)$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 6.2 Xavier initialization

**Main idea.** Balance fan-in and fan-out for tanh-like activations.

Core relation:

$$\mathrm{Var}(W)=2/(n_{in}+n_{out})$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 6.3 He initialization

**Main idea.** Scale for relu-like activations.

Core relation:

$$\mathrm{Var}(W)=2/n_{in}$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** Initialization is not decoration; it controls signal scale at depth.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 6.4 Symmetry breaking

**Main idea.** Random weights let units learn different features.

Core relation:

$$W_i\ne W_j$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 6.5 Depth instability

**Main idea.** Bad initialization makes gradients vanish or explode.

Core relation:

$$\prod_\ell J_\ell$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
## 7. Optimization

This part studies optimization as trainable representation learning. Keep track of forward values, backward gradients, scale, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [SGD](#7-sgd) | update parameters using mini-batch gradients | $\theta\leftarrow\theta-\eta g$ |
| [Momentum](#7-momentum) | smooth updates with velocity | $v\leftarrow\beta v+g$ |
| [Adam](#7-adam) | normalize first moment by second moment estimate | $\theta\leftarrow\theta-\eta\hat m/(\sqrt{\hat v}+\epsilon)$ |
| [Learning-rate schedule](#7-learningrate-schedule) | change step size over training | $\eta_t$ |
| [Gradient clipping](#7-gradient-clipping) | cap gradient norm to prevent unstable updates | $g\leftarrow g\min(1,c/\Vert g\Vert)$ |

### 7.1 SGD

**Main idea.** Update parameters using mini-batch gradients.

Core relation:

$$\theta\leftarrow\theta-\eta g$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 7.2 Momentum

**Main idea.** Smooth updates with velocity.

Core relation:

$$v\leftarrow\beta v+g$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 7.3 Adam

**Main idea.** Normalize first moment by second moment estimate.

Core relation:

$$\theta\leftarrow\theta-\eta\hat m/(\sqrt{\hat v}+\epsilon)$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 7.4 Learning-rate schedule

**Main idea.** Change step size over training.

Core relation:

$$\eta_t$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 7.5 Gradient clipping

**Main idea.** Cap gradient norm to prevent unstable updates.

Core relation:

$$g\leftarrow g\min(1,c/\Vert g\Vert)$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
## 8. Normalization and Regularization

This part studies normalization and regularization as trainable representation learning. Keep track of forward values, backward gradients, scale, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Batch normalization](#8-batch-normalization) | normalize features using batch statistics | $\hat x=(x-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$ |
| [Layer normalization](#8-layer-normalization) | normalize features within an example | $\hat x=(x-\mu)/\sqrt{\sigma^2+\epsilon}$ |
| [Dropout](#8-dropout) | randomly mask activations during training | $\tilde h=m\odot h/(1-p)$ |
| [Weight decay](#8-weight-decay) | penalize large weights | $L+\lambda\Vert\theta\Vert^2$ |
| [Early stopping](#8-early-stopping) | stop when validation loss stops improving | $L_\mathrm{val}$ |

### 8.1 Batch normalization

**Main idea.** Normalize features using batch statistics.

Core relation:

$$\hat x=(x-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 8.2 Layer normalization

**Main idea.** Normalize features within an example.

Core relation:

$$\hat x=(x-\mu)/\sqrt{\sigma^2+\epsilon}$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** LayerNorm is one of the basic stabilizers behind modern transformers.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 8.3 Dropout

**Main idea.** Randomly mask activations during training.

Core relation:

$$\tilde h=m\odot h/(1-p)$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 8.4 Weight decay

**Main idea.** Penalize large weights.

Core relation:

$$L+\lambda\Vert\theta\Vert^2$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 8.5 Early stopping

**Main idea.** Stop when validation loss stops improving.

Core relation:

$$L_\mathrm{val}$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
## 9. Expressivity and Generalization

This part studies expressivity and generalization as trainable representation learning. Keep track of forward values, backward gradients, scale, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Universal approximation](#9-universal-approximation) | wide nonlinear networks can approximate many functions | $f_\theta\approx f$ |
| [Depth efficiency](#9-depth-efficiency) | some functions are represented more compactly with depth | $L>1$ |
| [Overparameterization](#9-overparameterization) | large networks can fit data yet generalize with the right training biases | $P\gg n$ |
| [Double descent](#9-double-descent) | test error can be non-monotonic in model size | $E_\mathrm{test}(P)$ |
| [Inductive bias](#9-inductive-bias) | architecture and optimization shape what functions are easy to learn | $\theta_\mathrm{SGD}$ |

### 9.1 Universal approximation

**Main idea.** Wide nonlinear networks can approximate many functions.

Core relation:

$$f_\theta\approx f$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 9.2 Depth efficiency

**Main idea.** Some functions are represented more compactly with depth.

Core relation:

$$L>1$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 9.3 Overparameterization

**Main idea.** Large networks can fit data yet generalize with the right training biases.

Core relation:

$$P\gg n$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 9.4 Double descent

**Main idea.** Test error can be non-monotonic in model size.

Core relation:

$$E_\mathrm{test}(P)$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 9.5 Inductive bias

**Main idea.** Architecture and optimization shape what functions are easy to learn.

Core relation:

$$\theta_\mathrm{SGD}$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
## 10. Diagnostics

This part studies diagnostics as trainable representation learning. Keep track of forward values, backward gradients, scale, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Shape checks](#10-shape-checks) | track batch and feature axes at each layer | $(B,d_\ell)$ |
| [Activation statistics](#10-activation-statistics) | watch means, variances, and dead units | $\mu_\ell,\sigma_\ell$ |
| [Gradient norms](#10-gradient-norms) | monitor gradients by layer | $\Vert\nabla_{\theta_\ell}L\Vert$ |
| [Train validation curves](#10-train-validation-curves) | separate optimization failure from overfitting | $L_\mathrm{train},L_\mathrm{val}$ |
| [Ablations](#10-ablations) | compare width, depth, activation, optimizer, and normalization | $\Delta L$ |

### 10.1 Shape checks

**Main idea.** Track batch and feature axes at each layer.

Core relation:

$$(B,d_\ell)$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 10.2 Activation statistics

**Main idea.** Watch means, variances, and dead units.

Core relation:

$$\mu_\ell,\sigma_\ell$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 10.3 Gradient norms

**Main idea.** Monitor gradients by layer.

Core relation:

$$\Vert\nabla_{\theta_\ell}L\Vert$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** Layer-wise gradient norms are the first place to look when a network does not train.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 10.4 Train validation curves

**Main idea.** Separate optimization failure from overfitting.

Core relation:

$$L_\mathrm{train},L_\mathrm{val}$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.
### 10.5 Ablations

**Main idea.** Compare width, depth, activation, optimizer, and normalization.

Core relation:

$$\Delta L$$

A neural network is a differentiable program made from parameterized layers. The forward pass computes predictions. The backward pass uses the chain rule to assign credit to every parameter. Training works when signals and gradients stay numerically healthy across depth.

**Worked micro-example.** A two-layer network computes $h=\mathrm{ReLU}(W_1x+b_1)$ and $\hat y=W_2h+b_2$. If $W_1x+b_1$ is negative for a hidden unit, ReLU outputs zero and the local gradient through that unit is also zero for that example.

**Implementation check.** Log activation means, activation standard deviations, loss, gradient norms, and parameter update norms. A falling training loss is useful; a stable diagnostic picture is better.

**AI connection.** This is a practical neural-network control variable.

**Common mistake.** Do not debug deep networks only from the final loss. The final loss is a symptom; layer statistics often reveal the cause.

---

## Practice Exercises

1. Compute one affine layer.
2. Apply ReLU and its derivative.
3. Compute a two-layer forward pass.
4. Compute softmax cross-entropy.
5. Compute an affine-layer gradient.
6. Run a finite-difference gradient check.
7. Compare Xavier and He initialization scales.
8. Apply dropout with inverted scaling.
9. Compute LayerNorm for one example.
10. Write a neural-network debugging checklist.

## Why This Matters for AI

Transformers, CNNs, RNNs, diffusion models, and reward models are all neural networks. The same concepts repeat everywhere: differentiable computation, learned representations, chain-rule gradients, initialization, normalization, regularization, and diagnostics.

## Bridge to Probabilistic Models

The next section studies probabilistic models. Neural networks often parameterize probability distributions, so the next step is to connect learned functions with likelihoods, latent variables, and uncertainty.

## References

- Ian Goodfellow, Yoshua Bengio, and Aaron Courville, "Deep Learning", 2016: https://www.deeplearningbook.org/
- David Rumelhart, Geoffrey Hinton, and Ronald Williams, "Learning representations by back-propagating errors", Nature, 1986: https://www.nature.com/articles/323533a0
- Xavier Glorot and Yoshua Bengio, "Understanding the difficulty of training deep feedforward neural networks", 2010: https://proceedings.mlr.press/v9/glorot10a.html
- Kaiming He et al., "Delving Deep into Rectifiers", 2015: https://arxiv.org/abs/1502.01852
