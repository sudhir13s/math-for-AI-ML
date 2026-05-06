[Home](../../README.md) | [Neural Networks ->](../02-Neural-Networks/notes.md)

---

# Linear Models

Linear models are the first serious model family in machine learning: simple enough to solve and inspect, strong enough to be useful, and foundational for understanding optimization, regularization, classification, and modern neural network heads.

## Overview

The basic prediction is:

$$
\hat y = w^\top x+b.
$$

For a design matrix $X$, all predictions are:

$$
\hat y=Xw+b\mathbf{1}.
$$

From this one form we get least squares, ridge regression, lasso, logistic regression, softmax regression, linear probes, and the final LM head of a transformer.

## Prerequisites

- Vectors, matrices, dot products, and norms
- Gradients and convex optimization basics
- Probability and cross-entropy for classification
- Train/validation evaluation vocabulary

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates least squares, gradient descent, ridge, lasso-style shrinkage intuition, logistic regression, calibration, SVD conditioning, and linear probes. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for closed-form solves, gradients, regularization, classification probabilities, and diagnostics. |

## Learning Objectives

After this section, you should be able to:

- Write linear regression and classification models in vectorized form.
- Derive the least-squares normal equations.
- Explain projection, pseudoinverse, rank, and conditioning.
- Implement gradient descent and check its gradient.
- Explain ridge, lasso, elastic net, and the bias-variance tradeoff.
- Compute logistic and softmax probabilities.
- Interpret linear probes and LM heads in modern AI.
- Build a diagnostic checklist for linear models.

## Table of Contents

1. [Linear Prediction](#1-linear-prediction)
   - 1.1 [Feature vector](#11-feature-vector)
   - 1.2 [Affine score](#12-affine-score)
   - 1.3 [Design matrix](#13-design-matrix)
   - 1.4 [Vectorized prediction](#14-vectorized-prediction)
   - 1.5 [Linear decision boundary](#15-linear-decision-boundary)
2. [Least Squares Regression](#2-least-squares-regression)
   - 2.1 [Residuals](#21-residuals)
   - 2.2 [Squared loss](#22-squared-loss)
   - 2.3 [Normal equations](#23-normal-equations)
   - 2.4 [Projection view](#24-projection-view)
   - 2.5 [Pseudoinverse](#25-pseudoinverse)
3. [Gradient Descent](#3-gradient-descent)
   - 3.1 [Gradient](#31-gradient)
   - 3.2 [Update](#32-update)
   - 3.3 [Learning rate](#33-learning-rate)
   - 3.4 [Stochastic gradients](#34-stochastic-gradients)
   - 3.5 [Feature scaling](#35-feature-scaling)
4. [Regularization](#4-regularization)
   - 4.1 [Ridge](#41-ridge)
   - 4.2 [Ridge solution](#42-ridge-solution)
   - 4.3 [Lasso](#43-lasso)
   - 4.4 [Elastic net](#44-elastic-net)
   - 4.5 [Bias variance](#45-bias-variance)
5. [Linear Classification](#5-linear-classification)
   - 5.1 [Logistic regression](#51-logistic-regression)
   - 5.2 [Binary cross-entropy](#52-binary-crossentropy)
   - 5.3 [Softmax regression](#53-softmax-regression)
   - 5.4 [Margin](#54-margin)
   - 5.5 [Linear separability](#55-linear-separability)
6. [Optimization Geometry](#6-optimization-geometry)
   - 6.1 [Convexity](#61-convexity)
   - 6.2 [Condition number](#62-condition-number)
   - 6.3 [SVD view](#63-svd-view)
   - 6.4 [Collinearity](#64-collinearity)
   - 6.5 [Whitening](#65-whitening)
7. [Evaluation](#7-evaluation)
   - 7.1 [Train validation split](#71-train-validation-split)
   - 7.2 [MSE and MAE](#72-mse-and-mae)
   - 7.3 [Accuracy and log loss](#73-accuracy-and-log-loss)
   - 7.4 [Calibration](#74-calibration)
   - 7.5 [Cross-validation](#75-crossvalidation)
8. [Linear Models in AI](#8-linear-models-in-ai)
   - 8.1 [Baseline strength](#81-baseline-strength)
   - 8.2 [Linear probes](#82-linear-probes)
   - 8.3 [LM head](#83-lm-head)
   - 8.4 [Logit lens](#84-logit-lens)
   - 8.5 [Interpretability](#85-interpretability)
9. [Implementation Details](#9-implementation-details)
   - 9.1 [Intercept handling](#91-intercept-handling)
   - 9.2 [Numerical solve](#92-numerical-solve)
   - 9.3 [Standardization leakage](#93-standardization-leakage)
   - 9.4 [Rank checks](#94-rank-checks)
   - 9.5 [Residual diagnostics](#95-residual-diagnostics)
10. [Diagnostics](#10-diagnostics)
   - 10.1 [Shape checks](#101-shape-checks)
   - 10.2 [Gradient check](#102-gradient-check)
   - 10.3 [Regularization path](#103-regularization-path)
   - 10.4 [Influence](#104-influence)
   - 10.5 [Baseline comparison](#105-baseline-comparison)

---

## Shape Map

```text
features:       X       shape (n, d)
targets:        y       shape (n,)
weights:        w       shape (d,)
predictions:    y_hat   shape (n,)
multi-class W:  W       shape (K, d)
logits:         Z       shape (n, K)
```

## 1. Linear Prediction

This part studies linear prediction as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Feature vector](#1-feature-vector) | represent each example as coordinates | $x\in\mathbb{R}^d$ |
| [Affine score](#1-affine-score) | combine features by weighted sum plus bias | $\hat y=w^\top x+b$ |
| [Design matrix](#1-design-matrix) | stack examples row-wise | $X\in\mathbb{R}^{n\times d}$ |
| [Vectorized prediction](#1-vectorized-prediction) | predict all examples at once | $\hat y=Xw+b\mathbf{1}$ |
| [Linear decision boundary](#1-linear-decision-boundary) | classification threshold creates a hyperplane | $w^\top x+b=0$ |

### 1.1 Feature vector

**Main idea.** Represent each example as coordinates.

Core relation:

$$x\in\mathbb{R}^d$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 1.2 Affine score

**Main idea.** Combine features by weighted sum plus bias.

Core relation:

$$\hat y=w^\top x+b$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 1.3 Design matrix

**Main idea.** Stack examples row-wise.

Core relation:

$$X\in\mathbb{R}^{n\times d}$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 1.4 Vectorized prediction

**Main idea.** Predict all examples at once.

Core relation:

$$\hat y=Xw+b\mathbf{1}$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 1.5 Linear decision boundary

**Main idea.** Classification threshold creates a hyperplane.

Core relation:

$$w^\top x+b=0$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 2. Least Squares Regression

This part studies least squares regression as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Residuals](#2-residuals) | prediction errors form a vector | $r=y-Xw$ |
| [Squared loss](#2-squared-loss) | penalize large residuals quadratically | $L(w)=\frac12\Vert Xw-y\Vert_2^2$ |
| [Normal equations](#2-normal-equations) | set the gradient to zero | $X^\top Xw=X^\top y$ |
| [Projection view](#2-projection-view) | least squares projects y onto the column space of X | $\hat y=P_Xy$ |
| [Pseudoinverse](#2-pseudoinverse) | handle rank-deficient or rectangular systems | $w=X^+y$ |

### 2.1 Residuals

**Main idea.** Prediction errors form a vector.

Core relation:

$$r=y-Xw$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 2.2 Squared loss

**Main idea.** Penalize large residuals quadratically.

Core relation:

$$L(w)=\frac12\Vert Xw-y\Vert_2^2$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 2.3 Normal equations

**Main idea.** Set the gradient to zero.

Core relation:

$$X^\top Xw=X^\top y$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is the closed-form anchor for least-squares learning.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 2.4 Projection view

**Main idea.** Least squares projects y onto the column space of x.

Core relation:

$$\hat y=P_Xy$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 2.5 Pseudoinverse

**Main idea.** Handle rank-deficient or rectangular systems.

Core relation:

$$w=X^+y$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 3. Gradient Descent

This part studies gradient descent as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Gradient](#3-gradient) | loss gradient points uphill in parameter space | $\nabla_w L=X^\top(Xw-y)$ |
| [Update](#3-update) | move opposite the gradient | $w_{t+1}=w_t-\eta\nabla_wL$ |
| [Learning rate](#3-learning-rate) | step size must respect curvature | $0<\eta<2/\lambda_\max(X^\top X)$ |
| [Stochastic gradients](#3-stochastic-gradients) | estimate gradient from mini-batches | $g_B=X_B^\top(X_Bw-y_B)$ |
| [Feature scaling](#3-feature-scaling) | rescale features to improve conditioning | $x_j\leftarrow(x_j-\mu_j)/\sigma_j$ |

### 3.1 Gradient

**Main idea.** Loss gradient points uphill in parameter space.

Core relation:

$$\nabla_w L=X^\top(Xw-y)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 3.2 Update

**Main idea.** Move opposite the gradient.

Core relation:

$$w_{t+1}=w_t-\eta\nabla_wL$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 3.3 Learning rate

**Main idea.** Step size must respect curvature.

Core relation:

$$0<\eta<2/\lambda_\max(X^\top X)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 3.4 Stochastic gradients

**Main idea.** Estimate gradient from mini-batches.

Core relation:

$$g_B=X_B^\top(X_Bw-y_B)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 3.5 Feature scaling

**Main idea.** Rescale features to improve conditioning.

Core relation:

$$x_j\leftarrow(x_j-\mu_j)/\sigma_j$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 4. Regularization

This part studies regularization as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Ridge](#4-ridge) | penalize squared parameter norm | $L=\frac12\Vert Xw-y\Vert^2+\frac{\lambda}{2}\Vert w\Vert^2$ |
| [Ridge solution](#4-ridge-solution) | shift the Gram matrix spectrum | $w=(X^\top X+\lambda I)^{-1}X^\top y$ |
| [Lasso](#4-lasso) | penalize absolute values to encourage sparsity | $L=\frac12\Vert Xw-y\Vert^2+\lambda\Vert w\Vert_1$ |
| [Elastic net](#4-elastic-net) | combine L1 sparsity and L2 shrinkage | $\lambda_1\Vert w\Vert_1+\lambda_2\Vert w\Vert_2^2$ |
| [Bias variance](#4-bias-variance) | regularization trades bias for lower variance | $E[(\hat f-f)^2]=\mathrm{bias}^2+\mathrm{var}+\sigma^2$ |

### 4.1 Ridge

**Main idea.** Penalize squared parameter norm.

Core relation:

$$L=\frac12\Vert Xw-y\Vert^2+\frac{\lambda}{2}\Vert w\Vert^2$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 4.2 Ridge solution

**Main idea.** Shift the gram matrix spectrum.

Core relation:

$$w=(X^\top X+\lambda I)^{-1}X^\top y$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** Ridge is one of the cleanest examples of regularization improving numerical stability.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 4.3 Lasso

**Main idea.** Penalize absolute values to encourage sparsity.

Core relation:

$$L=\frac12\Vert Xw-y\Vert^2+\lambda\Vert w\Vert_1$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 4.4 Elastic net

**Main idea.** Combine l1 sparsity and l2 shrinkage.

Core relation:

$$\lambda_1\Vert w\Vert_1+\lambda_2\Vert w\Vert_2^2$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 4.5 Bias variance

**Main idea.** Regularization trades bias for lower variance.

Core relation:

$$E[(\hat f-f)^2]=\mathrm{bias}^2+\mathrm{var}+\sigma^2$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 5. Linear Classification

This part studies linear classification as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Logistic regression](#5-logistic-regression) | map affine score to probability | $p(y=1\mid x)=\sigma(w^\top x+b)$ |
| [Binary cross-entropy](#5-binary-crossentropy) | negative Bernoulli log likelihood | $-\left[y\log p+(1-y)\log(1-p)\right]$ |
| [Softmax regression](#5-softmax-regression) | multi-class linear logits | $p_k=\exp(w_k^\top x)/\sum_j\exp(w_j^\top x)$ |
| [Margin](#5-margin) | signed distance proxy for confidence | $y(w^\top x+b)$ |
| [Linear separability](#5-linear-separability) | a hyperplane can perfectly split classes only in some feature spaces | $y_i(w^\top x_i+b)>0$ |

### 5.1 Logistic regression

**Main idea.** Map affine score to probability.

Core relation:

$$p(y=1\mid x)=\sigma(w^\top x+b)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 5.2 Binary cross-entropy

**Main idea.** Negative bernoulli log likelihood.

Core relation:

$$-\left[y\log p+(1-y)\log(1-p)\right]$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 5.3 Softmax regression

**Main idea.** Multi-class linear logits.

Core relation:

$$p_k=\exp(w_k^\top x)/\sum_j\exp(w_j^\top x)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 5.4 Margin

**Main idea.** Signed distance proxy for confidence.

Core relation:

$$y(w^\top x+b)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 5.5 Linear separability

**Main idea.** A hyperplane can perfectly split classes only in some feature spaces.

Core relation:

$$y_i(w^\top x_i+b)>0$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 6. Optimization Geometry

This part studies optimization geometry as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Convexity](#6-convexity) | linear regression and logistic regression losses are convex in w | $\nabla^2L\succeq0$ |
| [Condition number](#6-condition-number) | ill-conditioned features slow optimization | $\kappa=\lambda_\max/\lambda_\min$ |
| [SVD view](#6-svd-view) | singular values reveal stable and unstable directions | $X=U\Sigma V^\top$ |
| [Collinearity](#6-collinearity) | correlated features make coefficients unstable | $X^\top X$ nearly singular |
| [Whitening](#6-whitening) | decorrelate features when appropriate | $\mathrm{Cov}(X)\approx I$ |

### 6.1 Convexity

**Main idea.** Linear regression and logistic regression losses are convex in w.

Core relation:

$$\nabla^2L\succeq0$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 6.2 Condition number

**Main idea.** Ill-conditioned features slow optimization.

Core relation:

$$\kappa=\lambda_\max/\lambda_\min$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 6.3 SVD view

**Main idea.** Singular values reveal stable and unstable directions.

Core relation:

$$X=U\Sigma V^\top$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 6.4 Collinearity

**Main idea.** Correlated features make coefficients unstable.

Core relation:

$$X^\top X$ nearly singular$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 6.5 Whitening

**Main idea.** Decorrelate features when appropriate.

Core relation:

$$\mathrm{Cov}(X)\approx I$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 7. Evaluation

This part studies evaluation as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Train validation split](#7-train-validation-split) | measure generalization on held-out data | $L_\mathrm{val}$ |
| [MSE and MAE](#7-mse-and-mae) | regression errors emphasize different residual behavior | $\mathrm{MSE}=n^{-1}\sum r_i^2$ |
| [Accuracy and log loss](#7-accuracy-and-log-loss) | classification quality needs both decisions and probabilities | $-\log p_y$ |
| [Calibration](#7-calibration) | predicted probabilities should match empirical frequencies | $P(y=1\mid \hat p=c)\approx c$ |
| [Cross-validation](#7-crossvalidation) | average validation over multiple folds | $K$ folds |

### 7.1 Train validation split

**Main idea.** Measure generalization on held-out data.

Core relation:

$$L_\mathrm{val}$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 7.2 MSE and MAE

**Main idea.** Regression errors emphasize different residual behavior.

Core relation:

$$\mathrm{MSE}=n^{-1}\sum r_i^2$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 7.3 Accuracy and log loss

**Main idea.** Classification quality needs both decisions and probabilities.

Core relation:

$$-\log p_y$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 7.4 Calibration

**Main idea.** Predicted probabilities should match empirical frequencies.

Core relation:

$$P(y=1\mid \hat p=c)\approx c$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 7.5 Cross-validation

**Main idea.** Average validation over multiple folds.

Core relation:

$$K$ folds$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 8. Linear Models in AI

This part studies linear models in ai as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Baseline strength](#8-baseline-strength) | linear models expose whether features already solve the task | $\hat y=Xw$ |
| [Linear probes](#8-linear-probes) | frozen representations can be tested by a linear head | $h=f_{\theta_0}(x),\ \hat y=Wh$ |
| [LM head](#8-lm-head) | language models end with a linear projection to vocabulary logits | $z=hW_E^\top$ |
| [Logit lens](#8-logit-lens) | intermediate states can be projected with the LM head | $z_\ell=h_\ell W_E^\top$ |
| [Interpretability](#8-interpretability) | linear weights are easier to inspect than deep nonlinear interactions | $w_j$ feature effect |

### 8.1 Baseline strength

**Main idea.** Linear models expose whether features already solve the task.

Core relation:

$$\hat y=Xw$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 8.2 Linear probes

**Main idea.** Frozen representations can be tested by a linear head.

Core relation:

$$h=f_{\theta_0}(x),\ \hat y=Wh$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** Linear probes are a practical way to test what a deep representation already contains.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 8.3 LM head

**Main idea.** Language models end with a linear projection to vocabulary logits.

Core relation:

$$z=hW_E^\top$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** Even the final step of a giant language model is a linear model over hidden features.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 8.4 Logit lens

**Main idea.** Intermediate states can be projected with the lm head.

Core relation:

$$z_\ell=h_\ell W_E^\top$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 8.5 Interpretability

**Main idea.** Linear weights are easier to inspect than deep nonlinear interactions.

Core relation:

$$w_j$ feature effect$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 9. Implementation Details

This part studies implementation details as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Intercept handling](#9-intercept-handling) | bias can be a separate scalar or a column of ones | $\tilde X=[X,\mathbf{1}]$ |
| [Numerical solve](#9-numerical-solve) | prefer stable solvers over explicit matrix inverse | $\mathrm{solve}(X^\top X,X^\top y)$ |
| [Standardization leakage](#9-standardization-leakage) | fit scaling parameters on train data only | $\mu_\mathrm{train},\sigma_\mathrm{train}$ |
| [Rank checks](#9-rank-checks) | inspect rank before trusting coefficients | $\mathrm{rank}(X)$ |
| [Residual diagnostics](#9-residual-diagnostics) | plot residuals to find nonlinearity or outliers | $r_i=y_i-\hat y_i$ |

### 9.1 Intercept handling

**Main idea.** Bias can be a separate scalar or a column of ones.

Core relation:

$$\tilde X=[X,\mathbf{1}]$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 9.2 Numerical solve

**Main idea.** Prefer stable solvers over explicit matrix inverse.

Core relation:

$$\mathrm{solve}(X^\top X,X^\top y)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 9.3 Standardization leakage

**Main idea.** Fit scaling parameters on train data only.

Core relation:

$$\mu_\mathrm{train},\sigma_\mathrm{train}$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 9.4 Rank checks

**Main idea.** Inspect rank before trusting coefficients.

Core relation:

$$\mathrm{rank}(X)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 9.5 Residual diagnostics

**Main idea.** Plot residuals to find nonlinearity or outliers.

Core relation:

$$r_i=y_i-\hat y_i$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 10. Diagnostics

This part studies diagnostics as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Shape checks](#10-shape-checks) | confirm matrix dimensions before fitting | $X:(n,d),\ w:(d,)$ |
| [Gradient check](#10-gradient-check) | compare analytic and finite-difference gradients | $\nabla L$ |
| [Regularization path](#10-regularization-path) | plot coefficients as lambda changes | $w(\lambda)$ |
| [Influence](#10-influence) | outliers can dominate squared loss | $r_i^2$ |
| [Baseline comparison](#10-baseline-comparison) | compare linear, regularized, and nonlinear models | $\Delta L$ |

### 10.1 Shape checks

**Main idea.** Confirm matrix dimensions before fitting.

Core relation:

$$X:(n,d),\ w:(d,)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 10.2 Gradient check

**Main idea.** Compare analytic and finite-difference gradients.

Core relation:

$$\nabla L$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** Linear models are perfect for learning how to verify optimization code.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 10.3 Regularization path

**Main idea.** Plot coefficients as lambda changes.

Core relation:

$$w(\lambda)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 10.4 Influence

**Main idea.** Outliers can dominate squared loss.

Core relation:

$$r_i^2$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 10.5 Baseline comparison

**Main idea.** Compare linear, regularized, and nonlinear models.

Core relation:

$$\Delta L$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.

---

## Practice Exercises

1. Compute an affine prediction.
2. Solve a small least-squares problem.
3. Derive and compute the squared-loss gradient.
4. Run a few gradient-descent steps.
5. Compute a ridge solution.
6. Compare coefficient shrinkage under different lambdas.
7. Compute logistic probability and binary cross-entropy.
8. Compute softmax probabilities.
9. Check feature standardization and leakage.
10. Write a linear-model debugging checklist.

## Why This Matters for AI

Linear models appear inside modern AI more often than they first seem. A classifier head is linear. A language model head is linear. A linear probe tests representation quality. Ridge and lasso teach regularization. Least squares teaches projection geometry. Logistic regression teaches probabilistic classification and cross-entropy.

## Bridge to Neural Networks

Neural networks can be viewed as learned feature maps followed by linear heads. The next section studies what changes when the feature map itself becomes trainable and nonlinear.

## References

- Trevor Hastie, Robert Tibshirani, and Jerome Friedman, "The Elements of Statistical Learning", 2nd ed.: https://web.stanford.edu/~hastie/ElemStatLearn/
- Christopher Bishop, "Pattern Recognition and Machine Learning", 2006.
- Arthur Hoerl and Robert Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems", Technometrics, 1970: https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634
- Robert Tibshirani, "Regression Shrinkage and Selection via the Lasso", 1996: https://academic.oup.com/jrsssb/article/58/1/267/7027929
