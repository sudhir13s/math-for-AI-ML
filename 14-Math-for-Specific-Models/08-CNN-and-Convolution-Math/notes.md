[<- Generative Models](../07-Generative-Models/notes.md) | [Home](../../README.md)

---

# CNN and Convolution Math

Convolutional neural networks use local filters, shared weights, and spatial hierarchies to model images and other grid-like data efficiently.

## Overview

The core 2D cross-correlation used in deep learning is:

$$
y[i,j]=\sum_{u,v}x[i+u,j+v]w[u,v].
$$

With channels, every output channel has a kernel over all input channels. Stride, padding, and dilation determine output shape. Stacking layers grows receptive fields. Pooling and strided convolution downsample. Residual blocks and normalization make deep CNNs trainable.

## Prerequisites

- Matrix and tensor shapes
- Dot products and gradients
- Basic neural-network loss functions
- Some image or grid-data intuition

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates convolution indexing, output shapes, parameter counts, pooling, receptive fields, im2col, gradient checks, residuals, and patch embeddings. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for CNN shape and convolution arithmetic. |

## Learning Objectives

After this section, you should be able to:

- Compute 1D and 2D convolution/cross-correlation by hand.
- Calculate output sizes from kernel, padding, stride, and dilation.
- Count convolution parameters and FLOPs.
- Explain pooling, strided convolution, receptive fields, and dilation.
- Derive kernel, input, and bias gradients at a high level.
- Explain residual blocks, bottlenecks, normalization, and feature pyramids.
- Connect CNN patch operations to vision transformer patch embeddings.
- Build shape and receptive-field diagnostics for CNN models.

## Table of Contents

1. [Convolution as Local Linear Map](#1-convolution-as-local-linear-map)
   - 1.1 [Local receptive field](#11-local-receptive-field)
   - 1.2 [Weight sharing](#12-weight-sharing)
   - 1.3 [Translation equivariance](#13-translation-equivariance)
   - 1.4 [Cross-correlation convention](#14-crosscorrelation-convention)
   - 1.5 [Channels](#15-channels)
2. [Output Shape Arithmetic](#2-output-shape-arithmetic)
   - 2.1 [Stride](#21-stride)
   - 2.2 [Padding](#22-padding)
   - 2.3 [Dilation](#23-dilation)
   - 2.4 [Output height](#24-output-height)
   - 2.5 [Same convolution](#25-same-convolution)
3. [Parameter and FLOP Counts](#3-parameter-and-flop-counts)
   - 3.1 [Parameter count](#31-parameter-count)
   - 3.2 [Output elements](#32-output-elements)
   - 3.3 [FLOPs](#33-flops)
   - 3.4 [1 by 1 convolution](#34-1-by-1-convolution)
   - 3.5 [Depthwise separable convolution](#35-depthwise-separable-convolution)
4. [Pooling and Downsampling](#4-pooling-and-downsampling)
   - 4.1 [Max pooling](#41-max-pooling)
   - 4.2 [Average pooling](#42-average-pooling)
   - 4.3 [Strided convolution](#43-strided-convolution)
   - 4.4 [Global average pooling](#44-global-average-pooling)
   - 4.5 [Aliasing](#45-aliasing)
5. [Receptive Field](#5-receptive-field)
   - 5.1 [Single layer](#51-single-layer)
   - 5.2 [Stacked layers](#52-stacked-layers)
   - 5.3 [Jump](#53-jump)
   - 5.4 [Dilation effect](#54-dilation-effect)
   - 5.5 [Effective receptive field](#55-effective-receptive-field)
6. [Backpropagation Through Convolution](#6-backpropagation-through-convolution)
   - 6.1 [Kernel gradient](#61-kernel-gradient)
   - 6.2 [Input gradient](#62-input-gradient)
   - 6.3 [Bias gradient](#63-bias-gradient)
   - 6.4 [im2col view](#64-im2col-view)
   - 6.5 [Autodiff check](#65-autodiff-check)
7. [CNN Building Blocks](#7-cnn-building-blocks)
   - 7.1 [Conv activation norm](#71-conv-activation-norm)
   - 7.2 [Residual block](#72-residual-block)
   - 7.3 [Bottleneck block](#73-bottleneck-block)
   - 7.4 [Batch normalization](#74-batch-normalization)
   - 7.5 [Dropout and augmentation](#75-dropout-and-augmentation)
8. [Vision Tasks](#8-vision-tasks)
   - 8.1 [Classification](#81-classification)
   - 8.2 [Detection](#82-detection)
   - 8.3 [Segmentation](#83-segmentation)
   - 8.4 [Feature pyramids](#84-feature-pyramids)
   - 8.5 [Transfer learning](#85-transfer-learning)
9. [CNNs and Modern AI](#9-cnns-and-modern-ai)
   - 9.1 [Inductive bias](#91-inductive-bias)
   - 9.2 [Data efficiency](#92-data-efficiency)
   - 9.3 [Hybrid models](#93-hybrid-models)
   - 9.4 [Patch embedding](#94-patch-embedding)
   - 9.5 [When CNNs still matter](#95-when-cnns-still-matter)
10. [Diagnostics](#10-diagnostics)
   - 10.1 [Shape checks](#101-shape-checks)
   - 10.2 [Kernel visualization](#102-kernel-visualization)
   - 10.3 [Activation statistics](#103-activation-statistics)
   - 10.4 [Receptive field test](#104-receptive-field-test)
   - 10.5 [Ablations](#105-ablations)

---

## Shape Map

```text
image batch:      X      shape (N, C_in, H, W)
kernel bank:      W      shape (C_out, C_in, K_h, K_w)
feature map:      Y      shape (N, C_out, H_out, W_out)
classification:   logits shape (N, num_classes)
segmentation:     logits shape (N, num_classes, H, W)
```

## 1. Convolution as Local Linear Map

This part studies convolution as local linear map as spatial tensor math. Keep track of axes, output shapes, receptive fields, and parameter sharing.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Local receptive field](#1-local-receptive-field) | each output sees a small input neighborhood | $K_h\times K_w$ |
| [Weight sharing](#1-weight-sharing) | the same kernel is reused across spatial positions | $W_{u,v}$ shared |
| [Translation equivariance](#1-translation-equivariance) | shifting the input shifts the feature map | $f(T_\Delta x)=T_\Delta f(x)$ |
| [Cross-correlation convention](#1-crosscorrelation-convention) | deep learning libraries usually do not flip the kernel | $y[i,j]=\sum_{u,v}x[i+u,j+v]w[u,v]$ |
| [Channels](#1-channels) | filters mix input channels into output channels | $W\in\mathbb{R}^{C_{out}\times C_{in}\times K_h\times K_w}$ |

### 1.1 Local receptive field

**Main idea.** Each output sees a small input neighborhood.

Core relation:

$$K_h\times K_w$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 1.2 Weight sharing

**Main idea.** The same kernel is reused across spatial positions.

Core relation:

$$W_{u,v}$ shared$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 1.3 Translation equivariance

**Main idea.** Shifting the input shifts the feature map.

Core relation:

$$f(T_\Delta x)=T_\Delta f(x)$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 1.4 Cross-correlation convention

**Main idea.** Deep learning libraries usually do not flip the kernel.

Core relation:

$$y[i,j]=\sum_{u,v}x[i+u,j+v]w[u,v]$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 1.5 Channels

**Main idea.** Filters mix input channels into output channels.

Core relation:

$$W\in\mathbb{R}^{C_{out}\times C_{in}\times K_h\times K_w}$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
## 2. Output Shape Arithmetic

This part studies output shape arithmetic as spatial tensor math. Keep track of axes, output shapes, receptive fields, and parameter sharing.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Stride](#2-stride) | move the filter by more than one pixel | $S$ |
| [Padding](#2-padding) | add border values to control output size | $P$ |
| [Dilation](#2-dilation) | space kernel taps apart | $D$ |
| [Output height](#2-output-height) | compute spatial dimension after convolution | $H_{out}=\lfloor(H+2P-D(K-1)-1)/S\rfloor+1$ |
| [Same convolution](#2-same-convolution) | choose padding so output size is preserved for stride one | $P=(K-1)/2$ for odd K |

### 2.1 Stride

**Main idea.** Move the filter by more than one pixel.

Core relation:

$$S$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 2.2 Padding

**Main idea.** Add border values to control output size.

Core relation:

$$P$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 2.3 Dilation

**Main idea.** Space kernel taps apart.

Core relation:

$$D$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 2.4 Output height

**Main idea.** Compute spatial dimension after convolution.

Core relation:

$$H_{out}=\lfloor(H+2P-D(K-1)-1)/S\rfloor+1$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** Most CNN implementation bugs are shape arithmetic bugs.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 2.5 Same convolution

**Main idea.** Choose padding so output size is preserved for stride one.

Core relation:

$$P=(K-1)/2$ for odd K$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
## 3. Parameter and FLOP Counts

This part studies parameter and flop counts as spatial tensor math. Keep track of axes, output shapes, receptive fields, and parameter sharing.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Parameter count](#3-parameter-count) | kernel parameters do not depend on image size | $P=C_{out}C_{in}K_hK_w+C_{out}$ |
| [Output elements](#3-output-elements) | number of spatial positions times output channels | $C_{out}H_{out}W_{out}$ |
| [FLOPs](#3-flops) | each output element performs a dot product over channel and kernel axes | $O(C_{out}H_{out}W_{out}C_{in}K_hK_w)$ |
| [1 by 1 convolution](#3-1-by-1-convolution) | mix channels without spatial neighborhood | $K_h=K_w=1$ |
| [Depthwise separable convolution](#3-depthwise-separable-convolution) | factor spatial and channel mixing | $C_{in}K^2+C_{in}C_{out}$ |

### 3.1 Parameter count

**Main idea.** Kernel parameters do not depend on image size.

Core relation:

$$P=C_{out}C_{in}K_hK_w+C_{out}$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is why CNNs can process large images with far fewer parameters than dense layers.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 3.2 Output elements

**Main idea.** Number of spatial positions times output channels.

Core relation:

$$C_{out}H_{out}W_{out}$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 3.3 FLOPs

**Main idea.** Each output element performs a dot product over channel and kernel axes.

Core relation:

$$O(C_{out}H_{out}W_{out}C_{in}K_hK_w)$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 3.4 1 by 1 convolution

**Main idea.** Mix channels without spatial neighborhood.

Core relation:

$$K_h=K_w=1$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 3.5 Depthwise separable convolution

**Main idea.** Factor spatial and channel mixing.

Core relation:

$$C_{in}K^2+C_{in}C_{out}$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
## 4. Pooling and Downsampling

This part studies pooling and downsampling as spatial tensor math. Keep track of axes, output shapes, receptive fields, and parameter sharing.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Max pooling](#4-max-pooling) | take the largest value in a local window | $y=\max_{(u,v)\in R}x[u,v]$ |
| [Average pooling](#4-average-pooling) | average local values | $y=|R|^{-1}\sum_{(u,v)\in R}x[u,v]$ |
| [Strided convolution](#4-strided-convolution) | learned downsampling alternative | $S>1$ |
| [Global average pooling](#4-global-average-pooling) | collapse spatial dimensions into channel statistics | $z_c=(HW)^{-1}\sum_{i,j}x_{c,i,j}$ |
| [Aliasing](#4-aliasing) | downsampling without low-pass behavior can lose or distort information | $\mathrm{sample}\downarrow$ |

### 4.1 Max pooling

**Main idea.** Take the largest value in a local window.

Core relation:

$$y=\max_{(u,v)\in R}x[u,v]$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 4.2 Average pooling

**Main idea.** Average local values.

Core relation:

$$y=|R|^{-1}\sum_{(u,v)\in R}x[u,v]$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 4.3 Strided convolution

**Main idea.** Learned downsampling alternative.

Core relation:

$$S>1$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 4.4 Global average pooling

**Main idea.** Collapse spatial dimensions into channel statistics.

Core relation:

$$z_c=(HW)^{-1}\sum_{i,j}x_{c,i,j}$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 4.5 Aliasing

**Main idea.** Downsampling without low-pass behavior can lose or distort information.

Core relation:

$$\mathrm{sample}\downarrow$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
## 5. Receptive Field

This part studies receptive field as spatial tensor math. Keep track of axes, output shapes, receptive fields, and parameter sharing.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Single layer](#5-single-layer) | kernel size determines immediate field | $R=K$ |
| [Stacked layers](#5-stacked-layers) | receptive field grows through depth | $R_l=R_{l-1}+(K_l-1)J_{l-1}$ |
| [Jump](#5-jump) | effective stride between neighboring output positions | $J_l=J_{l-1}S_l$ |
| [Dilation effect](#5-dilation-effect) | dilation expands field without more parameters | $K_\mathrm{eff}=D(K-1)+1$ |
| [Effective receptive field](#5-effective-receptive-field) | learned influence is often concentrated near the center | $\partial y/\partial x$ |

### 5.1 Single layer

**Main idea.** Kernel size determines immediate field.

Core relation:

$$R=K$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 5.2 Stacked layers

**Main idea.** Receptive field grows through depth.

Core relation:

$$R_l=R_{l-1}+(K_l-1)J_{l-1}$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 5.3 Jump

**Main idea.** Effective stride between neighboring output positions.

Core relation:

$$J_l=J_{l-1}S_l$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 5.4 Dilation effect

**Main idea.** Dilation expands field without more parameters.

Core relation:

$$K_\mathrm{eff}=D(K-1)+1$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 5.5 Effective receptive field

**Main idea.** Learned influence is often concentrated near the center.

Core relation:

$$\partial y/\partial x$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
## 6. Backpropagation Through Convolution

This part studies backpropagation through convolution as spatial tensor math. Keep track of axes, output shapes, receptive fields, and parameter sharing.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Kernel gradient](#6-kernel-gradient) | sum input patches weighted by output gradients | $\partial L/\partial W=\sum \delta y\cdot x_\mathrm{patch}$ |
| [Input gradient](#6-input-gradient) | spread output gradients back through kernel taps | $\partial L/\partial x$ |
| [Bias gradient](#6-bias-gradient) | sum output gradients over batch and spatial axes | $\partial L/\partial b_c=\sum_{n,i,j}\delta y_{n,c,i,j}$ |
| [im2col view](#6-im2col-view) | convolution can be lowered to matrix multiplication | $Y=W_\mathrm{mat}X_\mathrm{col}$ |
| [Autodiff check](#6-autodiff-check) | finite differences can verify small convolution gradients | $\frac{L(W+\epsilon)-L(W-\epsilon)}{2\epsilon}$ |

### 6.1 Kernel gradient

**Main idea.** Sum input patches weighted by output gradients.

Core relation:

$$\partial L/\partial W=\sum \delta y\cdot x_\mathrm{patch}$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 6.2 Input gradient

**Main idea.** Spread output gradients back through kernel taps.

Core relation:

$$\partial L/\partial x$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 6.3 Bias gradient

**Main idea.** Sum output gradients over batch and spatial axes.

Core relation:

$$\partial L/\partial b_c=\sum_{n,i,j}\delta y_{n,c,i,j}$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 6.4 im2col view

**Main idea.** Convolution can be lowered to matrix multiplication.

Core relation:

$$Y=W_\mathrm{mat}X_\mathrm{col}$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 6.5 Autodiff check

**Main idea.** Finite differences can verify small convolution gradients.

Core relation:

$$\frac{L(W+\epsilon)-L(W-\epsilon)}{2\epsilon}$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
## 7. CNN Building Blocks

This part studies cnn building blocks as spatial tensor math. Keep track of axes, output shapes, receptive fields, and parameter sharing.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Conv activation norm](#7-conv-activation-norm) | standard block combines convolution, nonlinearity, and normalization | $\mathrm{Norm}(\phi(\mathrm{Conv}(x)))$ |
| [Residual block](#7-residual-block) | learn a correction around identity | $y=x+F(x)$ |
| [Bottleneck block](#7-bottleneck-block) | use 1 by 1 convolutions to reduce and restore channels | $1\times1\rightarrow3\times3\rightarrow1\times1$ |
| [Batch normalization](#7-batch-normalization) | normalize channel statistics over batch and spatial axes | $\hat x=(x-\mu)/\sqrt{\sigma^2+\epsilon}$ |
| [Dropout and augmentation](#7-dropout-and-augmentation) | regularize feature learning | $\tilde x=A(x)$ |

### 7.1 Conv activation norm

**Main idea.** Standard block combines convolution, nonlinearity, and normalization.

Core relation:

$$\mathrm{Norm}(\phi(\mathrm{Conv}(x)))$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 7.2 Residual block

**Main idea.** Learn a correction around identity.

Core relation:

$$y=x+F(x)$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** Residual connections made very deep CNNs practical by preserving an identity path.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 7.3 Bottleneck block

**Main idea.** Use 1 by 1 convolutions to reduce and restore channels.

Core relation:

$$1\times1\rightarrow3\times3\rightarrow1\times1$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 7.4 Batch normalization

**Main idea.** Normalize channel statistics over batch and spatial axes.

Core relation:

$$\hat x=(x-\mu)/\sqrt{\sigma^2+\epsilon}$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 7.5 Dropout and augmentation

**Main idea.** Regularize feature learning.

Core relation:

$$\tilde x=A(x)$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
## 8. Vision Tasks

This part studies vision tasks as spatial tensor math. Keep track of axes, output shapes, receptive fields, and parameter sharing.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Classification](#8-classification) | map image features to class logits | $p(y\mid x)=\mathrm{softmax}(Wh+b)$ |
| [Detection](#8-detection) | predict boxes and classes over spatial anchors or queries | $(b_i,c_i)$ |
| [Segmentation](#8-segmentation) | predict a class for each pixel | $p(y_{i,j}\mid x)$ |
| [Feature pyramids](#8-feature-pyramids) | combine multi-scale feature maps | $F_1,\ldots,F_L$ |
| [Transfer learning](#8-transfer-learning) | reuse pretrained convolutional features | $\theta=\theta_0+\Delta\theta$ |

### 8.1 Classification

**Main idea.** Map image features to class logits.

Core relation:

$$p(y\mid x)=\mathrm{softmax}(Wh+b)$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 8.2 Detection

**Main idea.** Predict boxes and classes over spatial anchors or queries.

Core relation:

$$(b_i,c_i)$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 8.3 Segmentation

**Main idea.** Predict a class for each pixel.

Core relation:

$$p(y_{i,j}\mid x)$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 8.4 Feature pyramids

**Main idea.** Combine multi-scale feature maps.

Core relation:

$$F_1,\ldots,F_L$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 8.5 Transfer learning

**Main idea.** Reuse pretrained convolutional features.

Core relation:

$$\theta=\theta_0+\Delta\theta$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
## 9. CNNs and Modern AI

This part studies cnns and modern ai as spatial tensor math. Keep track of axes, output shapes, receptive fields, and parameter sharing.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Inductive bias](#9-inductive-bias) | locality and translation equivariance suit images | $x_{i,j}$ neighborhoods |
| [Data efficiency](#9-data-efficiency) | weight sharing reduces sample complexity compared with dense layers | $P_\mathrm{conv}\ll P_\mathrm{dense}$ |
| [Hybrid models](#9-hybrid-models) | modern vision systems often mix CNNs and attention | $\mathrm{Conv}+\mathrm{Attention}$ |
| [Patch embedding](#9-patch-embedding) | ViT patch projection is a strided convolution view | $p=W\mathrm{vec}(\mathrm{patch})$ |
| [When CNNs still matter](#9-when-cnns-still-matter) | edge vision and dense prediction often benefit from convolutional efficiency | $\mathrm{latency},\mathrm{memory}$ |

### 9.1 Inductive bias

**Main idea.** Locality and translation equivariance suit images.

Core relation:

$$x_{i,j}$ neighborhoods$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 9.2 Data efficiency

**Main idea.** Weight sharing reduces sample complexity compared with dense layers.

Core relation:

$$P_\mathrm{conv}\ll P_\mathrm{dense}$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 9.3 Hybrid models

**Main idea.** Modern vision systems often mix cnns and attention.

Core relation:

$$\mathrm{Conv}+\mathrm{Attention}$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 9.4 Patch embedding

**Main idea.** Vit patch projection is a strided convolution view.

Core relation:

$$p=W\mathrm{vec}(\mathrm{patch})$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This connects classical convolution math to modern vision transformers.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 9.5 When CNNs still matter

**Main idea.** Edge vision and dense prediction often benefit from convolutional efficiency.

Core relation:

$$\mathrm{latency},\mathrm{memory}$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
## 10. Diagnostics

This part studies diagnostics as spatial tensor math. Keep track of axes, output shapes, receptive fields, and parameter sharing.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Shape checks](#10-shape-checks) | track batch, channel, height, and width explicitly | $(N,C,H,W)$ |
| [Kernel visualization](#10-kernel-visualization) | inspect early filters and feature maps | $W_{c,:,:}$ |
| [Activation statistics](#10-activation-statistics) | dead channels or saturated activations reveal training issues | $\mu_c,\sigma_c$ |
| [Receptive field test](#10-receptive-field-test) | verify output depends on intended input region | $\partial y/\partial x$ |
| [Ablations](#10-ablations) | compare kernel size, stride, depthwise factorization, residuals, and normalization | $\Delta S,\Delta T$ |

### 10.1 Shape checks

**Main idea.** Track batch, channel, height, and width explicitly.

Core relation:

$$(N,C,H,W)$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 10.2 Kernel visualization

**Main idea.** Inspect early filters and feature maps.

Core relation:

$$W_{c,:,:}$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 10.3 Activation statistics

**Main idea.** Dead channels or saturated activations reveal training issues.

Core relation:

$$\mu_c,\sigma_c$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 10.4 Receptive field test

**Main idea.** Verify output depends on intended input region.

Core relation:

$$\partial y/\partial x$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** A receptive-field check catches accidental padding, stride, or dilation mistakes.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.
### 10.5 Ablations

**Main idea.** Compare kernel size, stride, depthwise factorization, residuals, and normalization.

Core relation:

$$\Delta S,\Delta T$$

Convolutional networks are built from local linear maps with shared weights. The math is simple but unforgiving: every padding, stride, dilation, and channel convention changes the output shape and the information path through the model.

**Worked micro-example.** A 3 by 3 convolution from 64 input channels to 128 output channels has $128\cdot64\cdot3\cdot3=73,728$ weights, independent of image height and width. A dense layer over a 224 by 224 image would scale with every pixel location.

**Implementation check.** For every layer, write the tensor shape before and after. Verify output size with the shape formula before debugging the model logic.

**AI connection.** This is a practical convolutional-model control variable.

**Common mistake.** Do not confuse mathematical convolution with the cross-correlation operation used by most deep learning libraries. The learned kernel adapts either way, but the indexing convention matters for hand calculations.

---

## Practice Exercises

1. Compute a 1D cross-correlation output.
2. Compute output size from kernel, padding, stride, and dilation.
3. Count convolution parameters.
4. Compare dense and convolutional parameter counts.
5. Compute max pooling and average pooling.
6. Compute receptive field through stacked layers.
7. Compute depthwise separable convolution parameters.
8. Build an im2col matrix for a tiny input.
9. Compute a residual block output.
10. Write a CNN debugging checklist.

## Why This Matters for AI

Even in an LLM-heavy world, convolution math remains central for vision, audio, time series, edge models, segmentation, detection, multimodal encoders, and efficient local feature extraction. CNNs also teach key design principles: locality, weight sharing, equivariance, receptive fields, and hierarchy.

## Bridge Forward

This completes the model-specific chapter arc: dense models, neural networks, probabilistic models, RNNs, transformers, reinforcement learning, generative models, and CNNs. The same accounting skills reappear in modern multimodal LLMs.

## References

- Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner, "Gradient-Based Learning Applied to Document Recognition", 1998: https://doi.org/10.1109/5.726791
- Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", 2012: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
- Kaiming He et al., "Deep Residual Learning for Image Recognition", 2015: https://arxiv.org/abs/1512.03385
- Stanford CS231n, "Convolutional Neural Networks": https://cs231n.github.io/convolutional-networks/
