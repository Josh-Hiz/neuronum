# Report: Custom Neural Network Library in C++

## Architecture Overview

```
CSV Dataset
    |
    V
Dataloader (rapidcsv)
    |
    V
 Matrix
    |
    V
  Dense
    |
    V
Layers (Dense -> Activation -> Dense -> Softmax)
    |
    V
Forward Pass
    |
    V
Loss (Cross Entropy)
    |
    V
Backward pass/ Back propagation
    |
    V
Gradient Descent
```

- Dataloader takes in the csv file and turns it into a dataframe (similar to how
  pandas works in Python). We used an external library (rapidcsv) for that. We
  also use convert the `std::vector` that it transforms it into our `matrix` class. 
- The matrix class includes addition, subtraction, multiplication, transpose.
- The network class calls the forward and backward passes for the layers and
  activation functions and computes the loss and accuracy.
- Layers are dense and activation.

## Math

### Layers

Each layer has:
```
forward()
backward()
update() // exclusive to Dense
```

#### Dense

$$Z = WX + b$$

- $W$: weight matrix
- $X$: input matrix
- $b$: bias vector
- $Z$: output matrix before activation

### Activation Functions/ Layers

#### ReLU

$$f(x) = \text{max}(0, x)$$

- derivative used in back propagation

#### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

derivative cached and used for forward pass:
$$\sigma^{\prime}(x) = \sigma(x)(1 - \sigma(x))$$

#### Softmax

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\Sum_je^{z_j}}$$

- used for multi class classification

#### Simple Gradient Descent

$$L = - \Sum y \text{log}(\hat{y})$$

- measures distance between predicted probabilities and true labels

## Limitations

- It can't handle image data. In order to make that work, we would need a
  dataloader that would support images. Currently, we use a dataloader for CSV
  parsing, which only supports numerical data only. For categorical data, we
  one-hot encoded it.

## Design Decisions

- We decided to use rapidcsv to parse the csv.
- We decided to use a `double** data` to represent the matrices in the matrix
  class instead of a `std::vector` because we did not need dynamic sizing after
  initial initialization.
- We used a `Layer` class which provided inheritance for the `Dense` and
  `Activation` classes. We used the `virtual` keyword to signify that the
  `forward()` and `backward()` pass functions were being inherited from the base
  `Layer` class. We also used an `= 0` to show that they are **pure virtual**,
  which means that the classes that inherit from the base `Layer` class must
  provide an implementation for those functions.
- We used a `std::unique_ptr` for improved memory allocation. We considered
  using a regular pointer, but 

## Future Improvements

- Extend the data loader class to support image data.
- We could use multithreading to speed up matrix mathematical calculations.
