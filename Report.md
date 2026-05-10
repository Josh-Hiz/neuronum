# Report: Custom Neural Network Library in C++

## 1. Architecture Overview

`neuronum` is a simple C++17 neural network library for classification on numerical CSV datasets. The main data flow is:

```text
CSV Dataset
    |
    V
Dataloader
    |
    V
Matrix
    |
    V
Dense and Activation Layers
    |
    V
Forward Pass
    |
    V
Loss Calculation
    |
    V
Backward Pass
    |
    V
Gradient Descent Update
```

- The `DataLoader` does CSV parsing and preprocessing. It uses `rapidcsv` to read columns from a CSV file, separates features from the label column, ignores user-specified columns, converts labels into one-hot encoded matrices, normalizes feature columns, shuffles the data, and splits it into train/test matrices.

- The `Matrix` class does basic linear algebra operations. It stores data using `double**` and implements matrix addition, subtraction, multiplication, scalar multiplication, transpose, and bias broadcasting.

The `Network` class stores layers in a `std::vector<std::unique_ptr<Layer>>`, runs each layer in order during the forward pass, runs layers in reverse order during backpropagation, updates trainable parameters, and computes loss and accuracy.

`Dense` layers have trainable weights and biases, while activation layers such as ReLU, Sigmoid, and Softmax transform values without trainable parameters. The typical architecture is:

```
Dense -> ReLU -> Dense -> Softmax
```

## 2. Mathematical Methods

The network is a fully connected neural network trained with full-batch gradient descent.

The data loader normalizes each feature column using mean and standard deviation:

$$
x' = \frac{x - \mu}{\sigma}
$$

Each dense layer computes the transformation:

$$
Z = WX + b
$$

where $X$ is the input matrix, $W$ is the weight matrix, $b$ is the bias vector, and $Z$ is the pre-activation output. In this implementation, training data is transposed before entering the network, so each column is one sample:

$$
X \in \mathbb{R}^{\text{input size} \times m}
$$

$$
W \in \mathbb{R}^{\text{output size} \times \text{input size}}
$$

$$
Z \in \mathbb{R}^{\text{output size} \times m}
$$

The dense layer computes `W * input` and then adds the bias using broadcast addition.

The ReLU activation is:

$$
f(x) = \max(0, x)
$$

Its derivative is:

$$
f'(x) =
\begin{cases}
1, & x > 0 \
0, & x \leq 0
\end{cases}
$$

The implementation caches the input and sets the backward gradient to zero wherever the original input was non-positive.

The sigmoid activation is:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

with derivative:

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

The implementation caches the sigmoid output so the derivative can be computed efficiently during backpropagation.

The output layer uses softmax for multi-class classification:

$$
\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Softmax converts raw scores into probabilities that sum to 1, which is why it is commonly paired with cross-entropy loss for classification.

The loss function is categorical cross-entropy:

$$
L = -\frac{1}{m}\sum_{j=1}^{m}\sum_i y_{ij}\log(\hat{y}_{ij})
$$

The implementation only adds loss for the true class and divides by the number of samples. It also adds a small constant, $10^{-7}$, inside the log to avoid taking $\log(0)$.

Backpropagation begins with the simplified gradient for softmax plus cross-entropy:

$$
dZ = \hat{Y} - Y
$$

Then each dense layer computes:

$$
dW = dZ X^T
$$

$$
db = \sum dZ
$$

$$
dX = W^T dZ
$$

Finally, parameters are updated using gradient descent:

$$
W = W - \alpha dW
$$

$$
b = b - \alpha db
$$

where $\alpha$ is the learning rate. The dense layer implements these gradient and update equations directly.

The weights are initialized from a normal distribution with scale:

$$
\sqrt{\frac{2}{\text{input size}}}
$$

This is called Xavier initialization, and is a standard way of initializing weights, its also the simplest way of doing so, which is why we did it for our project.

## 3. Key Design Decisions and Trade-offs

- The first major design decision was using `rapidcsv` instead of writing a custom CSV parser. This greatly simplified our project as we didn't need to implement our own CSV parser where we need to consider manual data-type conversion. The trade-off is that the project depends on a third-party parser and mainly supports numerical CSV data.

- The second design decision was using inheritance for layers. A common `Layer` interface makes it easy for the `Network` to store `Dense`, `ReLU`, `Sigmoid`, and `Softmax` layers together and call `forward()` and `backward()` polymorphically.

- We also used the `virtual` keyword to signify that the `forward()` and `backward()` pass functions were being inherited from the base `Layer` class. We also used an `= 0` to show that they are **pure virtual**, which means that the classes that inherit from the base `Layer` class must provide an implementation for those functions.

- We decided to use a `double** data` to represent the matrices in the matrix class instead of a `std::vector` because we did not need dynamic sizing after initial initialization. The size of the matricies we are using are predictable, thus we can avoid having to use vectors which have dynamic reallocation overhead.

- The fourth design decision was storing layers with `std::unique_ptr`. We decided to use unique pointer for layers because due to our `Layer` class being polymorphic, we need to dynamically allocate different types of layers so the correct function is called via the virtual table. So in order to avoid any complication with manually freeing this data, we wrapped these raw pointers around a `std::unique_ptr` that will handle the memory deallocation for us.

## 4. Improvements Given More Time

- The first improvement would be adding mini-batch training. Full-batch training is fine for small datasets, but mini-batches would allow to process larger datasets faster.

- Other useful improvements would include Adam or momentum-based optimization, model saving/loading, better categorical feature support, numerical gradient checking, and unit tests for matrix operations, layer gradients, and loss calculations.

- Extend the data loader class to support image data, this is a very large increase in complexity and out-of-scope for this current project.

- We could use multithreading to speed up matrix mathematical calculations for matrix operations as these are highly parallelizable instead of using our current approach where calculations are done sequentially.

## 5. Sources

- CS 583: Deep Learning
- Claude.ai
- <https://medium.com/@sanathshetty444/the-mathematics-behind-deep-learning-a-complete-guide-2f40d3d55a76>
- <https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi>
