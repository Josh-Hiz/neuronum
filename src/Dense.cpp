#include "../include/Dense.h"

#include <cmath>
#include <random>

Dense::Dense(size_t input_size, size_t output_size)
    : W(output_size, input_size), b(output_size, 1),
      dW(output_size, input_size), db(output_size, 1), input_cache(1, 1) {

    // Perform Xavier initialization
    std::mt19937 rng(42);
    double scale = std::sqrt(2.0 / input_size);
    std::normal_distribution<double> dist(0.0, scale);

    for (size_t i = 0; i < output_size; i++) {
        for (size_t j = 0; j < input_size; j++) {
            W.at(i, j) = dist(rng);
        }
    }
}

// forward function, W·input + b, cache input for backprop
Matrix Dense::forward(const Matrix &input_mat) {
    input_cache = input_mat;
    return (W * input_mat).broadcast_add(b); // not W * input_mat + b
}

// backward propagation, compute dW, db, return gradient for previous layer
Matrix Dense::backward(const Matrix &grad_mat) {
    dW = grad_mat * input_cache.transpose();

    // sum grad_mat across columns into a (output x 1) vector
    db = Matrix(grad_mat.rows(), 1);
    for (size_t i = 0; i < grad_mat.rows(); i++) {
        double sum = 0.0;
        for (size_t j = 0; j < grad_mat.cols(); j++)
            sum += grad_mat.at(i, j);
        db.at(i, 0) = sum;
    }
    return W.transpose() * grad_mat;
}

// gradient descent
void Dense::update(double learning_rate) {
    W = W - dW * learning_rate;
    b = b - db * learning_rate;
}