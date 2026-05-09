#include "../include/Dense.h"

// forward: W·input + b, cache input for backprop
Matrix Dense::forward(const Matrix &input_mat) {
    input_cache = input_mat;
    return W * input_mat + b;
}

// backward: compute dW, db, return delta for previous layer
Matrix Dense::backward(const Matrix &grad_mat) {
    dW = grad_mat * input_cache.transpose();
    db = grad_mat;
    return W.transpose() * grad_mat;
}

// update: gradient descent
void Dense::update(double learning_rate) {
    W = W - dW * learning_rate;
    b = b - db * learning_rate;
}