#include "../include/Network.h"

void Network::add(Layer *layer) {
    // Wrap the existing raw pointer around unique pointer
    layers.push_back(std::unique_ptr<Layer>(layer));
}

Matrix Network::forward(const Matrix &X) {
    Matrix output = X;
    for (auto &layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void Network::backward(const Matrix &Y_pred, const Matrix &Y_true) {
    Matrix output_gradient = Y_pred - Y_true;
    for (int i = layers.size() - 1; i >= 0; i--) {
        output_gradient = layers[i]->backward(output_gradient);
    }
}

void Network::update(double learning_rate) {
    for (auto &layer : layers) {
        layer->update(learning_rate);
    }
}

void Network::train(const Matrix &X, const Matrix &Y, int epochs, double lr) {
    size_t n_samples = X.rows();
    size_t n_features = X.cols();
    size_t n_classes = Y.cols();

    // transpose once — Dense expects (features x samples)
    Matrix X_t(n_features, n_samples);
    Matrix Y_t(n_classes, n_samples);
    for (size_t i = 0; i < n_samples; i++) {
        for (size_t j = 0; j < n_features; j++)
            X_t.at(j, i) = X.at(i, j);
        for (size_t k = 0; k < n_classes; k++)
            Y_t.at(k, i) = Y.at(i, k);
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        Matrix Y_pred = forward(X_t);
        backward(Y_pred, Y_t);
        update(lr);

        if (epoch % 10 == 0) {
            double loss = cross_entropy(Y_pred, Y_t);
            double acc = accuracy(Y_pred, Y_t) * 100.0;
            printf("Epoch %3d — loss: %.4f  acc: %.1f%%\n", epoch, loss, acc);
        }
    }
}