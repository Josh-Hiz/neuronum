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