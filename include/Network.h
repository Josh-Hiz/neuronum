#pragma once

#include "Layer.h"
#include "Matrix.h"
#include <memory>
#include <vector>

class Network {
  public:
    void add(Layer *layer);
    Matrix forward(const Matrix &X);
    void backward(const Matrix &Y_pred, const Matrix &Y_true);
    void update(double learning_rate);
    void train(const Matrix &X, const Matrix &Y, int epochs,
               double learning_rate);

  private:
    std::vector<std::unique_ptr<Layer>> layers;

    static double cross_entropy(const Matrix &Y_pred, const Matrix &Y_true);
    static double accuracy(const Matrix &Y_pred, const Matrix &Y_true);
};