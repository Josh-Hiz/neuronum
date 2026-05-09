#pragma once

#include "Layer.h"
#include "Matrix.h"
#include <vector>

class Network {
  public:
    void add(Layer *layer);
    Matrix forward(const Matrix *X);
    void backward(const Matrix &Y_pred, const Matrix &Y_true);
    void update(double learning_rate);

  private:
    std::vector<Layer *> layers;
};
