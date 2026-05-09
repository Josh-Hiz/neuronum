#pragma once

#include "Layer.h"

class Dense : public Layer {
  public:
    Dense(size_t input_size, size_t output_size);
    Matrix forward(const Matrix &input_mat) override;
    Matrix backward(const Matrix &grad_mat) override;
    void update(double learning_rate) override;

  private:
    Matrix W, b, dW, db;
    // The original input is needed for backwards propagation
    Matrix input_cache;
};