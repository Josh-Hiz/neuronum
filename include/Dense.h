#pragma once

#include "Layer.h"

class Dense : public Layer {
public:
  Dense(size_t input_size, size_t output_size);
  Matrix forward(const Matrix &input) override;
  Matrix backward(const Matrix &grad, double learning_rate) override;

private:
  Matrix weights;
  Matrix bias;
};
