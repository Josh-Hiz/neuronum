#pragma once

#include "Matrix.h"

class Layer {
public:
  virtual ~Layer() = default;
  virtual Matrix forward(const Matrix &input) = 0;
  virtual Matrix backward(const Matrix &grad, double learning_rate) = 0;
};
