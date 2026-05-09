#pragma once

#include "Matrix.h"

class Layer {
  public:
    // Pure virtual functions to implement in layer subclasses
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix &input_mat) = 0;
    virtual Matrix backward(const Matrix &grad_mat) = 0;
    // Not a pure virtual function because activation functions dont update
    virtual void update(double learning_rate) {};
};
