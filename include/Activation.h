#pragma once

#include "Layer.h"

class ReLU : public Layer {
public:
  Matrix forward(const Matrix &input_matrix) override;
  Matrix backward(const Matrix &gradient_matrix) override;

private:
  Matrix input_cache;
};

class Sigmoid : public Layer {
public:
  Matrix forward(const Matrix &input_matrix) override;
  Matrix backward(const Matrix &gradient_matrix) override;

private:
  Matrix output_cache;
};

class Softmax : public Layer {
public:
  Matrix forward(const Matrix &input_matrix) override;
  Matrix backward(const Matrix &gradient_matrix);
};
