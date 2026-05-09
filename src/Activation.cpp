#include "../include/Activation.h"
#include <cmath>

Matrix ReLU::forward(const Matrix &input_matrix) {
  input_cache = input_matrix;
  Matrix output_matrix = input_matrix;
  for (size_t i = 0; i < output_matrix.rows(); i++) {
    for (size_t j = 0; j < output_matrix.cols(); j++) {
      if (output_matrix.at(i, j) < 0) {
        output_matrix.at(i, j) = 0;
      }
    }
  }
  return output_matrix;
}

Matrix ReLU::backward(const Matrix &gradient_matrix) {
  Matrix result = gradient_matrix;
  for (size_t i = 0; i < result.rows(); i++) {
    for (size_t j = 0; j < result.cols(); j++) {
      if (input_cache.at(i, j) <= 0) {
        result.at(i, j) = 0;
      }
    }
  }
  return result;
}

Matrix Sigmoid::forward(const Matrix &input_matrix) {
  Matrix output_matrix(input_matrix.rows(), input_matrix.cols());
  for (size_t i = 0; i < output_matrix.rows(); i++) {
    for (size_t j = 0; j < output_matrix.cols(); j++) {
      output_matrix.at(i, j) = 1.0 / (1.0 + std::exp(-input_matrix.at(i, j)));
    }
  }
  output_cache = output_matrix;
  return output_matrix;
}

Matrix Sigmoid::backward(const Matrix &gradient_matrix) {
  Matrix result = gradient_matrix;
  for (size_t i = 0; i < result.rows(); i++) {
    for (size_t j = 0; j < result.cols(); j++) {
      result.at(i, j) *= output_cache.at(i, j) * (1.0 - output_cache.at(i, j));
    }
  }
  return result;
}

Matrix Softmax::forward(const Matrix &input_matrix) {
  Matrix output_matrix(input_matrix.rows(), input_matrix.cols());
  for (size_t j = 0; j < input_matrix.cols(); j++) {
    double max_val = input_matrix.at(0, j);
    for (size_t i = 1; i < input_matrix.rows(); i++) {
      max_val = std::max(max_val, input_matrix.at(i, j));
    }
    double sum = 0.0;
    for (size_t i = 0; i < input_matrix.rows(); i++) {
      output_matrix.at(i, j) = std::exp(input_matrix.at(i, j) - max_val);
      sum += output_matrix.at(i, j);
    }
    for (size_t i = 0; i < input_matrix.rows(); i++) {
      output_matrix.at(i, j) /= sum;
    }
  }
  return output_matrix;
}

Matrix Softmax::backward(const Matrix &gradient_matrix) {
  return gradient_matrix;
}
