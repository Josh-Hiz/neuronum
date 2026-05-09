#include "../include/Matrix.h"

Matrix::Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
  data_ = new double[rows_ * cols_];
  for (size_t i = 0; i < rows_ * cols_; i++) {
    data_[i] = 0.0;
  }
}

Matrix::~Matrix() {
    delete[] data_;
}

Matrix::Matrix(const Matrix &other) 

    double &Matrix::at(size_t r, size_t c) {
  return data_[r * cols_ + c];
}

const double &Matrix::at(size_t r, size_t c) const {
  return data_[r * cols_ + c];
}
