#include "../include/Matrix.h"

Matrix::Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
  data_ = new double *[rows_];
  for (size_t i = 0; i < rows_; i++) {
    data_[i] = new double[cols_];
    for (size_t j = 0; j < cols_; j++) {
      data_[i][j] = 0.0;
    }
  }
}

Matrix::~Matrix() {
  for (size_t i = 0; i < rows_; i++) {
    delete[] data_[i];
  }
  delete[] data_;
}

Matrix::Matrix(const Matrix &other) : rows_(other.rows_), cols_(other.cols_) {
  data_ = new double *[rows_];
  for (size_t i = 0; i < rows_; i++) {
    data_[i] = new double[cols_];
    for (size_t j = 0; j < cols_; j++) {
      data_[i][j] = other.data_[i][j];
    }
  }
}

Matrix &Matrix::operator=(const Matrix &other) {
  if (this == &other) {
    return *this;
  }
  for (size_t i = 0; i < rows_; i++) {
    delete[] data_[i];
  }
  delete[] data_;

  rows_ = other.rows_;
  cols_ = other.cols_;
  data_ = new double *[rows_];
  for (size_t i = 0; i < rows_; i++) {
    data_[i] = new double[cols_];
    for (size_t j = 0; j < cols_; j++) {
      data_[i][j] = other.data_[i][j];
    }
  }
  return *this;
}

Matrix Matrix::operator+(const Matrix &other) const {
  Matrix result(rows_, cols_);
  for (size_t i = 0; i < rows_; i++) {
    for (size_t j = 0; i < cols_; j++) {
      result.data_[i][j] = data_[i][j] + other.data_[i][j];
    }
  }
  return result;
}

Matrix Matrix::operator-(const Matrix &other) const {
  Matrix result(rows_, cols_);
  for (size_t i = 0; i < rows_; i++) {
    for (size_t j = 0; j < other.cols_; j++) {
      result.data_[i][j] = data_[i][j] - other.data_[i][j];
    }
  }
  return result;
}

Matrix Matrix::operator*(const Matrix &other) const {
  Matrix result(rows_, other.cols_);
  for (size_t i = 0; i < rows_; i++) {
    for (size_t j = 0; j < other.cols_; j++) {
      result.data_[i][j] = data_[i][j] - other.data_[i][j];
    }
  }
  return result;
}

double &Matrix::at(size_t r, size_t c) { return data_[r][c]; }

const double &Matrix::at(size_t r, size_t c) const { return data_[r][c]; }

size_t Matrix::rows() const { return rows_; }

size_t Matrix::cols() const { return cols_; }
