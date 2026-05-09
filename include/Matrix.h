#pragma once

#include <cstddef>

class Matrix {
public:
  Matrix();
  Matrix(size_t rows, size_t cols);
  ~Matrix();
  Matrix(const Matrix &other);
  Matrix &operator=(const Matrix &other);

  Matrix operator+(const Matrix &other) const;
  Matrix operator-(const Matrix &other) const;
  Matrix operator*(const Matrix &other) const;

  Matrix operator*(double scalar) const;

  Matrix transpose() const;

  double &at(size_t r, size_t c);
  const double &at(size_t r, size_t c) const;

  size_t rows() const;
  size_t cols() const;

  void print(size_t max_rows = 10) const;

private:
  size_t rows_;
  size_t cols_;
  double **data_;
};
