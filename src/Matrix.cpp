#include "../include/Matrix.h"
#include <cstdio>
#include <stdexcept>

Matrix::Matrix() : rows_(0), cols_(0), data_(nullptr) {}

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
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("matrix addition: dimensions must match");
    }
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
            result.data_[i][j] = data_[i][j] + other.data_[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument(
            "matrix subtraction: dimensions must match");
    }
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
            result.data_[i][j] = data_[i][j] - other.data_[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix &other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument(
            "matrix multiplication: dimensions must match");
    }
    Matrix result(rows_, other.cols_);
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < other.cols_; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < cols_; k++) {
                sum += data_[i][k] * other.data_[k][j];
            }
            result.data_[i][j] = sum;
        }
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
            result.data_[j][i] = data_[i][j];
        }
    }
    return result;
}

double &Matrix::at(size_t r, size_t c) { return data_[r][c]; }

const double &Matrix::at(size_t r, size_t c) const { return data_[r][c]; }

size_t Matrix::rows() const { return rows_; }

size_t Matrix::cols() const { return cols_; }

void Matrix::print(size_t max_rows) const {
    size_t row_limit = (max_rows > rows_) ? rows_ : max_rows;
    for (size_t i = 0; i < row_limit; i++) {
        for (size_t j = 0; j < cols_; j++) {
            printf("%.2f ", data_[i][j]);
        }
        printf("\n");
    }
    if (row_limit < rows_) {
        printf("... %zu more rows\n", rows_ - row_limit);
    }
}
