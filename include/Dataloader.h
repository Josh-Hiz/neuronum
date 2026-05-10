#pragma once

#include "Matrix.h"
#include <string>
#include <unordered_set>
#include <vector>

struct Dataset {
    Matrix X_train;
    Matrix X_test;
    Matrix Y_train;
    Matrix Y_test;
};

class DataLoader {
  public:
    static Dataset
    load(const std::string &path, const std::string &label_col,
         double test_ratio = 0.2,
         const std::unordered_set<std::string> &ignore_cols = {});

  private:
    static Matrix normalize(const Matrix &X);

    static Matrix one_hot_encode(const std::vector<std::string> &labels,
                                 size_t num_classes);

    static size_t num_classes(const std::vector<std::string> &labels);
};