#pragma once

#include "Matrix.h"
#include <string>
#include <unordered_set>
#include <vector>

/*
 * A struct representing all the data we need to use.
 */
struct Dataset {
    Matrix X_train;
    Matrix X_test;
    Matrix Y_train;
    Matrix Y_test;
};

class DataLoader {
  public:
    /*
     * Loads a CSV file, takes a path and column name representing the target
     * label. Allows you to also specifiy the train-test-split ratio, by default
     * is 0.2 representing 20% test 80% train.
     */
    static Dataset
    load(const std::string &path, const std::string &label_col,
         double test_ratio = 0.2,
         const std::unordered_set<std::string> &ignore_cols = {});

  private:
    /*
     * Normalizes the data in the matrix.
     */
    static Matrix normalize(const Matrix &X);

    /*
     * One hot encodes variables given a vector of labels and number of classes.
     */
    static Matrix one_hot_encode(const std::vector<std::string> &labels,
                                 size_t num_classes);

    /*
     * Returns the number of classes.
     */
    static size_t num_classes(const std::vector<std::string> &labels);
};