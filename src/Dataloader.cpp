#include "../include/Dataloader.h"
#include "../external/rapidcsv.h"

#include <random>
#include <unordered_map>
#include <unordered_set>

Matrix DataLoader::normalize(const Matrix &X) {
    Matrix res(X.rows(), X.cols());
    // Normalize each column
    for (size_t j = 0; j < X.cols(); j++) {
        // Get mean of column
        double sum = 0.0;
        for (size_t i = 0; i < X.rows(); i++) {
            sum += X.at(i, j);
        }
        double mean = sum / X.rows();

        // Get standard deviation of column
        double sq_sum = 0.0;
        for (size_t i = 0; i < X.rows(); i++) {
            sq_sum += (X.at(i, j) - mean) * (X.at(i, j) - mean);
        }
        double std_dev = std::sqrt(sq_sum / X.rows());

        // Normalize the data for the column
        for (size_t i = 0; i < X.rows(); i++) {
            res.at(i, j) = (X.at(i, j) - mean) / std_dev;
        }
    }
    return res;
}

size_t DataLoader::num_classes(const std::vector<std::string> &labels) {
    size_t res = 0;
    std::unordered_set<std::string> seen(labels.size());
    for (const auto &label : labels) {
        if (seen.find(label) == seen.end()) {
            seen.insert(label);
            res++;
        }
    }
    return res;
}

Matrix DataLoader::one_hot_encode(const std::vector<std::string> &labels,
                                  size_t num_classes) {
    std::unordered_map<std::string, int> label_map;
    int idx = 0;
    for (const auto &label : labels) {
        if (label_map.find(label) == label_map.end()) {
            label_map[label] = idx++;
        }
    }

    Matrix Y(labels.size(), num_classes);
    for (size_t i = 0; i < labels.size(); i++) {
        int class_idx = label_map.at(labels[i]);
        Y.at(i, class_idx) = 1.0;
    }
    return Y;
}

Dataset DataLoader::load(const std::string &path, const std::string &label_col,
                         double test_ratio,
                         const std::unordered_set<std::string> &ignore_cols) {
    rapidcsv::Document doc(path);
    size_t rows = doc.GetRowCount(), cols = doc.GetColumnCount();
    std::vector<std::string> feature_names;
    for (size_t i = 0; i < cols; i++) {
        std::string cname = doc.GetColumnName(i);
        if ((cname != label_col) &&
            (ignore_cols.find(cname) == ignore_cols.end())) {
            feature_names.push_back(cname);
        }
    }
    size_t n_features = feature_names.size();

    // This matrix includes all input data, NOT output, this is raw feature
    // matrix
    Matrix X(rows, n_features);
    // For each column
    for (size_t j = 0; j < n_features; j++) {
        std::vector<double> col = doc.GetColumn<double>(feature_names[j]);
        // For each row
        for (size_t k = 0; k < rows; k++) {
            X.at(k, j) = col[k];
        }
    }

    // Get output vector
    std::vector<std::string> labels = doc.GetColumn<std::string>(label_col);
    size_t n_classes = num_classes(labels);
    Matrix Y = one_hot_encode(labels, n_classes);

    // Normalize features, = operator will free the data AFTER normalize
    // finishes execution
    X = normalize(X);

    // shuffle indices to get random rows
    std::vector<size_t> indices(rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    size_t test_size = static_cast<size_t>(rows * test_ratio);
    size_t train_size = rows - test_size;

    Matrix X_train(train_size, n_features);
    Matrix X_test(test_size, n_features);
    Matrix Y_train(train_size, n_classes);
    Matrix Y_test(test_size, n_classes);

    for (size_t i = 0; i < train_size; i++) {
        size_t src = indices[i];
        for (size_t j = 0; j < n_features; j++) {
            X_train.at(i, j) = X.at(src, j);
        }
        for (int k = 0; k < n_classes; k++) {
            Y_train.at(i, k) = Y.at(src, k);
        }
    }

    for (size_t i = 0; i < test_size; i++) {
        size_t src = indices[train_size + i];
        for (size_t j = 0; j < n_features; j++) {
            X_test.at(i, j) = X.at(src, j);
        }
        for (int k = 0; k < n_classes; k++) {
            Y_test.at(i, k) = Y.at(src, k);
        }
    }

    return {X_train, X_test, Y_train, Y_test};
}
