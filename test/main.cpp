#include "Activation.h"
#include "Dataloader.h"
#include "Dense.h"
#include "Network.h"
#include <iostream>

int main() {

    // Iris
    Dataset iris = DataLoader::load("data/Iris.csv", "Species", 0.2, {"Id"});

    std::cout << "=== IRIS ===\n";
    std::cout << "X_train: " << iris.X_train.rows() << " x "
              << iris.X_train.cols() << "\n";
    std::cout << "Y_train: " << iris.Y_train.rows() << " x "
              << iris.Y_train.cols() << "\n";
    std::cout << "X_test:  " << iris.X_test.rows() << " x "
              << iris.X_test.cols() << "\n";
    std::cout << "Y_test:  " << iris.Y_test.rows() << " x "
              << iris.Y_test.cols() << "\n";

    iris.X_train.print();

    // Wine
    Dataset wine = DataLoader::load("data/winequality-red.csv", "quality", 0.2);

    std::cout << "\n=== WINE ===\n";
    std::cout << "X_train: " << wine.X_train.rows() << " x "
              << wine.X_train.cols() << "\n";
    std::cout << "Y_train: " << wine.Y_train.rows() << " x "
              << wine.Y_train.cols() << "\n";
    std::cout << "X_test:  " << wine.X_test.rows() << " x "
              << wine.X_test.cols() << "\n";
    std::cout << "Y_test:  " << wine.Y_test.rows() << " x "
              << wine.Y_test.cols() << "\n";

    wine.X_train.print();

    Network net;
    net.add(new Dense(4, 16));
    net.add(new ReLU());
    net.add(new Dense(16, 3));
    net.add(new Softmax());
    net.train(iris.X_train, iris.Y_train, 200, 0.02);

    return 0;
}