#include "Activation.h"
#include "Dataloader.h"
#include "Dense.h"
#include "Network.h"
#include <chrono>
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

    // Make a network for IRIS
    Network net_iris;
    net_iris.add(new Dense(4, 16));
    net_iris.add(new ReLU());
    net_iris.add(new Dense(16, 3));
    net_iris.add(new Softmax());

    auto train_start = std::chrono::high_resolution_clock::now();
    net_iris.train(iris.X_train, iris.Y_train, 200, 0.02);
    auto train_end = std::chrono::high_resolution_clock::now();

    auto eval_start = std::chrono::high_resolution_clock::now();
    net_iris.evaluate(iris.X_test, iris.Y_test);
    auto eval_end = std::chrono::high_resolution_clock::now();

    double train_ms =
        std::chrono::duration<double, std::milli>(train_end - train_start)
            .count();
    double eval_ms =
        std::chrono::duration<double, std::milli>(eval_end - eval_start)
            .count();

    printf("Training time:  %.2f ms\n", train_ms);
    printf("Evaluation time:  %.2f ms\n", eval_ms);

    printf("========================================\n");

    // Make a network wine
    Network net_wine;
    net_wine.add(new Dense(11, 64));
    net_wine.add(new ReLU());
    net_wine.add(new Dense(64, 32));
    net_wine.add(new ReLU());
    net_wine.add(new Dense(32, 6));
    net_wine.add(new Softmax());

    train_start = std::chrono::high_resolution_clock::now();
    net_wine.train(wine.X_train, wine.Y_train, 500, 0.001);
    train_end = std::chrono::high_resolution_clock::now();

    eval_start = std::chrono::high_resolution_clock::now();
    net_wine.evaluate(wine.X_test, wine.Y_test);
    eval_end = std::chrono::high_resolution_clock::now();

    train_ms =
        std::chrono::duration<double, std::milli>(train_end - train_start)
            .count();
    eval_ms = std::chrono::duration<double, std::milli>(eval_end - eval_start)
                  .count();

    printf("Training time:  %.2f ms\n", train_ms);
    printf("Evaluation time:  %.2f ms\n", eval_ms);

    return 0;
}