#include "Activation.h"
#include "Dataloader.h"
#include "Dense.h"
#include "Network.h"
#include <chrono>

int main() {
    // IRIS Example
    {
        // Load data matrix
        Dataset iris =
            DataLoader::load("data/Iris.csv", "Species", 0.2, {"Id"});

        printf("=== IRIS ===\n");
        printf("X_train: %zu x %zu\n", iris.X_train.rows(),
               iris.X_train.cols());
        printf("Y_train: %zu x %zu\n", iris.Y_train.rows(),
               iris.Y_train.cols());
        printf("X_test:  %zu x %zu\n", iris.X_test.rows(), iris.X_test.cols());
        printf("Y_test:  %zu x %zu\n", iris.Y_test.rows(), iris.Y_test.cols());
        iris.X_train.print();

        // Create network
        Network net_iris;
        net_iris.add(new Dense(4, 16));
        net_iris.add(new ReLU());
        net_iris.add(new Dense(16, 3));
        net_iris.add(new Softmax());

        // Run training and evaluation
        auto train_start = std::chrono::high_resolution_clock::now();
        net_iris.train(iris.X_train, iris.Y_train, 200, 0.02);
        auto train_end = std::chrono::high_resolution_clock::now();

        auto eval_start = std::chrono::high_resolution_clock::now();
        net_iris.evaluate(iris.X_test, iris.Y_test);
        auto eval_end = std::chrono::high_resolution_clock::now();

        // Print the times
        double train_ms =
            std::chrono::duration<double, std::milli>(train_end - train_start)
                .count();
        double eval_ms =
            std::chrono::duration<double, std::milli>(eval_end - eval_start)
                .count();
        printf("Training time:   %.2f ms\n", train_ms);
        printf("Evaluation time: %.2f ms\n", eval_ms);
        printf("========================================\n");
    }

    // Wine Example
    {
        // Load data matrix
        Dataset wine =
            DataLoader::load("data/winequality-red.csv", "quality", 0.2);

        printf("\n=== WINE ===\n");
        printf("X_train: %zu x %zu\n", wine.X_train.rows(),
               wine.X_train.cols());
        printf("Y_train: %zu x %zu\n", wine.Y_train.rows(),
               wine.Y_train.cols());
        printf("X_test:  %zu x %zu\n", wine.X_test.rows(), wine.X_test.cols());
        printf("Y_test:  %zu x %zu\n", wine.Y_test.rows(), wine.Y_test.cols());
        wine.X_train.print();

        // Create network
        Network net_wine;
        net_wine.add(new Dense(11, 64));
        net_wine.add(new ReLU());
        net_wine.add(new Dense(64, 32));
        net_wine.add(new ReLU());
        net_wine.add(new Dense(32, 6));
        net_wine.add(new Softmax());

        // Run training and evaluation
        auto train_start = std::chrono::high_resolution_clock::now();
        net_wine.train(wine.X_train, wine.Y_train, 500, 0.001);
        auto train_end = std::chrono::high_resolution_clock::now();

        auto eval_start = std::chrono::high_resolution_clock::now();
        net_wine.evaluate(wine.X_test, wine.Y_test);
        auto eval_end = std::chrono::high_resolution_clock::now();

        // Print the times
        double train_ms =
            std::chrono::duration<double, std::milli>(train_end - train_start)
                .count();
        double eval_ms =
            std::chrono::duration<double, std::milli>(eval_end - eval_start)
                .count();
        printf("Training time:   %.2f ms\n", train_ms);
        printf("Evaluation time: %.2f ms\n", eval_ms);
        printf("========================================\n");
    }

    // Wisconsin Example
    {
        // Load data matrix
        Dataset cancer =
            DataLoader::load("data/bc.csv", "diagnosis", 0.2, {"id"});

        printf("\n=== BREAST CANCER ===\n");
        printf("X_train: %zu x %zu\n", cancer.X_train.rows(),
               cancer.X_train.cols());
        printf("Y_train: %zu x %zu\n", cancer.Y_train.rows(),
               cancer.Y_train.cols());
        printf("X_test:  %zu x %zu\n", cancer.X_test.rows(),
               cancer.X_test.cols());
        printf("Y_test:  %zu x %zu\n", cancer.Y_test.rows(),
               cancer.Y_test.cols());

        // Create network
        Network net_cancer;
        net_cancer.add(new Dense(30, 64));
        net_cancer.add(new ReLU());
        net_cancer.add(new Dense(64, 32));
        net_cancer.add(new ReLU());
        net_cancer.add(new Dense(32, 2));
        net_cancer.add(new Softmax());

        auto train_start = std::chrono::high_resolution_clock::now();
        net_cancer.train(cancer.X_train, cancer.Y_train, 300, 0.001);
        auto train_end = std::chrono::high_resolution_clock::now();

        auto eval_start = std::chrono::high_resolution_clock::now();
        net_cancer.evaluate(cancer.X_test, cancer.Y_test);
        auto eval_end = std::chrono::high_resolution_clock::now();

        double train_ms =
            std::chrono::duration<double, std::milli>(train_end - train_start)
                .count();
        double eval_ms =
            std::chrono::duration<double, std::milli>(eval_end - eval_start)
                .count();
        printf("Training time:   %.2f ms\n", train_ms);
        printf("Evaluation time: %.2f ms\n", eval_ms);
    }

    {
        Dataset liver = DataLoader::load("data/indian_liver.csv", "Dataset",
                                         0.2, {"Gender"});

        printf("\n=== LIVER DISEASE ===\n");
        printf("X_train: %zu x %zu\n", liver.X_train.rows(),
               liver.X_train.cols());
        printf("Y_train: %zu x %zu\n", liver.Y_train.rows(),
               liver.Y_train.cols());
        printf("X_test:  %zu x %zu\n", liver.X_test.rows(),
               liver.X_test.cols());
        printf("Y_test:  %zu x %zu\n", liver.Y_test.rows(),
               liver.Y_test.cols());

        Network net_liver;
        net_liver.add(new Dense(9, 32));
        net_liver.add(new ReLU());
        net_liver.add(new Dense(32, 64));
        net_liver.add(new ReLU());
        net_liver.add(new Dense(64, 32));
        net_liver.add(new ReLU());
        net_liver.add(new Dense(32, 16));
        net_liver.add(new ReLU());
        net_liver.add(new Dense(16, 8));
        net_liver.add(new ReLU());
        net_liver.add(new Dense(8, 2));
        net_liver.add(new Softmax());

        auto train_start = std::chrono::high_resolution_clock::now();
        net_liver.train(liver.X_train, liver.Y_train, 500, 0.0001);
        auto train_end = std::chrono::high_resolution_clock::now();

        auto eval_start = std::chrono::high_resolution_clock::now();
        net_liver.evaluate(liver.X_test, liver.Y_test);
        auto eval_end = std::chrono::high_resolution_clock::now();

        double train_ms =
            std::chrono::duration<double, std::milli>(train_end - train_start)
                .count();
        double eval_ms =
            std::chrono::duration<double, std::milli>(eval_end - eval_start)
                .count();
        printf("Training time:   %.2f ms\n", train_ms);
        printf("Evaluation time: %.2f ms\n", eval_ms);
    }

    {
        Dataset credit_card =
            DataLoader::load("data/credit_card.csv", "default", 0.2, {"ID"});

        printf("\n=== CREDIT CARD ===\n");
        printf("X_train: %zu x %zu\n", credit_card.X_train.rows(),
               credit_card.X_train.cols());
        printf("Y_train: %zu x %zu\n", credit_card.Y_train.rows(),
               credit_card.Y_train.cols());
        printf("X_test:  %zu x %zu\n", credit_card.X_test.rows(),
               credit_card.X_test.cols());
        printf("Y_test:  %zu x %zu\n", credit_card.Y_test.rows(),
               credit_card.Y_test.cols());

        Network cc_net;
        cc_net.add(new Dense(23, 32));
        cc_net.add(new ReLU());
        cc_net.add(new Dense(32, 64));
        cc_net.add(new ReLU());
        cc_net.add(new Dense(64, 16));
        cc_net.add(new ReLU());
        cc_net.add(new Dense(16, 2));
        cc_net.add(new Softmax());

        auto train_start = std::chrono::high_resolution_clock::now();
        cc_net.train(credit_card.X_train, credit_card.Y_train, 100, 0.0001);
        auto train_end = std::chrono::high_resolution_clock::now();

        auto eval_start = std::chrono::high_resolution_clock::now();
        cc_net.evaluate(credit_card.X_test, credit_card.Y_test);
        auto eval_end = std::chrono::high_resolution_clock::now();

        double train_ms =
            std::chrono::duration<double, std::milli>(train_end - train_start)
                .count();
        double eval_ms =
            std::chrono::duration<double, std::milli>(eval_end - eval_start)
                .count();
        printf("Training time:   %.2f ms\n", train_ms);
        printf("Evaluation time: %.2f ms\n", eval_ms);
    }

    return 0;
}