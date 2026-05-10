![neuronum_logo](assets/logo.svg)

# neuronum

A simple, high-performance C++17 neural network library for classification problems. This library is made specifically to work with numerical UCI Machine Learning datasets (IRIS, Wine, Wisconsin, Liver, etc.).

## How to build and run

We use CMake for our project, thus, to run our project you need CMake version 3.13 or greater, and a compiler that supports C++17 at the very least.

For building, run the following from your root directory:

```bash
cmake -B build
cmake --build build
```

This will build a **static library** and will link it against our test `main.cpp` for you to see. However, the output of that build is simply a `.a` file.

Then, to test our library within the test/main.cpp file, run the following from your root directory:

```bash
./build/main
```

The provided test file for you created 4 neural networks each for a dataset within our `data/` folder, running each sequentially.

### Third-Party dependencies

We use [rapidcsv](https://github.com/d99kris/rapidcsv) for CSV parsing of numerical data, as well as categorical. The reason for this is that making a CSV parser is out of scope for our project as we are building a neural network library, so rapidcsv provides a fast and reliable CSV parser for us to use.

Since [rapidcsv](https://github.com/d99kris/rapidcsv) is a **header-only** library, we simply have the header file itself in our `external/` folder. So there is no need to git-clone or externally install the library.

## Example

Here is an example of usingo our library for IRIS Dataset

```cpp
    Dataset iris = DataLoader::load("data/Iris.csv", "Species", 0.2, {"Id"});

    printf("=== IRIS ===\n");
    printf("X_train: %zu x %zu\n", iris.X_train.rows(), iris.X_train.cols());
    printf("Y_train: %zu x %zu\n", iris.Y_train.rows(), iris.Y_train.cols());
    printf("X_test:  %zu x %zu\n", iris.X_test.rows(), iris.X_test.cols());
    printf("Y_test:  %zu x %zu\n", iris.Y_test.rows(), iris.Y_test.cols());
    iris.X_train.print();

    // IRIS network
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

    double train_ms = std::chrono::duration<double, std::milli>(train_end - train_start).count();
    double eval_ms = std::chrono::duration<double, std::milli>(eval_end - eval_start).count();
    printf("Training time:   %.2f ms\n", train_ms);
    printf("Evaluation time: %.2f ms\n", eval_ms);
```

You will see epochs and loss values in real time.

## Datasets Tested

- [Iris Species](https://www.kaggle.com/datasets/uciml/iris>)
- [Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
- [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- [Indian Liver Patient Records](https://www.kaggle.com/datasets/uciml/indian-liver-patient-records)
- [Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

## Disclaimer

IMAGE DATA IS NOT SUPPORTED!

## Credits

Created by Joshua Hizgiaev and Rishi Bidarkota for Stevens class FE522: C++ Programming in Finance.
