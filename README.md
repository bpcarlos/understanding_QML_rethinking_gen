# Understanding quantum machine learning also requires rethinking generalization

This repository contains the data and code in the paper "Understanding quantum machine learning also requires rethinking generalization," found in [arXiv](https://arxiv.org/abs/2306.13461). The code relies on the following packages: TensorCircuit [(GitHub)](https://github.com/tencent-quantum-lab/tensorcircuit) and Qibo [(GitHub)](https://github.com/qiboteam/qibo). Please ensure that these packages are installed before running the code.


## Repository structure

The repository is organized into folders corresponding to different experiments conducted. Three primary code files can be executed:

1. `main_code.py`: This file trains and executes the quantum convolutional neural network from scratch for the different experiments. It accepts the following arguments:

- `--training_data` (int): Training data size. It supports `5`, `8`, `10`, `14`, and `20`. Default = `5`.
- `--accuracy_training` (int): minimum training accuracy that shall be achieved. The code performs new random initialization and training if the accuracy falls below this threshold. Default = `100`

Note that executing `main_code.py` can be computationally demanding. To reproduce the results presented in the paper, consider running the following files instead.

2. `accuracy_train.py`: Run this file to obtain the training accuracy using the best parameters determined by the authors. It accepts the same `--training_data` argument as `main_code.py`.

3. `accuracy_test.py`: Run this file to obtain the test accuracy using the best parameters determined by the authors. Again, it accepts the same `--training_data` argument as `main_code.py`.


## Running the code

To run the code, follow these steps:

1. Install the required packages: TensorCircuit and Qibo.

2. Choose the appropriate code file based on your requirements:
- To train and execute the quantum convolutional neural network from scratch, run `main_code.py`.
- For reproducing the paper's results, execute `accuracy_train.py` on the training set and `accuracy_test.py` on the test set.

3. Set the desired values for the arguments `--training_data` and `--accuracy_training` (if applicable) to customize the execution.

4. Execute the chosen code file, ensuring the required packages are accessible.

For further assistance or inquiries, please refer to the paper or contact the authors directly.
