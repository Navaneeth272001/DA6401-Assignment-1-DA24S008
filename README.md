# DA6401-Assignment-1-DA24S008
Repository for Assignment 1 - Navaneethakrishnan - DA24S008

To View the train.py test run:

wandb: ⭐️ View project at https://wandb.ai/da24s008-indian-institute-of-technology-madras/Assignment 1
wandb: 🚀 View run at https://wandb.ai/da24s008-indian-institute-of-technology-madras/Assignment 1/runs/hz9xie4m
wandb: 🚀 View run resilient-wood-11 at: https://wandb.ai/da24s008-indian-institute-of-technology-madras/Assignment 1/runs/hz9xie4m

# MNIST Top 3 Model Training with Weights & Biases

## Overview
This project trains the top three machine learning models for the MNIST dataset using the best configurations obtained from Weights & Biases (W&B). It evaluates their performance using a confusion matrix and logs the results to W&B.

## Features
- Loads the MNIST dataset and preprocesses it.
- Retrieves the top three model configurations based on validation accuracy.
- Trains and evaluates the models.
- Logs confusion matrices for each model to W&B.

## Requirements
To run this project, install the required dependencies:

```bash
pip install numpy seaborn matplotlib wandb tensorflow
```

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/Navaneeth272001/DA6401-Assignment-1-DA24S008.git
   ```
2. Log in to Weights & Biases:
   ```bash
   wandb login
   ```
3. Run the training script:
   ```bash
    python train.py --wandb_entity myname --wandb_project myprojectname
   ```
|           Name           | Default Value | Description                                                               |
| :----------------------: | :-----------: | :------------------------------------------------------------------------ |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard      |
|  `-we`, `--wandb_entity` |     myname    | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
|     `-d`, `--dataset`    | fashion_mnist | choices:  ["mnist", "fashion_mnist"]                                      |
|     `-e`, `--epochs`     |       1       | Number of epochs to train neural network.                                 |
|   `-b`, `--batch_size`   |       4       | Batch size used to train neural network.                                  |
|      `-l`, `--loss`      | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"]                         |
|    `-o`, `--optimizer`   |      sgd      | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]          |
| `-lr`, `--learning_rate` |      0.1      | Learning rate used to optimize model parameters                           |
|    `-m`, `--momentum`    |      0.5      | Momentum used by momentum and nag optimizers.                             |
|     `-beta`, `--beta`    |      0.5      | Beta used by rmsprop optimizer                                            |
|    `-beta1`, `--beta1`   |      0.5      | Beta1 used by adam and nadam optimizers.                                  |
|    `-beta2`, `--beta2`   |      0.5      | Beta2 used by adam and nadam optimizers.                                  |
|    `-eps`, `--epsilon`   |    0.000001   | Epsilon used by optimizers.                                               |
| `-w_d`, `--weight_decay` |       .0      | Weight decay used by optimizers.                                          |
|  `-w_i`, `--weight_init` |     random    | choices:  ["random", "Xavier"]                                            |
|  `-nhl`, `--num_layers`  |       1       | Number of hidden layers used in feedforward neural network.               |
|  `-sz`, `--hidden_size`  |       4       | Number of hidden neurons in a feedforward layer.                          |
|   `-a`, `--activation`   |    sigmoid    | choices:  ["identity", "sigmoid", "tanh", "ReLU"]                         |

## Expected Outputs
- The script will display confusion matrices for each model.
- Model performance metrics will be logged in W&B.

## File Structure
```
mnist-top-models/
│── train.py  # Main script for training models
│── README.md # Project documentation
```
