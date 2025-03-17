from keras.datasets import fashion_mnist
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
import seaborn as sns


#Activation functions:

class Activations:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def onehot(y, num_class):
    one_hot = np.zeros((y.shape[0], num_class))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def cross_ent_loss(yTrue, yPred):
    return -np.sum(yTrue * np.log(yPred + 1e-8)) / yTrue.shape[0]

def accuracy_calc(y_true, y_pred):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

def sqrd_error_loss(yTrue, yPred):
    return np.mean(np.square(yTrue - yPred))


# Optimisation Algorithms:
class Optimisation:
    @staticmethod
    def sgd(weights, biases, gradients, params):
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']
        for i in range(len(weights)):
            weights[i] -= learning_rate * (gradients[i][0] + weight_decay * weights[i])
            biases[i] -= learning_rate * gradients[i][1]

    @staticmethod
    def momentum(weights, biases, gradients, params):
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']
        beta = params.get('beta', 0.9)
        if 'velocity' not in params:
            params['velocity'] = [(np.zeros_like(w), np.zeros_like(b)) for w, b in zip(weights, biases)]
        for i in range(len(weights)):
            params['velocity'][i] = (beta * params['velocity'][i][0] + (1 - beta) * gradients[i][0],
                                     beta * params['velocity'][i][1] + (1 - beta) * gradients[i][1])
            weights[i] -= learning_rate * (params['velocity'][i][0] + weight_decay * weights[i])
            biases[i] -= learning_rate * params['velocity'][i][1]

    @staticmethod
    def adam(weights, biases, gradients, params):
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        if 'm' not in params:
            params['m'] = [(np.zeros_like(w), np.zeros_like(b)) for w, b in zip(weights, biases)]
            params['v'] = [(np.zeros_like(w), np.zeros_like(b)) for w, b in zip(weights, biases)]
        for i in range(len(weights)):
            params['m'][i] = (beta1 * params['m'][i][0] + (1 - beta1) * gradients[i][0],
                              beta1 * params['m'][i][1] + (1 - beta1) * gradients[i][1])
            params['v'][i] = (beta2 * params['v'][i][0] + (1 - beta2) * (gradients[i][0] ** 2),
                              beta2 * params['v'][i][1] + (1 - beta2) * (gradients[i][1] ** 2))
            weights[i] -= learning_rate * params['m'][i][0] / (np.sqrt(params['v'][i][0]) + epsilon) + weight_decay * weights[i]
            biases[i] -= learning_rate * params['m'][i][1] / (np.sqrt(params['v'][i][1]) + epsilon)

    @staticmethod
    def nesterov(weights, biases, gradients, params):
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']
        beta = params.get('beta', 0.9)
        if 'velocity' not in params:
            params['velocity'] = [(np.zeros_like(w), np.zeros_like(b)) for w, b in zip(weights, biases)]
        for i in range(len(weights)):
            prev_velocity = params['velocity'][i]
            params['velocity'][i] = (beta * prev_velocity[0] + learning_rate * gradients[i][0],
                                     beta * prev_velocity[1] + learning_rate * gradients[i][1])
            weights[i] -= beta * prev_velocity[0] + (1 + beta) * (params['velocity'][i][0] + weight_decay * weights[i])
            biases[i] -= beta * prev_velocity[1] + (1 + beta) * params['velocity'][i][1]

    @staticmethod
    def rmsprop(weights, biases, gradients, params):
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']
        beta = params.get('beta', 0.9)
        epsilon = params.get('epsilon', 1e-8)
        if 'cache' not in params:
            params['cache'] = [(np.zeros_like(w), np.zeros_like(b)) for w, b in zip(weights, biases)]
        for i in range(len(weights)):
            params['cache'][i] = (beta * params['cache'][i][0] + (1 - beta) * (gradients[i][0] ** 2),
                                  beta * params['cache'][i][1] + (1 - beta) * (gradients[i][1] ** 2))
            weights[i] -= learning_rate * gradients[i][0] / (np.sqrt(params['cache'][i][0]) + epsilon) + weight_decay * weights[i]
            biases[i] -= learning_rate * gradients[i][1] / (np.sqrt(params['cache'][i][1]) + epsilon)


    @staticmethod
    def nadam(weights, biases, gradients, params):
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        
        if 'm' not in params:
            params['m'] = [(np.zeros_like(w), np.zeros_like(b)) for w, b in zip(weights, biases)]
            params['v'] = [(np.zeros_like(w), np.zeros_like(b)) for w, b in zip(weights, biases)]
            params['t'] = 0  # Time step
        
        params['t'] += 1
        mu_t = beta1 * (1 - 0.5 * (0.96 ** (params['t'] / 250)))

        for i in range(len(weights)):
            params['m'][i] = (beta1 * params['m'][i][0] + (1 - beta1) * gradients[i][0],
                            beta1 * params['m'][i][1] + (1 - beta1) * gradients[i][1])
            params['v'][i] = (beta2 * params['v'][i][0] + (1 - beta2) * (gradients[i][0] ** 2),
                            beta2 * params['v'][i][1] + (1 - beta2) * (gradients[i][1] ** 2))

            m_hat = (mu_t * params['m'][i][0] + (1 - mu_t) * gradients[i][0]) / (1 - beta1 ** params['t'])
            v_hat = params['v'][i][0] / (1 - beta2 ** params['t'])

            weights[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * weights[i]

            m_hat_b = (mu_t * params['m'][i][1] + (1 - mu_t) * gradients[i][1]) / (1 - beta1 ** params['t'])
            v_hat_b = params['v'][i][1] / (1 - beta2 ** params['t'])

            biases[i] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)


#confusion matrix:

def confustion_mat(y_true, y_pred):
    classes = np.unique(y_true)
    no_classes = len(classes)
    conf = np.zeros((no_classes, no_classes), dtype = int)

    for i in range(no_classes):
        for j in range(no_classes):
            conf[i, j] = np.sum((y_true == classes[i]) & (y_pred == classes[j]))
    return conf

# Neural Network:

def neuron(input, w, b):
    a = np.dot(input, w)+b
    return a

def hiddenlayer(a, activation):
    h = activation(a)
    return h

class FeedForwardNN:
    def __init__(self, input_dim, num_hl, num_neurons, output_dim, params):
        layers = [input_dim] + num_neurons + [output_dim]
        np.random.seed(42)

        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            if params['init_method'] == 'random': 
                w = np.random.randn(layers[i], layers[i+1]) * 0.01
            elif params['init_method'] == 'xavier': 
                w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(1 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward_propagation(self, X, activation_function):
        activations = [X]
        for i in range(len(self.weights) - 1):
            X = hiddenlayer(neuron(X, self.weights[i], self.biases[i]), activation_function)
            activations.append(X)
        output = softmax(neuron(X, self.weights[-1], self.biases[-1]))
        activations.append(output)
        return activations
    
    def backward_propagation(self, X, Y, optimizer, params, activation_function, activation_derivative):
        batch_size = params['batch_size']
        num_samples = X.shape[0]
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            X_batch, Y_batch = X[batch_start:batch_end], Y[batch_start:batch_end]
            activations = self.forward_propagation(X_batch, activation_function)
            gradients_w, gradients_b = [], []
            dZ = activations[-1] - Y_batch
            for i in range(len(self.weights) - 1, -1, -1):
                dW = np.dot(activations[i].T, dZ) / batch_size
                dB = np.sum(dZ, axis=0, keepdims=True) / batch_size
                gradients_w.insert(0, dW)
                gradients_b.insert(0, dB)
                if i > 0:
                    dA = np.dot(dZ, self.weights[i].T)
                    dZ = dA * activation_derivative(activations[i])
            optimizer(self.weights, self.biases, list(zip(gradients_w, gradients_b)), params)
    
    def train(self, X, Y, optimizer, params, activation_function, activation_derivative):
        
        X_train, Y_train, X_val, Y_val = X[:int(0.9 * X.shape[0])], Y[:int(0.9 * X.shape[0])], X[int(0.9 * X.shape[0]):], Y[int(0.9 * X.shape[0]):]
        for epoch in range(params['epochs']):
            self.backward_propagation(X_train, Y_train, optimizer, params, activation_function, activation_derivative)
            wandb.log({"train_loss": cross_ent_loss(Y_train, self.forward_propagation(X_train, activation_function)[-1]),
                       "train_accuracy": accuracy_calc(Y_train, self.forward_propagation(X_train, activation_function)[-1]),
                       "val_loss": cross_ent_loss(Y_val, self.forward_propagation(X_val, activation_function)[-1]),
                       "val_accuracy": accuracy_calc(Y_val, self.forward_propagation(X_val, activation_function)[-1]),
                       "epoch": epoch})
            
    def train_se(self, X, Y, optimizer, params, activation_function, activation_derivative, best_val_acc_ce):
        """Train using Squared Error Loss."""
        X_train, Y_train, X_val, Y_val = X[:int(0.9 * X.shape[0])], Y[:int(0.9 * X.shape[0])], X[int(0.9 * X.shape[0]):], Y[int(0.9 * X.shape[0]):]
        for epoch in range(params['epochs']):
            self.backward_propagation(X_train, Y_train, optimizer, params, activation_function, activation_derivative)
            output_train = self.forward_propagation(X_train, activation_function)[-1]
            output_val = self.forward_propagation(X_val, activation_function)[-1]

            loss_se_train = np.mean((output_train - Y_train) ** 2)
            loss_se_val = np.mean((output_val - Y_val) ** 2)
            train_acc = np.mean(np.argmax(output_train, axis=1) == np.argmax(Y_train, axis=1))
            val_acc = np.mean(np.argmax(output_val, axis=1) == np.argmax(Y_val, axis=1))

            

    def test(self, Xt, Yt, activation):
        pred = self.forward_propagation(Xt, activation)[-1]
        y_pred = np.argmax(pred, axis=1)
        return y_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-wp','--wandb_project',type=str,default='project', help='Project Name')
    parser.add_argument('-we','--wandb_entity',type=str,default='name',help='Name of Wandb entity')
    parser.add_argument('-d','--dataset',type=str,default='fashion_mnist',choices=['fashion_mnist','mnist'],help='Dataset choice')
    parser.add_argument('-e','--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('-b','--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-l','--loss', type=str, default='mse', choices=['mse', 'cross_ent'], help='Loss function')
    parser.add_argument('-o','--optimizer', type=str, default='nadam', choices=['adam', 'sgd','sgdm','nag','nadam'], help='Optimizer')
    parser.add_argument('-lr','--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-m','--momentum',type=float,default=0.9,help='Momentum for sgdm and nag')
    parser.add_argument('-beta','--beta',type=float,default=0.999,help='For RMSProp')
    parser.add_argument('-beta1','--beta1',type=float,default=0.9,help='Beta1 used by adam and nadam optimizers')
    parser.add_argument('-beta2','--beta2',type=float,default=0.999,help='Beta2 used by adam and nadam optimizers')
    parser.add_argument('-eps','--epsilon',type=float,default=1e-8,help='Epsilon for optimizers')
    parser.add_argument('-w_d','--weight_decay', type=float, default=0.0, help='L2 regularization coefficient')
    parser.add_argument('-w_i','--weight_init', type=str, default='xavier', choices=['xavier', 'random'], help='Weight initialization method')
    parser.add_argument('-nhl','--num_layers', type=int, default=5, help='Number of hidden layers')
    parser.add_argument('-sz','--hidden_size', type=int, default=128, help='Number of hidden units per layer')
    parser.add_argument('-a','--activation', type=str, default='sigmoid', choices=['relu', 'sigmoid','tanh','identity'], help='Activation function')
    
    args = parser.parse_args()
    config = vars(args)
    wandb.init(config=config, entity=args.wandb_entity, project=args.wandb_project)

    # Load dataset
    if args.dataset == 'fashion_mnist':
        (xTrain, yTrain), (x_test, yTest) = fashion_mnist.load_data()
    elif args.dataset == 'mnist':
        (xTrain, yTrain), (x_test, yTest) = mnist.load_data()
    else:
        raise ValueError('Choose from mnist or fashion_mnist ...')

    # Preprocessing
    xTrain, x_valid, yTrain, y_valid = train_test_split(xTrain, yTrain, test_size=0.1, random_state=0, stratify=yTrain)
    xTrain = xTrain.reshape((len(xTrain), 28*28)).astype('float32') / 255
    x_valid = x_valid.reshape((len(x_valid), 28*28)).astype('float32') / 255
    xTest = x_test.reshape((len(x_test), 28*28)).astype('float32') / 255

    # Initialize FeedForwardNN with parsed arguments
    model = FeedForwardNN(
    input_dim=784,
    num_hl=args.num_layers,
    num_neurons=[args.hidden_size] * args.num_layers,
    output_dim=10,
    params={'init_method': args.weight_init}
    )

    model.train(
        xTrain.reshape(xTrain.shape[0], -1),
        onehot(yTrain, 10),
        getattr(Optimisation(), args.optimizer),  # Fix optimizer
        {
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'batch_size': args.batch_size
        },
        getattr(Activations, args.activation),  
        getattr(Activations, f"{args.activation}_derivative")
        )

 

    y_pred = model.test(xTest.reshape(xTest.shape[0], -1), onehot(yTest, 10), getattr(Activations, config['activation']))
    print(y_pred)

    conf_mat = confustion_mat(yTest, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="RdYlBu", xticklabels=range(10), yticklabels=range(10))
    plt.figure(figsize=(8, 6))
    fig1 = sns.heatmap(conf_mat, annot=True, fmt="d", cmap="RdYlBu", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    wandb.log({"Confusion Matrix": wandb.Image(plt)})

