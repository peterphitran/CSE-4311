import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Layer Class
class Layer: 

    def forward_pass(self, X):
        return

    def backward_pass(self, gradient):
        return

    def update(self, lr):
        return

# Linear Layer
class LinearLayer(Layer):

    # X = number of samples x number of input features
    # W = number of output features x number of input features
    # bias number of output features

    def __init__(self, weights, bias):
        self.W = weights
        self.b = bias
    
    def forward_pass(self,X):
        self.X = X
        return np.dot(self.X , self.W.T) + self.b

    def backward_pass(self, gradient):
        self.weight_gradient = np.dot(gradient.T , self.X)
        self.bias_gradient = np.sum(gradient, axis=0)
        return np.dot(gradient, self.W)

    def update(self, lr):
        self.W -= lr * self.weight_gradient
        self.b -= lr * self.bias_gradient


# Sigmoid Function
class Sigmoid(Layer):

    def forward_pass(self, X):
        self.z = 1 / (1 + np.exp(-X))
        return self.z

    def backward_pass(self, gradient):
        return gradient * self.z * (1 - self.z)

# ReLU 
class ReLU(Layer):

    def forward_pass(self, X):
        self.positive = (X > 0)
        return np.maximum(0, X)

    def backward_pass(self, gradient):
        return gradient * self.positive

class Tanh(Layer):

    def forward_pass(self, X):
        self.outputs = np.tanh(X)
        return self.outputs

    def backward_pass(self, gradient):
        return gradient * (1 - self.outputs**2)

# Binary Cross Entropy Lossi
class BCE(Layer):

    def forward_pass(self, X):
        predictions, targets = X
        predictions = np.array(predictions)
        targets = np.array(targets).reshape(predictions.shape)

        self.predictions = predictions
        self.targets = targets

        # -(y log y^ + (1-y) log (1-y^))
        return np.mean(-(targets * np.log(self.predictions)+ (1 - targets) * np.log(1 - self.predictions)))

    def backward_pass(self, gradient):
        count = self.targets.size
        # (y^ - y) / y^(1-y^)
        return gradient * (self.predictions - self.targets) / (self.predictions * (1-self.predictions) * count)

# Mean Squared Eror for taxi dataset 
class MSE(Layer):

    def forward_pass(self, X):
        predictions, targets = X
        predictions = np.array(predictions)
        targets = np.array(targets).reshape(predictions.shape)

        self.difference = predictions - targets
        return np.mean(self.difference ** 2)

    def backward_pass(self, gradient):
        return gradient *2 * self.difference / self.difference.size

# Sequential Class
class Sequential(Layer):

    def __init__(self):
        self.layers = []

    def append(self, layer):
        self.layers.append(layer)
    
    def forward_pass(self, X):
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X

    def backward_pass(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward_pass(gradient)
        return gradient

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

class HandleWeights:

    def __init__(self, network):
        self.network = network

    def save_weights(self, filename):
        parameters =  {}

        for index, layer in enumerate(self.network.layers):
            if isinstance(layer,LinearLayer):
                parameters["W_"+ str(index)] = layer.W
                parameters["b_"+ str(index)] = layer.b

        with open(filename, "wb") as f:
            np.savez(f, **parameters)

    def load_weights(self, filename):
        with np.load(filename) as parameters:
            loaded = []
            for index, layer in enumerate(self.network.layers):
                if isinstance(layer, LinearLayer):
                    weights_name = "W_" + str(index)
                    bias_name = "b_" + str(index)

                    weights = parameters[weights_name]
                    bias = parameters[bias_name]
                    loaded.append((layer, weights.copy(), bias.copy()))

        for layer, weights, bias in loaded:
            layer.W = weights
            layer.b = bias

def train_xor(hidden_activation, seed):
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    # y
    targets = np.array([[0], [1],[1],[0]])

    lr = 1.0

    # set up our nn 
    rng = np.random.default_rng(seed)
    network = Sequential()
    network.append(LinearLayer(rng.normal(0, 0.5, (2, 2)), np.zeros(2)))
    network.append(hidden_activation())
    network.append(LinearLayer(rng.normal(0, 0.5, (1, 2)), np.zeros(1)))
    network.append(Sigmoid())

    loss_function = BCE()

    for epoch in range(1000):
        predictions = network.forward_pass(X)
        loss = loss_function.forward_pass((predictions, targets))

        guesses = (predictions >= 0.5).astype(int)
        if loss < 0.1 and np.array_equal(guesses, targets):
            return network, epoch + 1, X

        gradient = loss_function.backward_pass(1.0)
        network.backward_pass(gradient)
        network.update(lr)

def eval_xor():
    activations = ((Sigmoid, "Sigmoid"), (Tanh, "Tanh"))
    for activation, activation_name in activations:
        result = train_xor(activation, seed=42)

        if result is None:
            print(activation_name, "did not converge")
            continue

        network, epochs, X = result
        predictions = network.forward_pass(X)
        guesses = (predictions >= 0.5).astype(int)

        print(activation_name)
        print("Epochs:", epochs)
        print("Predictions:", predictions.reshape(-1))
        print("Classes:", guesses.reshape(-1))

        if activation is Tanh:
            HandleWeights(network).save_weights("XOR_solved.w")

def make_taxi_network(input_count, hidden_count, seed, hidden_activation):
    rng = np.random.default_rng(seed)
    network = Sequential()
    network.append(LinearLayer(rng.normal(0,0.5, (hidden_count, input_count)), np.zeros(hidden_count)))
    network.append(hidden_activation())
    network.append(LinearLayer(rng.normal(0, 0.5, (1, hidden_count)),np.zeros(1)))

    return network

def train_taxi_model(network, X_train, y_train, X_validation, y_validation):
    loss_function = MSE()

    batch_size = 4096
    lr = 0.001
    best_validation_loss = float("inf")
    best_parameters = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(50):
        epoch_loss_total = 0

        for start in range(0, len(X_train), batch_size):
            end = start + batch_size
            predictions = network.forward_pass(X_train[start:end])
            batch_loss = loss_function.forward_pass((predictions, y_train[start:end]))

            gradient = loss_function.backward_pass(1.0)
            network.backward_pass(gradient)
            network.update(lr)

            epoch_loss_total += batch_loss * (end - start)

        training_loss = epoch_loss_total / len(X_train)

        # validating for overfitting
        validation_predictions = network.forward_pass(X_validation)
        validation_loss = loss_function.forward_pass((validation_predictions, y_validation))
        history.append((training_loss, validation_loss))

        print(
            f"Epoch {epoch + 1}: "
            f"training MSE={training_loss:.4f}, "
            f"validation MSE={validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_parameters = [
                (layer.W.copy(), layer.b.copy())
                for layer in network.layers
                if isinstance(layer, LinearLayer)
            ]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == 3:
                break

    linear_layers = [
        layer for layer in network.layers
        if isinstance(layer, LinearLayer)
    ]
    for layer, (weights, bias) in zip(linear_layers, best_parameters):
        layer.W = weights
        layer.b = bias

    return history, best_validation_loss

def main():
    eval_xor()
    print()

    dataset = np.load("nyc_taxi_data.npy", allow_pickle=True).item()
    X_train, y_train, X_test, y_test = dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"] 

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    feature_columns = [
        "vendor_id",
        "passenger_count",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]

    # normalization
    X_train = X_train[feature_columns].to_numpy(dtype=float)
    X_validation = X_validation[feature_columns].to_numpy(dtype=float)
    X_test = X_test[feature_columns].to_numpy(dtype=float)

    y_train = y_train.to_numpy(dtype=float).reshape(-1, 1)
    y_validation = y_validation.to_numpy(dtype=float).reshape(-1, 1)
    y_test = y_test.to_numpy(dtype=float).reshape(-1, 1)

    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)

    # z score & log 
    X_train = (X_train - feature_mean) / feature_std
    X_validation = (X_validation - feature_mean) / feature_std
    X_test = (X_test - feature_mean) / feature_std

    y_train = np.log1p(y_train)
    y_validation = np.log1p(y_validation)

    # training taxi
    taxi_models = []
    configurations = (
        (8, ReLU, 42, "ReLU"), (16, ReLU, 42, "ReLU"), (32, ReLU, 42, "ReLU"),
        (8, Tanh, 42, "Tanh"), (16, Tanh, 42, "Tanh"), (32, Tanh, 42, "Tanh"),
    )

    # train
    for hidden_count, activation, seed, activation_name in configurations:
        taxi_network = make_taxi_network(X_train.shape[1], hidden_count, seed=seed, hidden_activation=activation)
        history, best_validation_loss = train_taxi_model(taxi_network,X_train, y_train, X_validation, y_validation)
        taxi_models.append((hidden_count, activation_name, taxi_network,best_validation_loss))
        print(
            f"{hidden_count} {activation_name} hidden nodes: "
            f"best validation MSE={best_validation_loss:.4f}"
        )

        epochs = range(1, len(history) + 1)
        loss_values = np.array(history)
        training_losses = loss_values[:, 0]
        validation_losses = loss_values[:, 1]
        plt.plot(epochs, training_losses,label="Training")
        plt.plot(epochs, validation_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("MSE on log duration")
        plt.title(f"Taxi model: {hidden_count} {activation_name} hidden nodes")
        plt.legend()
        plot_name = f"taxi_{hidden_count}_{activation_name.lower()}_loss.png"
        plt.savefig(plot_name)
        plt.close()

    # eval
    for hidden_count, activation_name, network, validation_loss in taxi_models:
        predicted_log_duration = network.forward_pass(X_test)

        y_actual = y_test.reshape(-1)
        y_actual_log = np.log1p(y_actual)
        y_pred_log = predicted_log_duration.reshape(-1)
        log_mse = mean_squared_error(y_actual_log, y_pred_log)

        print(f"{hidden_count}-hidden-node {activation_name} network")
        print(f"Validation MSE: {validation_loss:.4f}")
        print(f"Test MSE: {log_mse:.4f}")
        print(f"Test RMSE: {np.sqrt(log_mse):.4f}")
        print(f"Test MAE: {mean_absolute_error(y_actual_log, y_pred_log):.4f}")
        print(f"Test R^2: {r2_score(y_actual_log, y_pred_log):.4f}")
        print("\n")

if __name__ == "__main__":
    main()


