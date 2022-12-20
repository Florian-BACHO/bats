import csv
import time
from pathlib import Path
import cupy as cp

# Dataset paths
import matplotlib.pyplot as plt
import numpy as np

from bats import AbstractOptimizer, AbstractLoss, AbstractLayer
from bats.Layers import InputLayer, LIFLayer
from bats.Losses import TTFSSoftmaxCrossEntropy
from bats.Network import Network
from bats.Optimizers import GradientDescentOptimizer

# Dataset
SPIKE_TIMES = np.array([[[0.0],
                         [0.0],
                         [1.0],
                         [2.0]],
                        [[0.0],
                         [0.0],
                         [2.0],
                         [1.0]]
                        ])
N_SPIKE_TIMES = np.array([[1, 1, 1, 1],
                          [1, 1, 1, 1]])
LABELS = np.array([0, 1])
LABELS_GPU = cp.array(LABELS, dtype=cp.int32)

# Model parameters
N_INPUTS = 4
SIMULATION_TIME = 10.0

# Output_layer
N_OUTPUTS = 2
TAU_S_OUTPUT = 1.0
THRESHOLD_HAT_OUTPUT = 1.0
DELTA_THRESHOLD_OUTPUT = THRESHOLD_HAT_OUTPUT
SPIKE_BUFFER_SIZE_OUTPUT = 1

# Training parameters
N_TRAINING_EPOCHS = 50
TAU_LOSS = 0.1
LEARNING_RATE = 1e-1  # np.full((3,), 1e-2)

RANGE_MEAN = [0.0, 2.5, 0.2]
RANGE_STD = [0.0, 2.5, 0.2]
N_REPEAT = 100


def get_predictions(output_spikes: cp.ndarray) -> cp.ndarray:
    return cp.argmin(output_spikes[:, :, 0], axis=1)


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.sum(predictions == labels) / len(labels)


def train(network: Network, output_layer: AbstractLayer, loss_fct: AbstractLoss, optimizer: AbstractOptimizer,
          weight_mean, weight_std):
    weights = np.random.normal(loc=weight_mean, scale=weight_std, size=(N_OUTPUTS, N_INPUTS))
    output_layer.weights = weights
    best_accuracy = 0.0
    for epoch in range(N_TRAINING_EPOCHS):
        # Training inference
        network.reset()
        network.forward(SPIKE_TIMES, N_SPIKE_TIMES, max_simulation=SIMULATION_TIME)
        out_spikes, n_out_spikes = network.output_spike_trains

        # Metrics
        pred = get_predictions(out_spikes)
        loss, errors = loss_fct.compute_loss_and_errors(out_spikes, n_out_spikes, LABELS_GPU)

        gradient = network.backward(errors, cp.array(LABELS))

        avg_gradient = [None if g is None else cp.mean(g, axis=0) for g in gradient]
        deltas = optimizer.step(avg_gradient)
        network.apply_deltas(deltas)

        acc = accuracy(pred.get(), LABELS) * 100
        if acc > best_accuracy:
            best_accuracy = acc
    return best_accuracy


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 16})

    print("Creating network...")
    network = Network()
    input_layer = InputLayer(n_neurons=N_INPUTS, name="Input layer")
    network.add_layer(input_layer, input=True)

    output_layer = LIFLayer(previous_layer=input_layer, n_neurons=N_OUTPUTS, tau_s=TAU_S_OUTPUT,
                            theta=THRESHOLD_HAT_OUTPUT,
                            delta_theta=DELTA_THRESHOLD_OUTPUT,
                            max_n_spike=SPIKE_BUFFER_SIZE_OUTPUT,
                            name="Output layer")
    network.add_layer(output_layer)

    loss_fct = TTFSSoftmaxCrossEntropy(TAU_LOSS)
    optimizer = GradientDescentOptimizer(LEARNING_RATE)

    all_means = np.arange(RANGE_MEAN[0], RANGE_MEAN[1], RANGE_MEAN[2])
    all_std = np.arange(RANGE_STD[0], RANGE_STD[1], RANGE_STD[2])

    csvfile = open("./convergence_ttfs.csv", 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["Mean", "Std", "Accuracy"])

    for std in all_std:
        for mean in all_means:
            all_acc = 0.0
            for _ in range(N_REPEAT):
                all_acc += train(network, output_layer, loss_fct, optimizer, mean, std)
            avg_acc = all_acc / N_REPEAT
            print(mean, std, avg_acc)
            writer.writerow([mean, std, avg_acc])
        writer.writerow([])
