from pathlib import Path
import cupy as cp
import numpy as np

import sys

sys.path.insert(0, "../../")  # Add repository root to python path

from Dataset import Dataset
from bats.Monitors import *
from bats.Layers import LIFLayer
from bats.Losses import *
from bats.Network import Network
from bats.Optimizers import *
from bats.Layers.ConvInputLayer import ConvInputLayer
from bats.Layers.ConvLIFLayer import ConvLIFLayer
from bats.Layers.PoolingLayer import PoolingLayer

# Dataset
DATASET_PATH = Path("../../datasets/mnist.npz")

INPUT_SHAPE = np.array([28, 28, 1])
N_INPUTS = 28 * 28
SIMULATION_TIME = 0.2

FILTER_1 = np.array([5, 5, 15])
TAU_S_1 = 0.130
THRESHOLD_HAT_1 = 0.04
DELTA_THRESHOLD_1 = 1 * THRESHOLD_HAT_1
SPIKE_BUFFER_SIZE_1 = 1

FILTER_2 = np.array([5, 5, 40])
TAU_S_2 = 0.130
THRESHOLD_HAT_2 = 0.8
DELTA_THRESHOLD_2 = 1 * THRESHOLD_HAT_2
SPIKE_BUFFER_SIZE_2 = 3

N_NEURONS_FC = 300
TAU_S_FC = 0.130
THRESHOLD_HAT_FC = 0.6
DELTA_THRESHOLD_FC = 1 * THRESHOLD_HAT_FC
SPIKE_BUFFER_SIZE_FC = 10

# Output_layer
N_OUTPUTS = 10
TAU_S_OUTPUT = 0.130
THRESHOLD_HAT_OUTPUT = 0.3
DELTA_THRESHOLD_OUTPUT = 1 * THRESHOLD_HAT_OUTPUT
SPIKE_BUFFER_SIZE_OUTPUT = 30

# Training parameters
N_TRAINING_EPOCHS = 100
N_TRAIN_SAMPLES = 60000
N_TEST_SAMPLES = 10000
TRAIN_BATCH_SIZE = 20
TEST_BATCH_SIZE = 50
N_TRAIN_BATCH = int(N_TRAIN_SAMPLES / TRAIN_BATCH_SIZE)
N_TEST_BATCH = int(N_TEST_SAMPLES / TEST_BATCH_SIZE)
TRAIN_PRINT_PERIOD = 0.1
TRAIN_PRINT_PERIOD_STEP = int(N_TRAIN_SAMPLES * TRAIN_PRINT_PERIOD / TRAIN_BATCH_SIZE)
TEST_PERIOD = 1.0  # Evaluate on test batch every TEST_PERIOD epochs
TEST_PERIOD_STEP = int(N_TRAIN_SAMPLES * TEST_PERIOD / TRAIN_BATCH_SIZE)
LEARNING_RATE = 0.003
LR_DECAY_EPOCH = 10  # Perform decay very n epochs
LR_DECAY_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-4
TARGET_FALSE = 3
TARGET_TRUE = 30

# Plot parameters
EXPORT_METRICS = True
EXPORT_DIR = Path("./output_metrics")
SAVE_DIR = Path("./best_model")


def weight_initializer_conv(c: int, x: int, y: int, pre_c: int) -> cp.ndarray:
    return cp.random.uniform(-1.0, 1.0, size=(c, x, y, pre_c), dtype=cp.float32)


def weight_initializer_ff(n_post: int, n_pre: int) -> cp.ndarray:
    return cp.random.uniform(-1.0, 1.0, size=(n_post, n_pre), dtype=cp.float32)


if __name__ == "__main__":
    max_int = np.iinfo(np.int32).max
    np_seed = np.random.randint(low=0, high=max_int)
    cp_seed = np.random.randint(low=0, high=max_int)
    np.random.seed(np_seed)
    cp.random.seed(cp_seed)
    print(f"Numpy seed: {np_seed}, Cupy seed: {cp_seed}")

    if EXPORT_METRICS and not EXPORT_DIR.exists():
        EXPORT_DIR.mkdir()

    print("Loading datasets...")
    dataset = Dataset(DATASET_PATH)  # , n_train_samples=N_TRAIN_SAMPLES, n_test_samples=N_TEST_SAMPLES)

    print("Creating network...")
    network = Network()

    input_layer = ConvInputLayer(neurons_shape=INPUT_SHAPE, name="Input layer")
    network.add_layer(input_layer, input=True)

    conv_1 = ConvLIFLayer(previous_layer=input_layer, filters_shape=FILTER_1, tau_s=TAU_S_1,
                          theta=THRESHOLD_HAT_1,
                          delta_theta=DELTA_THRESHOLD_1,
                          weight_initializer=weight_initializer_conv,
                          max_n_spike=SPIKE_BUFFER_SIZE_1,
                          name="Convolution 1")
    network.add_layer(conv_1)

    pool_1 = PoolingLayer(conv_1, name="Pooling 1")
    network.add_layer(pool_1)

    conv_2 = ConvLIFLayer(previous_layer=pool_1, filters_shape=FILTER_2, tau_s=TAU_S_2,
                          theta=THRESHOLD_HAT_2,
                          delta_theta=DELTA_THRESHOLD_2,
                          weight_initializer=weight_initializer_conv,
                          max_n_spike=SPIKE_BUFFER_SIZE_2,
                          name="Convolution 2")
    network.add_layer(conv_2)

    pool_2 = PoolingLayer(conv_2, name="Pooling 2")
    network.add_layer(pool_2)

    feedforward = LIFLayer(previous_layer=pool_2, n_neurons=N_NEURONS_FC, tau_s=TAU_S_FC,
                           theta=THRESHOLD_HAT_FC,
                           delta_theta=DELTA_THRESHOLD_FC,
                           weight_initializer=weight_initializer_ff,
                           max_n_spike=SPIKE_BUFFER_SIZE_FC,
                           name="Feedforward 1")
    network.add_layer(feedforward)

    output_layer = LIFLayer(previous_layer=feedforward, n_neurons=N_OUTPUTS, tau_s=TAU_S_OUTPUT,
                            theta=THRESHOLD_HAT_OUTPUT,
                            delta_theta=DELTA_THRESHOLD_OUTPUT,
                            weight_initializer=weight_initializer_ff,
                            max_n_spike=SPIKE_BUFFER_SIZE_OUTPUT,
                            name="Output layer")
    network.add_layer(output_layer)

    loss_fct = SpikeCountClassLoss(target_false=TARGET_FALSE, target_true=TARGET_TRUE)
    optimizer = AdamOptimizer(learning_rate=LEARNING_RATE)

    # Metrics
    training_steps = 0
    train_loss_monitor = LossMonitor(export_path=EXPORT_DIR / "loss_train")
    train_accuracy_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "accuracy_train")
    train_silent_label_monitor = SilentLabelsMonitor()
    train_time_monitor = TimeMonitor()
    train_monitors_manager = MonitorsManager([train_loss_monitor,
                                              train_accuracy_monitor,
                                              train_silent_label_monitor,
                                              train_time_monitor],
                                             print_prefix="Train | ")

    test_loss_monitor = LossMonitor(export_path=EXPORT_DIR / "loss_test")
    test_accuracy_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "accuracy_test")
    test_learning_rate_monitor = ValueMonitor(name="Learning rate", decimal=5)
    # Only monitor LIF layers
    test_spike_counts_monitors = {l: SpikeCountMonitor(l.name) for l in network.layers
                                  if (isinstance(l, LIFLayer) or isinstance(l, ConvLIFLayer))}
    test_silent_monitors = {l: SilentNeuronsMonitor(l.name) for l in network.layers
                            if (isinstance(l, LIFLayer) or isinstance(l, ConvLIFLayer))}
    test_time_monitor = TimeMonitor()
    all_test_monitors = [test_loss_monitor, test_accuracy_monitor, test_learning_rate_monitor]
    all_test_monitors.extend(test_spike_counts_monitors.values())
    all_test_monitors.extend(test_silent_monitors.values())
    all_test_monitors.append(test_time_monitor)
    test_monitors_manager = MonitorsManager(all_test_monitors,
                                            print_prefix="Test | ")

    best_acc = 0.0
    print("Training...")
    for epoch in range(N_TRAINING_EPOCHS):
        train_time_monitor.start()
        dataset.shuffle()

        # Learning rate decay
        if epoch > 0 and epoch % LR_DECAY_EPOCH == 0:
            optimizer.learning_rate = np.maximum(LR_DECAY_FACTOR * optimizer.learning_rate, MIN_LEARNING_RATE)

        for batch_idx in range(N_TRAIN_BATCH):
            # Get next batch
            spikes, n_spikes, labels = dataset.get_train_batch(batch_idx, TRAIN_BATCH_SIZE, augment=True)

            # Inference
            network.reset()
            network.forward(spikes, n_spikes, max_simulation=SIMULATION_TIME, training=True)
            out_spikes, n_out_spikes = network.output_spike_trains

            # Predictions, loss and errors
            pred = loss_fct.predict(out_spikes, n_out_spikes)
            loss, errors = loss_fct.compute_loss_and_errors(out_spikes, n_out_spikes, labels)

            pred_cpu = pred.get()
            loss_cpu = loss.get()
            n_out_spikes_cpu = n_out_spikes.get()

            # Update monitors
            train_loss_monitor.add(loss_cpu)
            train_accuracy_monitor.add(pred_cpu, labels)
            train_silent_label_monitor.add(n_out_spikes_cpu, labels)

            # Compute gradient
            gradient = network.backward(errors)
            avg_gradient = [None if g is None else cp.mean(g, axis=0) for g, layer in zip(gradient, network.layers)]
            del gradient

            # Apply step
            deltas = optimizer.step(avg_gradient)
            del avg_gradient

            network.apply_deltas(deltas)
            del deltas

            training_steps += 1
            epoch_metrics = training_steps * TRAIN_BATCH_SIZE / N_TRAIN_SAMPLES

            # Training metrics
            if training_steps % TRAIN_PRINT_PERIOD_STEP == 0:
                # Compute metrics

                train_monitors_manager.record(epoch_metrics)
                train_monitors_manager.print(epoch_metrics)
                train_monitors_manager.export()

            # Test evaluation
            if training_steps % TEST_PERIOD_STEP == 0:
                test_time_monitor.start()
                for batch_idx in range(N_TEST_BATCH):
                    spikes, n_spikes, labels = dataset.get_test_batch(batch_idx, TEST_BATCH_SIZE)
                    network.reset()
                    network.forward(spikes, n_spikes, max_simulation=SIMULATION_TIME)
                    out_spikes, n_out_spikes = network.output_spike_trains

                    pred = loss_fct.predict(out_spikes, n_out_spikes)
                    loss = loss_fct.compute_loss(out_spikes, n_out_spikes, labels)

                    pred_cpu = pred.get()
                    loss_cpu = loss.get()
                    test_loss_monitor.add(loss_cpu)
                    test_accuracy_monitor.add(pred_cpu, labels)

                    for l, mon in test_spike_counts_monitors.items():
                        mon.add(l.spike_trains[1])

                    for l, mon in test_silent_monitors.items():
                        mon.add(l.spike_trains[1])

                test_learning_rate_monitor.add(optimizer.learning_rate)

                records = test_monitors_manager.record(epoch_metrics)
                test_monitors_manager.print(epoch_metrics)
                test_monitors_manager.export()

                acc = records[test_accuracy_monitor]
                if acc > best_acc:
                    best_acc = acc
                    network.store(SAVE_DIR)
                    print(f"Best accuracy: {np.around(best_acc, 2)}%, Networks save to: {SAVE_DIR}")
