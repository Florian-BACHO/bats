from pathlib import Path
from typing import Tuple
from scipy import io as spio

import numpy as np

TIME_WINDOW = 100e-3
MAX_VALUE = 255
RESOLUTION = 28
N_NEURONS = RESOLUTION * RESOLUTION
# Data augmentation
MAX_AUGMENT_OFFSET = 2
TWO_AUGMENT_OFFSET = 2 * MAX_AUGMENT_OFFSET
PAD_WIDTH = [(0, 0),
             (MAX_AUGMENT_OFFSET, MAX_AUGMENT_OFFSET),
             (MAX_AUGMENT_OFFSET, MAX_AUGMENT_OFFSET)]
END_PADDED_IMG = MAX_AUGMENT_OFFSET + RESOLUTION
MAX_ROTATION = 5

class Dataset:
    def __init__(self, path: Path, n_train_samples: int = None, n_test_samples: int = None, bias=False):
        emnist = spio.loadmat(path)

        train_X = emnist["dataset"][0][0][0][0][0][0]
        train_X = train_X.astype(np.float32)
        self.__train_labels = emnist["dataset"][0][0][0][0][0][1].flatten()

        test_X = emnist["dataset"][0][0][1][0][0][0]
        test_X = test_X.astype(np.float32)

        # load test labels
        self.__test_labels = emnist["dataset"][0][0][1][0][0][1].flatten()

        self.__train_spike_times, self.__train_n_spike_per_neuron = self.__to_spikes(train_X)
        self.__test_spike_times, self.__test_n_spike_per_neuron = self.__to_spikes(test_X)

        if n_train_samples is not None:
            self.__train_spike_times, self.__train_n_spike_per_neuron, self.__train_labels = \
                self.__reduce_data(self.__train_spike_times, self.__train_n_spike_per_neuron, self.__train_labels,
                                   n_train_samples)

        if n_test_samples is not None:
            self.__test_spike_times, self.__test_n_spike_per_neuron, self.__test_labels = \
                self.__reduce_data(self.__test_spike_times, self.__test_n_spike_per_neuron, self.__test_labels,
                                   n_test_samples)

        if bias:
            self.__train_spike_times, self.__train_n_spike_per_neuron = \
                self.__add_bias(self.__train_spike_times, self.__train_n_spike_per_neuron)
            self.__test_spike_times, self.__test_n_spike_per_neuron = \
                self.__add_bias(self.__test_spike_times, self.__test_n_spike_per_neuron)

    def __reduce_data(self, spike_times, n_spike_per_neuron, labels, n):
        shuffled_indices = np.arange(len(self.__train_labels))
        np.random.shuffle(shuffled_indices)
        shuffled_indices = shuffled_indices[:n]
        return spike_times[shuffled_indices], n_spike_per_neuron[shuffled_indices], labels[shuffled_indices]

    def __add_bias(self, spike_times, n_spike_per_neuron):
        new_neuron = np.zeros((spike_times.shape[0], 1, 1))
        new_neuron_n_spikes = np.ones((spike_times.shape[0], 1))
        return np.append(spike_times, new_neuron, axis=1), np.append(n_spike_per_neuron, new_neuron_n_spikes, axis=1)

    @property
    def train_labels(self) -> np.ndarray:
        return self.__train_labels

    @property
    def test_labels(self) -> np.ndarray:
        return self.__test_labels

    def __to_spikes(self, samples):
        spike_times = samples.reshape((samples.shape[0], N_NEURONS, 1))
        spike_times = TIME_WINDOW * (1 - (spike_times / MAX_VALUE))
        spike_times[spike_times == TIME_WINDOW] = np.inf
        n_spike_per_neuron = np.isfinite(spike_times).astype('int').reshape((samples.shape[0], N_NEURONS))
        return spike_times, n_spike_per_neuron

    def shuffle(self) -> None:
        shuffled_indices = np.arange(len(self.__train_labels))
        np.random.shuffle(shuffled_indices)
        self.__train_spike_times = self.__train_spike_times[shuffled_indices]
        self.__train_n_spike_per_neuron = self.__train_n_spike_per_neuron[shuffled_indices]
        self.__train_labels = self.__train_labels[shuffled_indices]

    def __get_batch(self, spike_times, n_spikes_per_neuron, labels, batch_index, batch_size) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = batch_index * batch_size
        end = start + batch_size

        spikes_per_neuron = spike_times[start:end]
        n_spikes_per_neuron = n_spikes_per_neuron[start:end]
        labels = labels[start:end]

        return spikes_per_neuron, n_spikes_per_neuron, labels

    def get_train_batch(self, batch_index: int, batch_size: int) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.__get_batch(self.__train_spike_times, self.__train_n_spike_per_neuron, self.__train_labels,
                                batch_index, batch_size)

    def get_test_batch(self, batch_index: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.__get_batch(self.__test_spike_times, self.__test_n_spike_per_neuron, self.__test_labels,
                                batch_index, batch_size)
