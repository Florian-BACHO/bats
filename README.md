# Error Backpropagation Through Spikes (BATS)

Error Backpropagation Through Spikes (BATS) [1] is a GPU-compatible algorithm that extends Fast & Deep [2], 
a method to performs exact gradient descent in Deep Spiking Neural Networks (SNNs). 
In contrast with Fast & Deep, BATS allows error backpropagation with multiple spikes per neuron, leading to increased 
performances. The proposed algorithm backpropagates the errors through post-synaptic spikes with linear time complexity 
<em>O(N)</em> making the error backpropagation process fast for multi-spike SNNs.<br>
This repository contains the full Cuda implementations of our efficient event-based SNN simulator and the BATS algorithm.
All the experiments on the convergence of single and multi-spike models, on the MNIST dataset, its extended version 
EMNIST and Fashion MNIST are also provided to reproduce our results. 

## Dependencies and Libraries

Recommanded Python version: >= 3.8

Libraries:
- Cuda (we suggest Cuda 10.1 as this is the version that we used to develop BATS 
  but other versions should also work)
  
Python packages:
- CuPy [3] (corresponding to the installed version of Cuda)
- matplotlib (<em>Optional</em>. Install only if generate plots with monitors)
- requests (<em>Optional</em>. Install only if run the scripts to download the 
  experiments' datasets)
- scipy (<em>Optional</em>. Install only if run the EMNIST experiment)
- brian2 (<em>Optional</em>. Install only if run the unit tests)

## Experiments

Three experiments are available: a single-spike vs multi-spike convergence 
experiment as well as trainings on the MNIST and EMNIST datasets.

### Convergence experiment

```console
$ cd experiments/convergence
$ ls
convergence_count.py  convergence_ttfs.py
$ python3 convergence_count.py
...
$ python3 convergence_ttfs.py
...
$ ls *.pdf
convergence_count.pdf  convergence_ttfs.pdf
```

Running these python scripts should take a few minutes. 
The models are trained 100 times for different initial weight distributions.
After execution, each script generates a corresponding .pdf file, 
i.e. <em>convergence_count.pdf</em> and <em>convergence_ttfs.pdf</em>

### Download datasets

#### MNIST

```console
$ cd datasets
$ ls
download_file.py  get_emnist.py  get_fashion_mnist.py  get_mnist.py
$ python3 get_mnist.py
Downloading MNIST...
[██████████████████████████████████████████████████]
Done.
$ ls
download_file.py  get_emnist.py  get_fashion_mnist.py get_mnist.py  mnist.npz
```

#### EMNIST

```console
$ cd datasets
$ ls
download_file.py  get_emnist.py  get_fashion_mnist.py  get_mnist.py
$ python3 get_emnist.py
Downloading EMNIST...
[██████████████████████████████████████████████████]
Done.
Extracting EMNIST...
Done.
Cleaning...
Done.
$ ls
download_file.py  emnist-balanced.mat  get_emnist.py  get_fashion_mnist.py  get_mnist.py
```
Downloading the EMNIST dataset may take a few minutes due to the size of the file.

#### Fashion MNIST

```console
$ cd datasets
$ ls
download_file.py     get_emnist.py        get_fashion_mnist.py get_mnist.py
$ python3 get_fashion_mnist.py
Downloading Fashion MNIST...
[██████████████████████████████████████████████████]
[██████████████████████████████████████████████████]
[██████████████████████████████████████████████████]
[██████████████████████████████████████████████████]
Done.
$ ls
download_file.py           get_fashion_mnist.py       t10k-images-idx3-ubyte.gz  train-images-idx3-ubyte.gz
get_emnist.py              get_mnist.py               t10k-labels-idx1-ubyte.gz  train-labels-idx1-ubyte.gz
```

### Train models

#### MNIST

Tree models are available to train with the MNIST dataset: 
- a single-spike model (<em>train_ttfs.py</em>)
```console
$ cd experiments/mnist
$ python3 train_ttfs.py
...
```
- a multi-spike model (<em>train_spike_count.py</em>)
```console
$ cd experiments/mnist
$ python3 train_spike_count.py
...
```
- and a Convolutional SNN  (<em>train_conv.py</em>)
```console
$ cd experiments/mnist
$ python3 train_conv.py
...
```

During training, plots and data are saved in the <em>output_metrics</em> directory
and weights of the best model are saved in the <em>best_model</em> directory.

#### EMNIST

```console
$ cd experiments/emnist
$ python3 train.py
...
```
Similarly to the MNIST training, plots are saved in the <em>output_metrics</em> directory
and weights of the best model are saved in the <em>best_model</em> directory.

## Unit tests

Unit tests can be run by executing the following command:
```console
$ pipenv run python3 -m unittest discover -s tests/ -p "*.py"
.....................
----------------------------------------------------------------------
Ran 21 tests in 74.382s

OK
```
Some tests use the brian2 [4] clock-based simulator as truth for our event-based simulator.

## References

[1] Bacho, F., & Chu, D.. (2022). Exact Error Backpropagation Through Spikes for Precise Training of Spiking Neural Networks. https://arxiv.org/abs/2212.09500 <br>
[2] J. Göltz, L. Kriener, A. Baumbach, S. Billaudelle, O. Breitwieser, B. Cramer, D. Dold, A. F. Kungl, W. Senn, J. Schemmel, K. Meier, & M. A. Petrovici (2021). Fast and energy-efficient neuromorphic deep learning with first-spike times. <em>Nature Machine Intelligence, 3(9), 823–835.</em> <br>
[3] Okuta, R., Unno, Y., Nishino, D., Hido, S., & Loomis, C. (2017). CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations. In <em>Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS).</em> <br>
[4] Stimberg, M., Brette, R., & Goodman, D. (2019). Brian 2, an intuitive and efficient neural simulator. <em>eLife, 8, e47314.<em>