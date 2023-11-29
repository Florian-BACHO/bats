
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bats.Layers import InputLayer, LIFLayer, LIFLayerResidual
from bats.Network import Network

def build_network(N_HIDDEN_LAYERS: int, N_NEURONS_1: int, RESIDUAL_EVERY_N: int, USE_RESIDUAL: bool, N_OUTPUTS: int, TAU_S_1: float, THRESHOLD_HAT_1: float, DELTA_THRESHOLD_1: float, SPIKE_BUFFER_SIZE_1: int, TAU_S_OUTPUT: float, THRESHOLD_HAT_OUTPUT: float, DELTA_THRESHOLD_OUTPUT: float, SPIKE_BUFFER_SIZE_OUTPUT: int, weight_initializer: Callable[[int, int], cp.ndarray], network: Network, input_layer: InputLayer):
    # Build Network
    print("Creating network...")
    network = Network()
    input_layer = InputLayer(n_neurons=N_INPUTS, name="Input layer")
    network.add_layer(input_layer, input=True)

    hidden_layers = []
    for i in range(N_HIDDEN_LAYERS):
        if i == 0:
            hidden_layer = LIFLayer(previous_layer=input_layer, n_neurons=N_NEURONS_1, tau_s=TAU_S_1,
                                    theta=THRESHOLD_HAT_1,
                                    delta_theta=DELTA_THRESHOLD_1,
                                    weight_initializer=weight_initializer,
                                    max_n_spike=SPIKE_BUFFER_SIZE_1,
                                    name="Hidden layer 0")

        elif (i == N_HIDDEN_LAYERS - 1 or i % RESIDUAL_EVERY_N ==0) and N_HIDDEN_LAYERS > 5 and USE_RESIDUAL:
            hidden_layer = LIFLayerResidual(previous_layer=hidden_layers[i-1], jump_layer= input_layer, n_neurons=N_NEURONS_1, tau_s=TAU_S_1,
                                    theta=THRESHOLD_HAT_1,
                                    delta_theta=DELTA_THRESHOLD_1,
                                    weight_initializer=weight_initializer,
                                    max_n_spike=SPIKE_BUFFER_SIZE_1,
                                    name="Residual layer " + str(i))
        else:
            hidden_layer = LIFLayer(previous_layer=hidden_layers[i-1], n_neurons=N_NEURONS_1, tau_s=TAU_S_1,
                                    theta=THRESHOLD_HAT_1,
                                    delta_theta=DELTA_THRESHOLD_1,
                                    weight_initializer=weight_initializer,
                                    max_n_spike=SPIKE_BUFFER_SIZE_1,
                                    name="Hidden layer " + str(i))
        hidden_layers.append(hidden_layer)
        network.add_layer(hidden_layer)

    output_layer = LIFLayer(previous_layer=hidden_layer, n_neurons=N_OUTPUTS, tau_s=TAU_S_OUTPUT,
                            theta=THRESHOLD_HAT_OUTPUT,
                            delta_theta=DELTA_THRESHOLD_OUTPUT,
                            weight_initializer=weight_initializer,
                            max_n_spike=SPIKE_BUFFER_SIZE_OUTPUT,
                            name="Output layer")
    network.add_layer(output_layer)
    return network