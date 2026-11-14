import torch
import torch.nn as nn
import pennylane as qml

class AdvancedQuantumEBM(nn.Module):
    """
    An 8-Qubit Quantum Circuit Born Machine (QCBM).
    Outputs a 256-dimensional probability distribution (2^8 states), matching the 
    size of standard latent vectors in classical Variational Autoencoders.
    """
    def __init__(self, n_qubits=8, layers=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        weight_shape = {"weights": (layers, n_qubits, 3)}
        
        @qml.qnode(self.dev, interface="torch")
        def circuit(weights):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            
            for layer in range(layers):
                for i in range(n_qubits):
                    qml.Rot(*weights[layer, i], wires=i)
                # Linear entanglement map
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
                qml.CNOT(wires=[n_qubits-1, 0])
                
            return qml.probs(wires=range(n_qubits))
            
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shape)

    def forward(self):
        # Outputs probabilities across 256 distinct quantum states
        return self.q_layer(torch.tensor([]))
