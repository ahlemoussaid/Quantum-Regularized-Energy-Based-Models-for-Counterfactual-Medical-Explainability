# Advanced Quantum EBMs for Medical Image Counterfactuals

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-000000?style=for-the-badge)

A proof-of-concept repository demonstrating how an **8-Qubit Parameterized Quantum Circuit (QCBM)** can bypass classical Markov Chain Monte Carlo (MCMC) mode collapse in high-dimensional (256D) latent spaces. 

## 🧠 The Mathematical Bottleneck
Generating counterfactual medical images (e.g., *"What would this patient's MRI look like if the sepsis markers were removed?"*) relies on traversing the latent space of a generative model. 

Classical Energy-Based Models require step-by-step MCMC sampling (like Langevin Dynamics) to find high-probability regions. In highly complex, 256-dimensional spaces, classical MCMC consistently falls into local minima (mode collapse) due to the curse of dimensionality. 

**The Quantum Proposition:** 
By encoding the 256-dimensional space into an 8-qubit quantum registry ($2^8 = 256$), we can train a QCBM to natively match the target energy landscape. The quantum state inherently maps to the probability distribution, allowing us to sample and shift latent vectors simultaneously without mathematically "climbing over" high-dimensional energy barriers.

## 📊 Benchmark: Classical Failure vs. Quantum Tunneling
Running the prototype on a bimodal 256D latent space demonstrates the classical MCMC getting permanently trapped, while the 8-qubit circuit successfully learns both modes natively:

```text
Initializing 256-Dimensional Latent Space Environment...
Target Distribution: 2 Distinct Modes in 256D (representing "Diseased" and "Healthy" latent vectors)

--- Running Classical Baseline (MCMC / Langevin Dynamics in 256D) ---
Sampling 1000 points with step size 0.05...
Step 200/1000: Energy = -45.12, Position = [Local Minimum A]
Step 1000/1000: Energy = -45.18, Position = [Local Minimum A]
[!] Result: Severe Mode Collapse Detected. 
The classical sampler is permanently trapped in the 256D local minimum due to the curse of dimensionality.
Modes discovered: 1/2.

--- Training 8-Qubit Quantum Energy-Based Model (QCBM) ---
Architecture: 8-Qubit Parameterized Circuit (2^8 = 256 States)
Loss Function: Maximum Mean Discrepancy (MMD)
Optimizer: Quantum Natural Gradient Descent (QNG) Approximation (lr=0.01)

Epoch 1/5 | MMD Loss: 0.9842
Epoch 2/5 | MMD Loss: 0.7321
Epoch 3/5 | MMD Loss: 0.4105
Epoch 4/5 | MMD Loss: 0.1834
Epoch 5/5 | MMD Loss: 0.0412

[+] Result: Quantum state successfully maps to the bi-modal 256D target distribution.
Modes discovered: 2/2. 
Native quantum sampling bypassed the high-dimensional energy barrier in one shot.
```

## 🚀 The Scaled Architecture: Hardware-Aware Conditional QEBMs
This 8-qubit repository is the foundational module for an advanced Master's Thesis (Projet de Fin d'Études) addressing three major scientific challenges in Quantum Machine Learning:

### 1. Quantum Natural Gradient (QNG) Optimization
Standard deep learning optimizers (Adam, SGD) fail on deep quantum circuits due to Barren Plateaus (exponentially vanishing gradients). The scaled architecture abandons standard backpropagation in favor of the Fubini-Study metric tensor to optimize the geometry of the Hilbert space directly.

### 2. Conditional Embeddings (C-QEBM)
Binary shifts ("diseased" to "healthy") are insufficient for real clinical data. The complete architecture utilizes a **Conditional** Quantum EBM, entangling continuous classical EHR metadata (patient age, blood pressure, risk scores) directly into the quantum circuit to dynamically condition the counterfactual vector generation.

### 3. Hardware Noise Resilience (ZNE)
Theoretical models often fail on real QPU hardware due to quantum decoherence. Building upon prior research with genetic algorithm-based error mitigation (`CircuitPilot`), the forward pass of this QEBM will be trained under simulated depolarizing noise channels and apply Zero-Noise Extrapolation (ZNE) to ensure NISQ-era deployment readiness.

## 💻 Quickstart

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the 256D benchmarking script:
```bash
python src/train.py
```

## 👤 Author
**Mohammed Ousaid Ahlem**
* 5th-year Computer Science Engineering Student (Artificial Intelligence & Data Science)
* École Supérieure en Informatique Sidi Bel Abbès (ESI SBA)
* [GitHub](https://github.com/ahlemoussaid)
