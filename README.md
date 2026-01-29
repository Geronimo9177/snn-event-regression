# Spiking Neural Networks for Event-Based Regression

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align="center">
  <img src="docs/images/architecture_overview.png" alt="Architecture Overview" width="800"/>
</p>

Official implementation of **"Spiking Neural Networks for Event-Based Regression: Analysis of Neuron Dynamics and Normalization Strategies"**.

Event-based cameras provide asynchronous visual measurements with microsecond temporal resolution. This repository explores deep Spiking Neural Networks (SNNs) for continuous-valued regression on event streams, analyzing the fundamental design trade-offs between neuron dynamics, normalization mechanisms, and residual architectures.

---

## 🎯 Key Contributions

- **Gradient-level analysis** of Leaky (LIF) vs Non-leaky (IF) output neurons for regression
- **Normalization strategies comparison**: BatchNorm, RMSNorm, and Fixed Scaling  
- **Architectural blocks**: Plain, Spiking ResNet, and SEW-ResNet for deep SNNs
- **Experimental validation** on rotary inverted pendulum control and event-based ego-motion estimation

---

## 📊 Dataset Examples

### Rotary Inverted Pendulum
Event camera captures the angular motion of an inverted pendulum from a top-view perspective.

<p align="center">
  <img src="docs/videos/pendulum_example.gif" alt="Pendulum Dataset" width="400"/>
  <br>
  <em>Sample sequence showing ON events (blue) and OFF events (red)</em>
</p>

**Event Statistics:**
- Resolution: 346×260 pixels
- Integration window: Δt = 30ms
- Average events per frame: ~1,500-3,000
- Dataset size: 50,000 timesteps

### Event-Based Ego-Motion (IMU)
DAVIS346 camera moving along roll and yaw axes, with ground truth from onboard IMU.

<p align="center">
  <img src="docs/videos/imu_example.gif" alt="IMU Dataset" width="400"/>
  <br>
  <em>Camera motion with higher temporal resolution (Δt = 10ms)</em>
</p>

**Event Statistics:**
- Resolution: 346×260 pixels
- Integration window: Δt = 10ms
- Average events per frame: ~2,000-5,000
- Dataset size: Multiple sequences totaling >100k timesteps

---

## 🏗️ Architecture

### Network Components

The SNN consists of:
1. **Convolutional Blocks** with Leaky Integrate-and-Fire (LIF) neurons
2. **Normalization Layers** (BatchNorm / RMSNorm / Fixed Scaling)
3. **Residual Connections** (Plain / Spiking ResNet / SEW-ResNet)
4. **Regression Head** with high-tau LIF neuron (τ=20) for stable output

<p align="center">
  <img src="docs/images/residual_blocks.png" alt="Residual Block Types" width="700"/>
</p>

### Neuron Dynamics

**Hidden Layers (LIF):**
```
H[t] = (1 - 1/τ)V[t-1] + (1/τ)I[t]
S[t] = Θ(H[t] - V_th)
V[t] = H[t] - S[t]·V_th
```

**Output Layer (High-tau LIF):**
```
H[t] = (1 - 1/τ_final)V[t-1] + (1/τ_final)I[t]
y[t] = H[t]  (membrane potential as continuous prediction)
```

---

## 📈 Results

### Experiment 1: Rotary Inverted Pendulum

**Angle Estimation Error (MAE in degrees):**

| Architecture | BatchNorm | RMSNorm | Fixed Scaling |
|--------------|-----------|---------|---------------|
| **SEW**      | 1.698°    | **1.651°** | 1.734°      |
| **Spiking**  | 1.907°    | **1.632°** | 2.153°      |
| **Plain**    | 2.547°    | 1.709°  | 2.341°        |

✨ **Best result:** Spiking ResNet + RMSNorm = **1.632° MAE**

<p align="center">
  <img src="docs/images/pendulum_predictions.png" alt="Pendulum Predictions" width="800"/>
  <br>
  <em>Model predictions vs ground truth on test sequence</em>
</p>

### Impact of Output Layer Time Constant

<p align="center">
  <img src="docs/images/tau_comparison.png" alt="Tau Comparison" width="700"/>
  <br>
  <em>Comparison of τ=2.0 (noisy) vs τ=20.0 (stable) predictions</em>
</p>

### Experiment 2: Event-Based Ego-Motion Estimation

**Roll Angle Estimation:**

<p align="center">
  <img src="docs/images/imu_predictions.png" alt="IMU Predictions" width="800"/>
  <br>
  <em>Visual odometry from event stream (10ms windows)</em>
</p>

---

## 🔬 Normalization Analysis

### What Are Normalization Layers Doing?

Our experiments reveal that **normalization primarily acts as gain control** rather than statistical re-centering:

<p align="center">
  <img src="docs/images/normalization_analysis.png" alt="Normalization Analysis" width="800"/>
</p>

**Key Findings:**
1. Learnable parameters (γ, β, α) remain close to initialization
2. Effective scaling factors converge to similar values across BN/RMS/MUL
3. **Purpose**: Amplify membrane potentials to sustain spiking activity
4. Without sufficient scaling → vanishing spikes → gradient collapse

### Gradient Dynamics: LIF vs IF Output

For regression outputs, we analyze gradient growth over time T:

**LIF Output (with leak β < 1):**
```
∂L/∂W = (1-β) Σ_t ∂L[t]/∂V[t] (Σ_s β^(t-s) X[s])
→ Exponential decay prevents gradient explosion
```

**IF Output (no leak):**
```
∂L/∂W = Σ_t ∂L[t]/∂V[t] (Σ_s X[s])
→ Quadratic growth O(T²) causes instability
```

**Solution:** Use high-tau LIF (τ=20) to approximate IF while maintaining training stability.

---

## 🚀 Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/snn-event-regression.git
cd snn-event-regression

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 1.12+
- SpikingJelly 0.0.0.0.14+
- NumPy, Matplotlib, OpenCV
- Tonic (for event processing)
- Weights & Biases (optional, for logging)

### Quick Start

**1. Train on Pendulum Dataset:**
```bash
python main.py
```

**2. Configure Experiment:**
Edit [main.py](main.py#L40):
```python
experiment = "IMU"           # Options: "pendulum", "IMU"
block_type = 'SEW'           # Options: 'SEW', 'plain', 'spiking'
norm_type = 'RMS'            # Options: 'BN', 'RMS', 'MUL', None
train_model = True           # Set False to load pretrained weights
```

**3. Monitor Training:**
```python
use_wandb = True             # Enable Weights & Biases logging
monitor_mode = "both"        # Options: "none", "spikes", "norm", "both"
```

---

## 📁 Project Structure

```
snn-event-regression/
├── main.py                      # Main training/testing script
├── data/                        # Dataset directory
│   ├── imu_events_large.aedat4
│   └── pendulum_events.aedat4
├── src/
│   ├── train.py                 # Training loop with TBPTT
│   ├── test.py                  # Testing and evaluation
│   ├── utils.py                 # Visualization and utilities
│   ├── Dataset/
│   │   ├── dataloaders.py       # Sequential data loading
│   │   ├── datasets.py          # Event stream datasets
│   │   └── read_file.py         # AEDAT4 file parsing
│   └── Network/
│       ├── SNN.py               # Main network architecture
│       ├── blocks.py            # Residual blocks (Plain/Spiking/SEW)
│       └── norm.py              # Normalization layers (BN/RMS/MUL)
├── models/                      # Saved checkpoints
├── notebooks/                   # Jupyter notebooks for analysis
└── docs/                        # Documentation and figures
```

---

## ⚙️ Training Details

### Truncated Backpropagation Through Time (TBPTT)

Event streams are extremely long, making standard BPTT impractical. We use TBPTT:

```python
K = 10  # Backprop every 10 timesteps
for t in range(0, T, K):
    forward_pass(t, t+K)
    backward_pass()      # Gradients truncated at K steps
    model.detach()       # Detach hidden states
```

### Hyperparameters

| Parameter | Pendulum | IMU | Description |
|-----------|----------|-----|-------------|
| τ (hidden) | 2.0 | 2.0 | Time constant for LIF neurons |
| τ (output) | 20.0 | 20.0 | High-tau for stable regression |
| K (TBPTT) | 10 | 10 | Truncation window |
| Transient | 200 | 0 | Warmup timesteps to skip |
| Batch Size | 4 | 4 | Sequences per batch |
| Seq Length | 2000 | 2000 | Timesteps per sequence |
| Optimizer | SGD | SGD | With momentum 0.9 |
| LR | 1e-2 | 1e-2 | With ReduceLROnPlateau |
| Epochs | 20 | 30 | Early stopping enabled |

### Surrogate Gradient

Non-differentiable spike function requires surrogate:
```python
surrogate_function = surrogate.ATan()  # arctan(αx)
```

---

## 📊 Visualization & Monitoring

### Spike Activity Monitoring

Track spiking activity through the network:

```python
monitor_mode = "spikes"  # Record spike counts per layer
```

<p align="center">
  <img src="docs/images/spike_activity.png" alt="Spike Activity" width="700"/>
  <br>
  <em>Layer-wise spike rates showing effect of normalization</em>
</p>

### Normalization Parameter Evolution

Monitor how γ, β, α evolve during training:

```python
monitor_mode = "norm"  # Track normalization parameters
```

<p align="center">
  <img src="docs/images/norm_params.png" alt="Normalization Parameters" width="700"/>
  <br>
  <em>Parameters remain nearly constant, confirming fixed-gain behavior</em>
</p>

---

## 🔧 Advanced Usage

### Custom Dataset

```python
from src.Dataset import read_pendulum_file

events, labels = read_pendulum_file(
    aedat_path="data/custom.aedat4",
    csv_path="data/custom_labels.csv",
    time_window=30000  # 30ms integration
)
```

### Custom Architecture

```python
from src.Network import SNN_Net

model = SNN_Net(
    tau=2.0,              # Hidden neuron time constant
    final_tau=20.0,       # Output neuron time constant
    layer_list=custom_layers,
    hidden=512,           # FC layer size
    norm_type="RMS",      # Normalization type
    init_scale=6.0        # Scaling factor for MUL
)
```

### Custom Block Configuration

```python
layer_list_custom = [
    {"channels": 16, "mid_channels": 16, "num_blocks": 1, "block_type": "sew",
     "up_kernel_size": 3, "stride_1": 2, "stride_2": 1, "k_pool": (2,2)},
    {"channels": 32, "mid_channels": 32, "num_blocks": 2, "block_type": "sew",
     "up_kernel_size": 1, "stride_1": 1, "stride_2": 1, "k_pool": (2,2)},
]
```

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{anonymous2026snn,
  title={Spiking Neural Networks for Event-Based Regression: Analysis of Neuron Dynamics and Normalization Strategies},
  author={Anonymous},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2026},
  note={Under review}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **SpikingJelly** framework for SNN implementation
- **Tonic** library for event-based data processing
- Event datasets collected using DAVIS346 camera
- Inspired by [StereoSpike](https://github.com/urancon/StereoSpike) normalization strategies

---

## 📧 Contact

For questions or collaboration:
- **Paper**: [Anonymous 4open.science link](https://anonymous.4open.science/r/snn-event-regression/)
- **Issues**: [GitHub Issues](https://github.com/your-username/snn-event-regression/issues)

---

<p align="center">
  <em>Event-driven vision meets neuromorphic computing 🧠⚡</em>
</p>

