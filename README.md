# Spiking Neural Network for Event-Based Regression

This repository contains the codebase used in our upcoming paper (title TBD). It implements a **Spiking Neural Network (SNN)** for continuous regression from event-camera data.

## Overview
- Multiple architectures: Plain, Spiking ResNet, and SEW (Spiking Element-Wise) blocks
- Flexible normalization: Batch Normalization, RMS Normalization, or learnable scaling
- Training: Truncated Backpropagation Through Time (TBPTT) for temporal learning

## Quick start
- Place data in data/ (e.g., pendulum_events.aedat4 and pendulum_encoder.csv)
- Train: python main.py (defaults to mode="train")
- Test: set mode="test" in main.py and ensure best_model_weights.pth exists under the appropriate checkpoints folder

## Citation
If you use this code in your research, please cite our paper:
```
#TODO Citation
```

