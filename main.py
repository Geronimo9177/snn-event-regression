import random
from pathlib import Path
import os
import numpy as np
import torch
from spikingjelly.activation_based import surrogate

from src import (
	read_pendulum_file,
 	read_IMU_file,
	create_dataloaders,
	SNN_Net,
	train,
	test,
	plot_all,
	layer_list_sew,
	layer_list_plain,
    layer_list_spiking,
)


def main():
	# ============================================================================
	# Execution mode
	# ============================================================================
	mode = "train"  # Options: "train" or "test"

	# ============================================================================
	# Basic configuration
	# ============================================================================
	SEED = 42
	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# ============================================================================
	# Data parameters
	# ============================================================================
	experiment = "pendulum" # Options: "pendulum", "IMU"
	
	time_window = 30000
	START_FRAME = 300
	END_FRAME = -1

	# Splits and sequences
	test_ratio = 0.05
	val_ratio = 0.07
	
	SEQ_LENGTH = 2000
	BATCH_SIZE = 4
	
	# Dataloader optimization (reduce GPU waiting time)
	num_workers = 1
	prefetch_factor = 1
	pin_memory = False
	persistent_workers = True

	# ============================================================================
	# Model Hyperparameters
	# ============================================================================
	# Weights & Biases logging
	use_wandb = True

	# Select architecture type
	block_type = 'plain'  # Options: 'SEW', 'plain', 'spiking'
 
	monitor_mode = "none"  # Options: "none", "spikes", "norm", "both"
	
	K = 10           # TBPTT truncation window (backprop every K timesteps)
	transient = 200  # Initial timesteps to skip (warmup for recurrent states)

	num_epochs = 20  # Total training epochs
	hidden = 256     # Hidden layer size

	surrogate_function = surrogate.ATan()  # Gradient surrogate (arctan)
	Plif = False                          # Use standard LIF (not parametric)
	tau = 2.0                             # Time constant for hidden neurons
	final_tau = 20.0                      # Time constant for output neuron

	norm_type = 'BN'        # Normalization type: 'BN', 'RMS', 'MUL', or None
	learnable_norm = True   # Whether normalization parameters are learnable
	init_scale = 5.0        # Initial scale for Multiplication layers

	early_stop_patience = 10  # Stop if no improvement for this many epochs

	# ============================================================================
	# Event reading and preprocessing
	# ============================================================================
 
	if experiment.lower() == 'pendulum':
		FILE_PATH = "./data/pendulum_events.aedat4"
		CSV_PATH  = "./data/pendulum_encoder.csv"
  
		events_per_frame, labels = read_pendulum_file(
			FILE_PATH,
			CSV_PATH,
			time_window=time_window,
			START_FRAME=START_FRAME,
			END_FRAME=END_FRAME,
		)
	elif experiment.lower() == 'imu':
		FILE_PATH = "./data/imu_events.aedat4"
		events_per_frame, labels = read_IMU_file(
			FILE_PATH,
			time_window=time_window,
			START_FRAME=START_FRAME,
			END_FRAME=END_FRAME,
		)
	else:
		raise ValueError("Unsupported experiment type. Choose 'pendulum' or 'IMU'.")
	# ============================================================================
	# Dataloaders
	# ============================================================================
	trainloader, valloader, testloader = create_dataloaders(
		events_per_frame,
		labels,
		test_ratio=test_ratio,
		val_ratio=val_ratio,
		SEQ_LENGTH=SEQ_LENGTH,
		BATCH_SIZE=BATCH_SIZE,
		num_workers=num_workers,
		prefetch_factor=prefetch_factor,
		pin_memory=pin_memory,
		persistent_workers=persistent_workers,
	)

	# ============================================================================
	# Model
	# ============================================================================
	if block_type.lower() == 'sew':
		layer_list = layer_list_sew
	elif block_type.lower() == 'plain':
		layer_list = layer_list_plain
	elif block_type.lower() == 'spiking':
		layer_list = layer_list_spiking

	model = SNN_Net(
		tau=tau, 
		final_tau=final_tau,
		layer_list=layer_list, 
		hidden=hidden, 
		surrogate_function=surrogate_function,
		connect_f="ADD",  # Connection function for SEW blocks
		Plif=Plif, 
		norm_type=norm_type, 
		learnable_norm=learnable_norm,
		init_scale = init_scale
	).to(device)

	CONFIG = {
		# Experiment
		"experiment": experiment,
		# Model architecture
		"block_type": block_type,
		"model_name": model.__class__.__name__,
		"hidden": hidden,
		"tau": tau,
		"final_tau": final_tau,
		"surrogate_function": surrogate_function.__class__.__name__,
		"Plif": Plif,
		"norm_type": norm_type,
		"learnable_norm": learnable_norm,
		"init_scale": init_scale,

		# TBPTT parameters
		"K": K,                     
		"transient": transient,        
		"batch_size": BATCH_SIZE,        
		"sequence_length": SEQ_LENGTH,  

		# Optimizer configuration
		"optimizer": "SGD",
		"learning_rate": 1e-2,
		"momentum": 0.9,
		"weight_decay": 0.0,

		# Learning rate scheduler
		"scheduler": "ReduceLROnPlateau",
		"scheduler_factor": 0.5,         
		"scheduler_patience": 1,         
		"min_lr": 1e-6,                  

		# Training loop
		"num_epochs": num_epochs,
		"device": str(device),

		# Early stopping
		"early_stop_patience": early_stop_patience,
	}
 
	output_dir = Path(f"./models/model_{CONFIG['block_type']}_{CONFIG['norm_type']}/checkpoints_{CONFIG['experiment']}")

	# ============================================================================
	# Training or Testing
	# ============================================================================
	if mode.lower() == "train":
		print("\n" + "="*70)
		print("TRAINING MODE")
		print("="*70)
		output_dir.mkdir(parents=True, exist_ok=True)
		train(model, trainloader, valloader, CONFIG, output_dir, loss_fn=torch.nn.MSELoss(), use_wandb=use_wandb)

		print(f"\nModel trained successfully.")

	elif mode.lower() == "test":
		print("\n" + "="*70)
		print("TEST MODE")
		print("="*70)
		model_checkpoint_path = output_dir / "best_model_weights.pth"
		# Load pre-trained weights
		if not model_checkpoint_path.exists():
			raise FileNotFoundError(f"Model weights not found: {model_checkpoint_path}")
		
		print(f"Loading model weights from: {model_checkpoint_path}")
		model.load_state_dict(torch.load(str(model_checkpoint_path), map_location=device, weights_only=True))
		print("Model weights loaded successfully.\n")

	else:
		raise ValueError(f"Invalid mode: {mode}. Choose 'train' or 'test'.")

	# ============================================================================
	# Test
	# ============================================================================
	results = test(model, testloader, CONFIG, monitor_mode, loss_fn=torch.nn.MSELoss())
 
 	# ============================================================================
	# visualization
	# ============================================================================

	plot_all(results, window_start=0, window_end=-1)

	if use_wandb:
		import wandb
		wandb.finish()


if __name__ == "__main__":
	main()

