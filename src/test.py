import torch
import numpy as np
from tqdm import tqdm
from spikingjelly.activation_based import functional
import matplotlib.pyplot as plt


# ============================================================================
# Monitoring Mode Configuration
# ============================================================================
def enable_monitoring(model, mode):
    """Enable monitoring hooks based on selected mode."""
    if mode in ["spikes", "both"]:
        model.enable_spike_recording()
    if mode in ["norm", "both"]:
        model.enable_norm_monitoring()


def disable_monitoring(model, mode):
    """Disable monitoring hooks and clean up."""
    if mode in ["spikes", "both"]:
        model.disable_spike_recording()
    if mode in ["norm", "both"]:
        model.disable_norm_monitoring()


def test(model, testloader, CONFIG, monitor_mode="both", loss_fn=None):
    """Evaluate model on test set with comprehensive monitoring capabilities.
    
    Args:
        model: The trained neural network model
        testloader: DataLoader for test data
        CONFIG: Configuration dictionary containing device and other settings
        loss_fn: Loss function to use (default: MSELoss). Can be MSELoss or L1Loss
        monitor_mode: "none", "spikes", "norm", or "both" for monitoring during evaluation
    """
    # Evaluation on test set
    device = torch.device(CONFIG["device"])
    
    # Set default loss function if not provided
    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()
    
    # Determine loss type for reporting
    loss_type = "MSE" if isinstance(loss_fn, torch.nn.MSELoss) else "L1"
    
    # Initialize monitoring structures based on mode
    spike_activity_over_time = {} if monitor_mode in ["spikes", "both"] else None
    norm_activity_over_time = {} if monitor_mode in ["norm", "both"] else None

    with torch.no_grad():
        model.eval()
        
        enable_monitoring(model, monitor_mode)


        print(f"\nEvaluating model on test set (Monitor mode: {monitor_mode})...")
        
        test_loss_total = 0.0
        test_rel_err_total = 0.0
        iter_count = 0
        
        # Store all predictions for visualization
        all_predictions = []
        all_targets = []
        
        # Progress bar for test batches
        pbar_test = tqdm(testloader, desc="  Test batches", leave=True)
        
        for data, targets in pbar_test:
            iter_count += 1
            data = data.to(device)           # [T, B, C, H, W]
            targets = targets.to(device)     # [T, B]
            
            # Normalize targets
            targets = targets / np.pi
            
            num_steps = data.size(0)  # T
            
            # Reset hidden states at start of each sequence batch
            functional.reset_net(model)
            
            test_mem_list = []
            
            # Process through entire test sequence
            for step in range(num_steps):
                
                # Forward pass
                mem_out = model(data[step])  # [B, 1]
                test_mem_list.append(mem_out)
                
                # Initialize tracking lists on the first timestep
                if step == 0:
                    if monitor_mode in ["spikes", "both"]:
                        for k in model.spike_record.keys():
                            spike_activity_over_time[k] = []
                    
                    if monitor_mode in ["norm", "both"]:
                        for k in model.norm_stats.keys():
                            norm_activity_over_time[k] = {
                                'input_mean': [],
                                'input_std': [],
                                'output_mean': [],
                                'output_std': [],
                                'input_range': [],
                                'output_range': []
                            }
                
                # Record spike activity if enabled
                if monitor_mode in ["spikes", "both"]:
                    # For each LIF layer, compute spike activity for this timestep
                    for k, v in model.spike_record.items():
                        # v: [B, C, H, W] or [B, N]
                        # Average spikes per neuron in this layer at this timestep
                        activity = v.detach().cpu().mean().item()
                        spike_activity_over_time[k].append(activity)
                
                # Record normalization statistics if enabled
                if monitor_mode in ["norm", "both"]:
                    # For each BatchNorm layer, extract statistics at this timestep
                    for layer_name, stats in model.norm_stats.items():
                        # Average across all channels for this timestep
                        norm_activity_over_time[layer_name]['input_mean'].append(
                            np.mean(stats['input_mean_per_channel'])
                        )
                        norm_activity_over_time[layer_name]['input_std'].append(
                            np.mean(stats['input_std_per_channel'])
                        )
                        norm_activity_over_time[layer_name]['output_mean'].append(
                            np.mean(stats['output_mean_per_channel'])
                        )
                        norm_activity_over_time[layer_name]['output_std'].append(
                            np.mean(stats['output_std_per_channel'])
                        )
                        
                        # Ranges (max - min) averaged
                        input_range = np.mean(stats['input_max_per_channel'] - stats['input_min_per_channel'])
                        output_range = np.mean(stats['output_max_per_channel'] - stats['output_min_per_channel'])
                        
                        norm_activity_over_time[layer_name]['input_range'].append(input_range)
                        norm_activity_over_time[layer_name]['output_range'].append(output_range)
            
            # Stack all predictions for this batch
            batch_predictions = torch.stack(test_mem_list, dim=0)  # [T, B, 1]
            batch_predictions = batch_predictions.squeeze(-1)  # [T, B]
            
            # Calculate metrics for this batch
            batch_loss = loss_fn(batch_predictions, targets)
            batch_rel_err = torch.linalg.norm(batch_predictions - targets) / torch.linalg.norm(targets)
            
            test_loss_total += batch_loss.item()
            test_rel_err_total += batch_rel_err.item()
            
            # Update progress bar
            pbar_test.set_postfix({'loss': f'{batch_loss.item():.6f}'})
            
            # Store for visualization (flatten batch dimension)
            all_predictions.append(batch_predictions.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
        
        pbar_test.close()
        
        # Average metrics
        avg_test_loss = test_loss_total / iter_count
        avg_test_rel_err = test_rel_err_total / iter_count
        
        # Concatenate all predictions (flatten across batches and batch dimension)
        test_mem_continuous = np.concatenate([p.reshape(-1) for p in all_predictions])
        test_target_continuous = np.concatenate([t.reshape(-1) for t in all_targets])
    
    # Disable monitoring after evaluation
    disable_monitoring(model, monitor_mode)

    # ========================================================================
    # Print evaluation results
    # ========================================================================
    print(f"\n{'='*50}")
    print(f"{'Test ' + loss_type + ' Loss:':<{20}}{avg_test_loss:1.2e}")
    print(f"{'Test Rel. Error:':<{20}}{avg_test_rel_err:1.2e}")
    print(f"{'Total iterations:':<{20}}{iter_count}")
    print(f"{'Total timesteps:':<{20}}{len(test_mem_continuous)}")
    print(f"{'Monitoring mode:':<{20}}{monitor_mode}")
    print(f"{'='*50}")

    # ========================================================================
    # Extract spike and normalization statistics
    # ========================================================================
    
    # Process spike activity if monitoring was enabled
    if monitor_mode in ["spikes", "both"]:
        # Convert to arrays to calculate averages across the network
        all_layers = list(spike_activity_over_time.keys())
        T = len(next(iter(spike_activity_over_time.values())))  # number of timesteps

        # Average spike activity per timestep across the network
        spike_activity_total = np.zeros(T)
        for k in all_layers:
            spike_activity_total += np.array(spike_activity_over_time[k])
        spike_activity_total /= len(all_layers)
        
        # data: [T, B, C, H, W]
        num_timesteps = data.size(0)
        num_events_per_timestep = []

        for t in range(num_timesteps):
            # Count non-zero values at this timestep (sum over batch, channels, height, width)
            events = (data[t] != 0).sum().item()
            num_events_per_timestep.append(events)

        num_events_per_timestep = np.array(num_events_per_timestep)
            
    else:
        spike_activity_total = None
        num_events_per_timestep = None
    
    # Return results dictionary for further analysis
    results = {
        'avg_loss': avg_test_loss,
        'avg_rel_err': avg_test_rel_err,
        'test_output': test_mem_continuous,
        'test_target': test_target_continuous,
        'spike_activity': spike_activity_total,
        'num_events': num_events_per_timestep,
        'norm_stats': norm_activity_over_time,
        'spike_stats': spike_activity_over_time,
    }
    
    return results


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_prediction(results, window_start=0, window_end=-1):
    """
    Plot only model predictions vs targets.
    
    Args:
        results: Dictionary returned from test() function
        window_start: Start index for plotting window
        window_end: End index for plotting window (default: min(2000, total length))
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # Subplot 1: Model output vs target
    ax1.plot(results['test_output'][window_start:window_end] * 180, 
             label="Model Output", alpha=0.8, linewidth=1.5)
    ax1.plot(results['test_target'][window_start:window_end] * 180, 
             label="Target", alpha=0.8, linewidth=1.5)
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Angle (degrees)")
    ax1.set_title("Model Output vs Target (Continuous Evaluation)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Absolute error
    test_error = np.abs(results['test_output'][window_start:window_end] - 
                        results['test_target'][window_start:window_end]) * 180
    ax2.plot(test_error, color='orange', linewidth=1)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Absolute Error (degrees)")
    ax2.set_title("Absolute Error over Time")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    # Print error statistics
    print(f"\nError statistics (Full sequence):")
    print(f"  Mean error: {np.mean(np.abs(results['test_output'] - results['test_target']) * 180):.2f}°")
    print(f"  Std error: {np.std(np.abs(results['test_output'] - results['test_target']) * 180):.2f}°")
    print(f"  Max error: {np.max(np.abs(results['test_output'] - results['test_target']) * 180):.2f}°")

    print(f"\nError statistics (Window [{window_start}:{window_end}]):")
    print(f"  Mean error: {np.mean(np.abs(results['test_output'][window_start:window_end] - results['test_target'][window_start:window_end]) * 180):.2f}°")
    print(f"  Std error: {np.std(np.abs(results['test_output'][window_start:window_end] - results['test_target'][window_start:window_end]) * 180):.2f}°")
    print(f"  Max error: {np.max(np.abs(results['test_output'][window_start:window_end] - results['test_target'][window_start:window_end]) * 180):.2f}°")


def plot_spike_activity(results, window_start=0, window_end=-1):
    """Plot spike activity and input events."""
    if results['spike_activity'] is None:
        print("No spike activity data available. Run test() with monitor_mode='spikes' or 'both'")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # Subplot 1: Average spike activity
    ax1.plot(results['spike_activity'][window_start:window_end], 
             color='green', linewidth=1.5)
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Average Spike Activity")
    ax1.set_title("Average Spike Activity Across the Network Over Time")
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Number of input events
    ax2.plot(results['num_events'][window_start:window_end], 
             color='purple', linewidth=1.5)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Number of Events")
    ax2.set_title("Number of Input Events per Timestep")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    # Print spike statistics
    print(f"\nSpike activity statistics:")
    print(f"  Mean activity: {np.mean(results['spike_activity']):.6f}")
    print(f"  Std activity: {np.std(results['spike_activity']):.6f}")
    print(f"  Max activity: {np.max(results['spike_activity']):.6f}")
    print(f"  Min activity: {np.min(results['spike_activity']):.6f}")
    
    print(f"\nInput event statistics:")
    print(f"  Mean events per timestep: {np.mean(results['num_events']):.2f}")
    print(f"  Max events per timestep: {np.max(results['num_events']):.2f}")
    print(f"  Min events per timestep: {np.min(results['num_events']):.2f}")


def plot_normalization_stats(results, window_start=0, window_end=-1):
    """Plot normalization layer statistics."""
    if results['norm_stats'] is None:
        print("No normalization statistics data available. Run test() with monitor_mode='norm' or 'both'")
        return
    
    # Determine number of layers
    num_layers = len(results['norm_stats'])
    
    fig, axes = plt.subplots(num_layers, 1, figsize=(15, 5 * num_layers))
    
    # Handle single layer case
    if num_layers == 1:
        axes = [axes]
    
    # Plot statistics for each layer
    for idx, (layer_name, stats) in enumerate(results['norm_stats'].items()):
        ax = axes[idx]
        
        ax.plot(stats['input_mean'][window_start:window_end], 
                label="Input Mean", linewidth=1.5, alpha=0.8)
        ax.plot(stats['output_mean'][window_start:window_end], 
                label="Output Mean", linewidth=1.5, alpha=0.8)
        ax.plot(stats['input_std'][window_start:window_end], 
                label="Input Std", linewidth=1.5, alpha=0.8)
        ax.plot(stats['output_std'][window_start:window_end], 
                label="Output Std", linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel("Frame")
        ax.set_ylabel("Statistics Value")
        ax.set_title(f"Normalization Statistics - {layer_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_all(results, window_start=0, window_end=-1):
    """Plot all available data (predictions, spikes, normalization)."""
    plot_prediction(results, window_start, window_end)
    
    if results['spike_activity'] is not None:
        plot_spike_activity(results, window_start, window_end)
    
    if results['norm_stats'] is not None:
        plot_normalization_stats(results, window_start, window_end)

