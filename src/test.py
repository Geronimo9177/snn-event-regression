import torch
import numpy as np
from tqdm import tqdm
from spikingjelly.activation_based import functional
import matplotlib.pyplot as plt

from .Network.norm import RMSNorm2d, MultiplyBy


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


def _collect_norm_param_stats(model, norm_activity_over_time):
    """Collect mean/std for weights and bias, plus scale for MultiplyBy, for monitored norm layers."""
    if norm_activity_over_time is None:
        return None

    stats = {}

    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.BatchNorm2d, RMSNorm2d, MultiplyBy)):
            continue

        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        scale = getattr(module, "scale_value", None)

        stats[name] = {
            'weight_mean': float(weight.detach().mean().item()) if weight is not None else None,
            'weight_std': float(weight.detach().std().item()) if weight is not None else None,
            'bias_mean': float(bias.detach().mean().item()) if bias is not None else None,
            'bias_std': float(bias.detach().std().item()) if bias is not None else None,
            'scale': float(scale.detach().item()) if isinstance(scale, torch.Tensor) else (float(scale) if scale is not None else None),
        }

    return stats


def test(model, testloader, CONFIG, monitor_mode="both", loss_fn=None):
    """Evaluate model on test set with comprehensive monitoring capabilities.
    
    Args:
        model: The trained neural network model
        testloader: DataLoader for test data
        CONFIG: Configuration dictionary containing device and other settings
        loss_fn: Loss function to use (default: MSELoss). Can be MSELoss or L1Loss
        monitor_mode: "none", "spikes", "norm", or "both" for monitoring during evaluation
    """
    true_value_initialization = CONFIG["true_value_initialization"]
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
            
            if true_value_initialization:
                model.lif_out.v = targets[0].unsqueeze(-1)  # Initialize output neuron state
            
            test_mem_list = []
            
            # Process through entire test sequence
            start_step = 1 if true_value_initialization else 0
            for step in range(start_step, num_steps):
                
                # Forward pass
                mem_out = model(data[step])  # [B, 1]
                test_mem_list.append(mem_out)
                
                # Initialize tracking lists on the first timestep
                if step == start_step:
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

            # Align targets when using true value initialization
            targets_aligned = targets[start_step:]  # [T, B]

            # Calculate metrics for this batch over full aligned sequence
            batch_loss = loss_fn(batch_predictions, targets_aligned)
            batch_rel_err = torch.linalg.norm(batch_predictions - targets_aligned) / torch.linalg.norm(targets_aligned)
            
            test_loss_total += batch_loss.item()
            test_rel_err_total += batch_rel_err.item()
            
            # Update progress bar
            pbar_test.set_postfix({'loss': f'{batch_loss.item():.6f}'})
            
            # Store for visualization (flatten batch dimension) over full aligned sequence
            all_predictions.append(batch_predictions.detach().cpu().numpy())
            all_targets.append(targets_aligned.detach().cpu().numpy())
        
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

        # Align event counting with processed timesteps
        start_step = 1 if true_value_initialization else 0
        for t in range(start_step, num_timesteps):
            # Count non-zero values at this timestep (sum over batch, channels, height, width)
            events = (data[t] != 0).sum().item()
            num_events_per_timestep.append(events)

        num_events_per_timestep = np.array(num_events_per_timestep)
            
    else:
        spike_activity_total = None
        num_events_per_timestep = None
    
    # Collect static BatchNorm parameter statistics for convenience
    if monitor_mode in ["norm", "both"]:
        norm_param_stats = _collect_norm_param_stats(model, norm_activity_over_time)
    else:
        norm_param_stats = None

    # Return results dictionary for further analysis
    results = {
        'avg_loss': avg_test_loss,
        'avg_rel_err': avg_test_rel_err,
        'test_output': test_mem_continuous,
        'test_target': test_target_continuous,
        'spike_activity': spike_activity_total,
        'num_events': num_events_per_timestep,
        'norm_stats': norm_activity_over_time,
        'norm_params': norm_param_stats,
        'spike_stats': spike_activity_over_time,
    }
    
    return results

