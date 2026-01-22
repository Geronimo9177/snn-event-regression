import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def visualize_sequence_from_trainloader(trainloader, n_sequences=5, playback_fps=10, scale=1):
    """
    Visualize temporal sequences from the trainloader (output of SequentialRotatingBarDataset).
    
    Args:
        trainloader: DataLoader with frames shaped [T, B, C, H, W] and labels shaped [T, B].
        n_sequences: Number of sequences (batches) to visualize.
        playback_fps: Playback speed for each timestep in the sequence.
        scale: Scale factor for visualization in pixels.
    """
    
    for seq_idx, (frames_batch, labels_batch) in enumerate(trainloader):
        if seq_idx >= n_sequences:
            break
            
        # frames_batch: [T, B, C, H, W]
        # labels_batch: [T, B]
        T, B, C, H, W = frames_batch.shape
        
        print(f"\n=== Sequence {seq_idx+1}/{n_sequences} ===")
        print(f"Batch shape: {frames_batch.shape}, Labels shape: {labels_batch.shape}")
        
        # Visualize only the first element of the batch
        batch_item = 0
        
        for t in range(T):
            # Extract frame at time t for the first batch item
            # frame: [C, H, W] where C=2 (ON/OFF polarities)
            frame = frames_batch[t, batch_item].cpu().numpy()  # [2, H, W]
            angle = labels_batch[t, batch_item].item()
            
            # Create RGB visualization
            # Channel 0 = ON events (positive), Channel 1 = OFF events (negative)
            events_img = np.ones((H, W, 3), dtype=np.uint8) * 255
            
            # ON events in dark blue
            on_events = frame[0] > 0
            events_img[on_events] = [0, 0, 200]
            
            # OFF events in dark red
            off_events = frame[1] > 0
            events_img[off_events] = [200, 0, 0]
            
            # Scale for easier viewing
            events_resized = cv.resize(events_img, (W*scale, H*scale), 
                                      interpolation=cv.INTER_NEAREST)
            
            # Add overlay text
            info_text = [
                f"Seq: {seq_idx+1}/{n_sequences}  Time: {t+1}/{T}",
                f"Target angle: {np.rad2deg(angle):.2f} deg",
                f"Batch item: {batch_item+1}/{B}"
            ]
            
            y_offset = 30
            for text in info_text:
                cv.putText(events_resized, text,
                          (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 
                          0.6, (0, 0, 0), 2)
                y_offset += 25
            
            # Show frame
            cv.imshow("Trainloader Sequence Visualization", events_resized)
            
            key = cv.waitKey(int(1000 / playback_fps))
            if key == 27:  # ESC to exit
                cv.destroyAllWindows()
                return
            elif key == ord('n'):  # 'n' to move to the next sequence
                break
    
    cv.destroyAllWindows()
    print("\nVisualization completed!")


# ============================================================================
# Result Visualization Functions
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
    """Plot normalization summary using amplification factors."""
    if results['norm_stats'] is None:
        print("No normalization statistics data available. Run test() with monitor_mode='norm' or 'both'")
        return

    layer_names, mean_amplifications = compute_mean_amplification_per_layer(
        results['norm_stats'], results.get('norm_params'), window_start, window_end
    )

    if len(layer_names) == 0:
        print("No normalization amplification data available to plot.")
        return

    fig = plot_network_amplification(layer_names, mean_amplifications)
    plt.show()

    # Print normalization parameter statistics if available
    norm_params = results.get('norm_params')
    if norm_params:
        print("\nNormalization parameter statistics (mean +/- std):")
        names_to_print = list(dict.fromkeys(layer_names + [n for n in norm_params.keys() if n not in layer_names]))

        agg_weight_means = []
        agg_weight_stds = []
        agg_bias_means = []
        agg_bias_stds = []
        agg_scales = []

        for layer_name in names_to_print:
            layer_stats = norm_params.get(layer_name, {})
            w_mean = layer_stats.get('weight_mean')
            w_std = layer_stats.get('weight_std')
            b_mean = layer_stats.get('bias_mean')
            b_std = layer_stats.get('bias_std')
            scale = layer_stats.get('scale')

            parts = []
            if w_mean is not None and w_std is not None:
                parts.append(f"W {w_mean:.4f} +/- {w_std:.4f}")
                agg_weight_means.append(w_mean)
                agg_weight_stds.append(w_std)
            if b_mean is not None and b_std is not None:
                parts.append(f"b {b_mean:.4f} +/- {b_std:.4f}")
                agg_bias_means.append(b_mean)
                agg_bias_stds.append(b_std)
            if scale is not None:
                parts.append(f"scale {scale:.4f}")
                agg_scales.append(scale)

            if parts:
                print(f"  {layer_name}: " + ", ".join(parts))

        # Network-level averages across layers (ignore missing entries)
        def _safe_mean(values):
            return float(np.mean(values)) if len(values) > 0 else None

        net_w_mean = _safe_mean(agg_weight_means)
        net_w_std = _safe_mean(agg_weight_stds)
        net_b_mean = _safe_mean(agg_bias_means)
        net_b_std = _safe_mean(agg_bias_stds)
        net_scale = _safe_mean(agg_scales)

        summary_parts = []
        if net_w_mean is not None and net_w_std is not None:
            summary_parts.append(f"W mean {net_w_mean:.4f}, W std {net_w_std:.4f}")
        if net_b_mean is not None and net_b_std is not None:
            summary_parts.append(f"b mean {net_b_mean:.4f}, b std {net_b_std:.4f}")
        if net_scale is not None:
            summary_parts.append(f"scale mean {net_scale:.4f}")

        if summary_parts:
            print("  Network averages: " + " | ".join(summary_parts))


def compute_mean_amplification_per_layer(norm_activity_over_time, norm_params=None, window_start=0, window_end=-1):
    """Compute average amplification factor (std_out / std_in) over time for each layer.
    
    For BatchNorm/RMSNorm: computes std_out / std_in from activity stats.
    For MultiplyBy (learnable): uses scale_value directly as amplification factor.
    """
    layer_names = []
    mean_amplifications = []

    # Process layers with temporal activity stats (BatchNorm, RMSNorm)
    if norm_activity_over_time:
        for layer_name, activity in norm_activity_over_time.items():
            input_std = np.array(activity['input_std'][window_start:window_end])
            output_std = np.array(activity['output_std'][window_start:window_end])

            if input_std.size == 0 or output_std.size == 0:
                continue

            std_scaling = output_std / (input_std + 1e-8)
            mean_scaling = std_scaling.mean()

            layer_names.append(layer_name)
            mean_amplifications.append(mean_scaling)
    
    # Add MultiplyBy layers using their scale_value if learnable
    if norm_params:
        for layer_name, params in norm_params.items():
            scale = params.get('scale')
            # Only add if it has a scale and wasn't already added from activity stats
            if scale is not None and layer_name not in layer_names:
                layer_names.append(layer_name)
                mean_amplifications.append(scale)

    return layer_names, np.array(mean_amplifications)


def plot_network_amplification(layer_names, mean_amplifications):
    """Bar plot showing average amplification factor per layer."""
    layers = np.arange(len(layer_names))

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(layers, mean_amplifications, alpha=0.8, color='purple')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='No scaling (1x)')

    avg_network = mean_amplifications.mean()
    ax.axhline(avg_network, color='black', linestyle=':', linewidth=2,
               label=f'Network avg = {avg_network:.1f}x')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Amplification (Output / Input)')
    ax.set_title(f'Mean Amplification Factor per Layer (avg={avg_network:.1f}x)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_all(results, window_start=0, window_end=-1):
    """Plot all available data (predictions, spikes, normalization)."""
    plot_prediction(results, window_start, window_end)
    
    if results['spike_activity'] is not None:
        plot_spike_activity(results, window_start, window_end)
    
    if results['norm_stats'] is not None:
        plot_normalization_stats(results, window_start, window_end)