"""
Generate all media files for README documentation.

This script automates the creation of figures and videos for the project README.
Run after training models to generate comprehensive documentation.

Usage:
    python generate_media.py --experiment pendulum --model_path models/...
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import torch
from tqdm import tqdm

# Set consistent plotting style
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'prediction': '#2E86AB',  # Blue
    'ground_truth': '#F77F00',  # Orange
    'noise': '#EE6C4D',  # Red
}

def setup_directories():
    """Create docs directories if they don't exist."""
    Path("docs/images").mkdir(parents=True, exist_ok=True)
    Path("docs/videos").mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")


def generate_dataset_videos(experiment_type='pendulum', n_frames=150, fps=30):
    """
    Generate GIF visualization of dataset examples.
    
    Args:
        experiment_type: 'pendulum' or 'IMU'
        n_frames: Number of frames to include
        fps: Playback speed
    """
    print(f"\n📹 Generating {experiment_type} dataset video...")
    
    from src import read_pendulum_file, read_IMU_file, create_dataloaders
    
    # Load data
    if experiment_type.lower() == 'pendulum':
        events, labels = read_pendulum_file(
            "./data/pendulum_events.aedat4",
            "./data/pendulum_encoder.csv",
            time_window=30000,
            START_FRAME=300,
            END_FRAME=-1
        )
        output_name = "pendulum_example.gif"
    else:
        events, labels = read_IMU_file(
            "./data/imu_events_large.aedat4",
            time_window=10000,
            START_FRAME=0,
            END_FRAME=-2500
        )
        output_name = "imu_example.gif"
    
    # Create dataloader
    trainloader, _, _ = create_dataloaders(
        events, labels,
        test_ratio=0.05,
        val_ratio=0.07,
        SEQ_LENGTH=2000,
        BATCH_SIZE=1,
        num_workers=0
    )
    
    # Extract first sequence
    frames_batch, labels_batch = next(iter(trainloader))
    T, B, C, H, W = frames_batch.shape
    
    frames_to_save = []
    
    for t in range(min(n_frames, T)):
        frame = frames_batch[t, 0].cpu().numpy()  # [2, H, W]
        angle = labels_batch[t, 0].item()
        
        # Create RGB visualization
        img = np.ones((H, W, 3), dtype=np.uint8) * 255
        
        # ON events (blue), OFF events (red)
        img[frame[0] > 0] = [0, 0, 200]
        img[frame[1] > 0] = [200, 0, 0]
        
        # Resize for better visibility
        img_resized = cv2.resize(img, (W*2, H*2), interpolation=cv2.INTER_NEAREST)
        
        # Add text overlay
        cv2.putText(img_resized, f"Frame: {t}/{T}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img_resized, f"Angle: {np.rad2deg(angle):.1f}°", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        frames_to_save.append(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    
    # Save as GIF using imageio
    import imageio
    imageio.mimsave(f"docs/videos/{output_name}", frames_to_save, fps=fps)
    print(f"✓ Saved: docs/videos/{output_name}")


def plot_predictions(results, experiment_type='pendulum', window=(0, 1000)):
    """
    Generate prediction vs ground truth plot.
    
    Args:
        results: Dictionary with 'predictions' and 'targets' arrays
        experiment_type: 'pendulum' or 'IMU'
        window: (start, end) timesteps to plot
    """
    print(f"\n📊 Generating {experiment_type} predictions plot...")
    
    predictions = results['predictions']
    targets = results['targets']
    
    start, end = window
    if end == -1:
        end = len(predictions)
    
    pred_slice = predictions[start:end]
    target_slice = targets[start:end]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    time = np.arange(len(pred_slice))
    ax.plot(time, pred_slice, label='Prediction', 
            color=COLORS['prediction'], linewidth=1.5, alpha=0.9)
    ax.plot(time, target_slice, label='Ground Truth', 
            color=COLORS['ground_truth'], linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Angle (degrees)', fontsize=12)
    ax.set_title(f'{experiment_type.capitalize()} Angle Estimation', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'docs/images/{experiment_type}_predictions.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: docs/images/{experiment_type}_predictions.png")


def plot_tau_comparison(results_low_tau, results_high_tau, window=(0, 500)):
    """
    Compare output stability for different tau values.
    
    Args:
        results_low_tau: Results from model with tau=2.0
        results_high_tau: Results from model with tau=20.0
        window: (start, end) timesteps to plot
    """
    print("\n📊 Generating tau comparison plot...")
    
    start, end = window
    
    pred_low = results_low_tau['predictions'][start:end]
    pred_high = results_high_tau['predictions'][start:end]
    target = results_high_tau['targets'][start:end]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    time = np.arange(len(pred_low))
    ax.plot(time, pred_low, label='τ=2.0 (noisy)', 
            color=COLORS['noise'], linewidth=1.2, alpha=0.7)
    ax.plot(time, pred_high, label='τ=20.0 (stable)', 
            color=COLORS['prediction'], linewidth=1.5, alpha=0.9)
    ax.plot(time, target, label='Ground Truth', 
            color=COLORS['ground_truth'], linewidth=1.5, alpha=0.6, linestyle='--')
    
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Angle (degrees)', fontsize=12)
    ax.set_title('Impact of Output Layer Time Constant', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/images/tau_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: docs/images/tau_comparison.png")


def plot_spike_activity(spike_stats):
    """
    Visualize layer-wise spike activity.
    
    Args:
        spike_stats: Dictionary with layer names and spike counts
    """
    print("\n📊 Generating spike activity plot...")
    
    layer_names = list(spike_stats.keys())
    spike_means = [spike_stats[name]['mean'] for name in layer_names]
    spike_stds = [spike_stats[name]['std'] for name in layer_names]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(layer_names))
    ax.bar(x_pos, spike_means, yerr=spike_stds, 
           color=COLORS['prediction'], alpha=0.7, capsize=5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Average Spikes per Neuron', fontsize=12)
    ax.set_title('Layer-wise Spike Activity', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/images/spike_activity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: docs/images/spike_activity.png")


def plot_normalization_params(norm_stats):
    """
    Track normalization parameter evolution.
    
    Args:
        norm_stats: Dictionary with layer names and parameter values
    """
    print("\n📊 Generating normalization parameters plot...")
    
    layer_names = list(norm_stats.keys())
    param_values = [norm_stats[name]['scale'] for name in layer_names]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(layer_names))
    ax.bar(x_pos, param_values, color=COLORS['ground_truth'], alpha=0.7)
    
    # Add horizontal line at initialization value
    ax.axhline(y=5.0, color='red', linestyle='--', 
               label='Initialization (α=5.0)', linewidth=2)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Scale Parameter Value', fontsize=12)
    ax.set_title('Normalization Parameters (Post-Training)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/images/norm_params.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: docs/images/norm_params.png")


def create_architecture_placeholder():
    """Create placeholder for architecture overview diagram."""
    print("\n🖼️ Creating architecture placeholder...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.text(0.5, 0.5, 'Architecture Overview\n(Create diagram manually)',
            ha='center', va='center', fontsize=20, color='gray')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('docs/images/architecture_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: docs/images/architecture_overview.png (placeholder)")


def create_blocks_placeholder():
    """Create placeholder for residual blocks diagram."""
    print("\n🖼️ Creating residual blocks placeholder...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'Residual Block Types\n(Convert from LaTeX figure)',
            ha='center', va='center', fontsize=20, color='gray')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('docs/images/residual_blocks.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: docs/images/residual_blocks.png (placeholder)")


def main():
    parser = argparse.ArgumentParser(description='Generate README media files')
    parser.add_argument('--all', action='store_true', help='Generate all media')
    parser.add_argument('--videos', action='store_true', help='Generate dataset videos')
    parser.add_argument('--plots', action='store_true', help='Generate result plots')
    parser.add_argument('--placeholders', action='store_true', help='Create placeholder diagrams')
    parser.add_argument('--experiment', type=str, default='pendulum', 
                        choices=['pendulum', 'IMU'], help='Experiment type')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("README MEDIA GENERATOR")
    print("=" * 70)
    
    setup_directories()
    
    if args.all or args.placeholders:
        create_architecture_placeholder()
        create_blocks_placeholder()
    
    if args.all or args.videos:
        try:
            generate_dataset_videos(experiment_type='pendulum', n_frames=150, fps=30)
            generate_dataset_videos(experiment_type='IMU', n_frames=150, fps=30)
        except Exception as e:
            print(f"⚠️ Video generation failed: {e}")
            print("   Make sure data files are in the data/ directory")
    
    if args.all or args.plots:
        print("\n⚠️ Plot generation requires running main.py first to get results")
        print("   Run: python main.py")
        print("   Then modify this script to load the results and call plot functions")
    
    print("\n" + "=" * 70)
    print("✓ Media generation complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review generated files in docs/images/ and docs/videos/")
    print("2. Create architecture diagrams manually (PowerPoint/Inkscape)")
    print("3. Run experiments and generate result plots")
    print("4. See docs/MEDIA_GUIDE.md for detailed instructions")


if __name__ == "__main__":
    main()
