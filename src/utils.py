import numpy as np
import cv2 as cv


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