
from ultralytics import YOLO
from pathlib import Path


def main():
    # --- MODEL & DATA ---
    model = YOLO('../pre-trained_models/yolov8x.pt')

    # Use relative paths for better portability
    project_root = Path(__file__).resolve().parents[1]
    data_yaml_path = project_root / 'yolo_dataset/dataset.yaml'

    # --- TRAINING HYPERPARAMETERS ---

    print("Starting model training with improved hyperparameters...")
    results = model.train(
        # Essential Parameters
        data=str(data_yaml_path),
        epochs=150,  # Increase epochs, but use 'patience' to stop early if needed
        imgsz=640,  # Image size

        # Performance & Stability
        #batch=-1,  # BEST PRACTICE: Use AutoBatch to maximize GPU utilization
        batch=1,  # BEST PRACTICE: Use AutoBatch to maximize GPU utilization
        device=0,  # Use GPU '0'. Can be a list [0, 1] for multi-GPU.

        # Optimization
        optimizer='AdamW',  # 'AdamW' is a great default. 'SGD' is another option.
        lr0=0.00001,  # Initial learning rate. Default is 0.01. A slightly smaller LR can be more stable.
        lrf=0.01,  # Final learning rate factor (lr0 * lrf)

        # Regularization & Early Stopping
        patience=140,  # Stop training if no improvement is seen for 20 epochs. Prevents overfitting and saves time.
        weight_decay=0.0005,  # Regularization to prevent overfitting.

        # Augmentation (uncomment to override defaults)
        # mosaic=1.0,           # Mosaic augmentation (combining 4 images)
        # mixup=0.1,            # Mixup augmentation
        # hsv_h=0.015,          # Hue augmentation
        # hsv_s=0.7,            # Saturation augmentation
        # hsv_v=0.4,            # Value augmentation
        # degrees=0.0,          # Rotation
        # translate=0.1,        # Translation
        # scale=0.5,            # Scale
        # flipud=0.5,           # Flip up-down

        # Naming
        name='yolov8x_rat_detector_v4'  # Give your experiment a new version name
    )
    print("Training finished.")
    print(f"Model and results saved to: {results.save_dir}")


if __name__ == '__main__':
    main()