
from ultralytics import YOLO


    # Path to your custom trained model weights
    # Make sure to update this path to where your 'best.pt' is located
    model_path = r'C:\RatDetectandTrack\scripts\runs\detect\yolov8n_rat_detector\weights\best.pt'

    # Path to your dataset.yaml file
    data_yaml_path = r'C:\RatDetectandTrack\yolo_dataset\dataset.yaml'

    # Load the trained model
    model = YOLO(model_path)

    print("Starting model validation on the 'val' set...")
    # Evaluate the model's performance on the validation set
    metrics = model.val(
        data=data_yaml_path,
        split='val',  # can be 'train', 'val', or 'test'
        imgsz=640,
        conf=0.25,  # confidence threshold for detection
        iou=0.7  # IoU threshold for NMS
    )

    print("Validation metrics:")
    print(f"Mean Average Precision (mAP50-95): {metrics.box.map}")
    print(f"Mean Average Precision (mAP50): {metrics.box.map50}")

    # Example of running prediction on a single image to test
    # Replace 'path/to/your/test_image.jpg' with an actual image
    # test_image_path = '../path/to/your/test_image.jpg'
    # print(f"\nRunning prediction on a test image: {test_image_path}")
    # results = model.predict(source=test_image_path, save=True)
    # print(f"Prediction results saved in the 'runs/detect/predict' folder.")
