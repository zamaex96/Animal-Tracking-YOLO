
import json
import os
from pathlib import Path


def convert_coco_to_yolo(json_path, save_dir):
    """
    Converts COCO annotation format to YOLOv5 format.

    Args:
        json_path (str): Path to the COCO annotations JSON file.
        save_dir (str): Directory where the YOLO format .txt files will be saved.
    """
    path = Path(save_dir)
    if not path.exists():
        path.mkdir(parents=True)

    with open(json_path) as f:
        data = json.load(f)

    # Create a mapping from image ID to image filename
    images = {img['id']: img for img in data['images']}

    # Create a mapping from category ID to a continuous 0-based index
    cat_ids = {cat['id']: i for i, cat in enumerate(data['categories'])}

    # Process each annotation
    for ann in data['annotations']:
        image_id = ann['image_id']
        img_info = images[image_id]
        img_w, img_h = img_info['width'], img_info['height']

        # Get bounding box coordinates [x_min, y_min, width, height]
        bbox = ann['bbox']
        x1, y1, w, h = bbox

        # Convert to YOLO format (center_x, center_y, width, height) normalized
        x_center = (x1 + w / 2) / img_w
        y_center = (y1 + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h

        # Get the 0-based class index
        class_id = cat_ids[ann['category_id']]

        # Get the base filename without extension
        filename = Path(img_info['file_name']).stem
        txt_filename = path / f"{filename}.txt"

        # Append the annotation to the corresponding .txt file
        with open(txt_filename, 'a') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")


if __name__ == '__main__':
    # Define paths
    coco_train_json = r"C:\BULabAssets\BULabProjects\RatDetectandTrack\coco_dataset\annotations\instances_train.json"
    coco_val_json = r"C:\BULabAssets\BULabProjects\RatDetectandTrack\coco_dataset\annotations\instances_val.json"

    yolo_train_labels_dir = r"C:\BULabAssets\BULabProjects\RatDetectandTrack\yolo_dataset\labels\train"
    yolo_val_labels_dir = r"C:\BULabAssets\BULabProjects\RatDetectandTrack\yolo_dataset\labels\val"

    # Run conversion
    print("Converting training annotations...")
    convert_coco_to_yolo(coco_train_json, yolo_train_labels_dir)
    print("Converting validation annotations...")
    convert_coco_to_yolo(coco_val_json, yolo_val_labels_dir)
    print("Conversion complete!")

    # Also, copy image files to the correct yolo_dataset directory
    import shutil

    print("Copying image files...")
    for split in ['train', 'val']:
        src_img_dir = f'../coco_dataset/images/{split}'
        dst_img_dir = f'../yolo_dataset/images/{split}'
        Path(dst_img_dir).mkdir(parents=True, exist_ok=True)
        for img_file in os.listdir(src_img_dir):
            shutil.copy(os.path.join(src_img_dir, img_file), dst_img_dir)
    print("Image copying complete!")