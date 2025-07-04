
import cv2
from ultralytics import YOLO
import random
from pathlib import Path


def main():
    # --- BEST PRACTICE: USE RELATIVE PATHS ---
    # This makes your project work on any computer.
    project_root = Path(__file__).resolve().parents[1]  # This gets the 'yolo_rat_project' root directory

    # Path to your custom trained model
    model_path = project_root / 'scripts/runs/detect/yolov8m_rat_detector/weights/best.pt'

    # Path to the input video
    video_path = project_root / 'dataset/test1_out.avi'  # <--- UPDATE IF YOURS IS DIFFERENT

    # Path for the output video (changed to .mp4 for better compatibility)
    output_video_path = project_root / 'dataset/test1_output.mp4'

    # --- FIX 1: LOAD THE MODEL ---
    # Ensure the model path is a string for YOLO
    model = YOLO(str(model_path))

    # --- FIX 2: FORCE A RELIABLE VIDEO BACKEND ---
    # Use cv2.CAP_MSMF to bypass the broken GStreamer and use Windows Media Foundation
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_MSMF)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        print("This may be due to a missing codec or an issue with the video backend.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # --- FIX 3: USE A CORRECT FOURCC CODE AND MODERN CONTAINER ---
    # 'mp4v' is the FourCC code for MP4 video.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # Dictionary to store colors for each track ID
    track_colors = {}

    print("Processing video for rat detection and tracking...")
    print(f"Input: {video_path}")
    print(f"Output: {output_video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform tracking on the frame
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        # Check if tracking IDs are available
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, track_ids, confs):
                if track_id not in track_colors:
                    track_colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                color = track_colors[track_id]
                x1, y1, x2, y2 = box

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"Rat ID:{track_id} C:{conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)
        cv2.imshow('Rat Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nFinished processing. Tracked video saved to: {output_video_path}")


if __name__ == '__main__':
    main()
