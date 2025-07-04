# scripts/track_video.py
import cv2
from ultralytics import YOLO
from pathlib import Path
import csv


def main():
    # --- MODIFIED: Define a static color for all tracks (BGR format) ---
    TRAIL_COLOR = (255, 0, 0)  # Pure Blue

    # --- Project and File Paths ---
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / 'scripts/runs/detect/yolov8x_rat_detector_v3/weights/best.pt'
    video_path = project_root / 'dataset/test1_out.avi'
    output_video_path = project_root / 'dataset/test4_blue_trails.mp4'  # New output name
    csv_path = project_root / 'rat_path_log/log5.csv'

    # --- Model and Video Initialization ---
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_MSMF)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # --- Data structures for tracking paths ---
    track_history = {}
    # --- REMOVED: The track_colors dictionary is no longer needed ---
    frame_idx = 0

    # --- Open CSV file for writing and write the header ---
    with open(str(csv_path), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['frame_number', 'track_id', 'x_center', 'y_center'])

        print("Processing video for rat detection and tracking...")
        print(f"Input: {video_path}")
        print(f"Output Video: {output_video_path}")
        print(f"Output CSV: {csv_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Perform tracking on the frame
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")

            # --- STEP 1: UPDATE HISTORY AND DRAW CURRENT BOXES ---
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    # --- REMOVED: All random color assignment logic ---

                    # Calculate center point
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Append the new point to the track history
                    points = track_history.get(track_id, [])
                    points.append((center_x, center_y))
                    track_history[track_id] = points

                    # Log data to CSV
                    csv_writer.writerow([frame_idx, track_id, center_x, center_y])

                    # --- MODIFIED: Draw the current bounding box and label using the static blue color ---
                    cv2.rectangle(frame, (x1, y1), (x2, y2), TRAIL_COLOR, 2)
                    label = f"Rat ID:{track_id}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), TRAIL_COLOR, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # --- STEP 2: DRAW ALL PERSISTENT TRAILS ---
            for track_id, points in track_history.items():
                # --- MODIFIED: Draw all trails using the static blue color ---
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    cv2.line(frame, points[i - 1], points[i], TRAIL_COLOR, 2)

            # --- STEP 3: OUTPUT THE FRAME ---
            out.write(frame)
            cv2.imshow('Rat Tracking with Blue Trails', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release all resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nFinished processing. Tracked video saved to: {output_video_path}")
    print(f"Path data saved to: {csv_path}")


if __name__ == '__main__':
    main()