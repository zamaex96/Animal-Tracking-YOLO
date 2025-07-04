# scripts/track_video.py
import cv2
from ultralytics import YOLO
from pathlib import Path
import csv
import numpy as np
import math


def main():
    # --- REAL-WORLD CALIBRATION (in cm) ---
    REAL_WORLD_WIDTH_CM = 30.0
    REAL_WORLD_HEIGHT_CM = 25.0

    # --- Define a static color for all tracks (BGR format) ---
    TRAIL_COLOR = (255, 0, 0)  # Pure Blue

    # --- Project and File Paths ---
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / 'scripts/runs/detect/yolov8x_rat_detector_v4/weights/best.pt'
    video_path = project_root / 'dataset/test1_out.avi'
    output_video_path = project_root / 'outputs/test4_total_distance.mp4'

    # Define the primary log file path
    csv_path = project_root / 'rat_path_log/log_total_distance3.csv'
    # --- NEW: Define the summary file path based on the primary log file name ---
    summary_txt_path = csv_path.with_suffix('.summary.txt')

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists

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

    track_history = {}
    frame_idx = 0

    with open(str(csv_path), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['frame_number', 'track_id', 'x_center', 'y_center', 'confidence'])

        print("Processing video for rat detection and tracking...")
        print(f"Video Dimensions: {width}x{height} pixels")
        print(f"Real-world Dimensions: {REAL_WORLD_WIDTH_CM}x{REAL_WORLD_HEIGHT_CM} cm")

        # --- Main video processing loop (unchanged) ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()
                highest_conf_idx = np.argmax(confs)
                best_box = boxes[highest_conf_idx]
                best_track_id = track_ids[highest_conf_idx]
                best_conf = confs[highest_conf_idx]
                x1, y1, x2, y2 = best_box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                points = track_history.get(best_track_id, [])
                points.append((center_x, center_y))
                track_history[best_track_id] = points
                csv_writer.writerow([frame_idx, best_track_id, center_x, center_y, f"{best_conf:.4f}"])
            for track_id, points in track_history.items():
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None: continue
                    cv2.line(frame, points[i - 1], points[i], TRAIL_COLOR, 2)
            out.write(frame)
            cv2.imshow('Rat Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    # --- All video processing is done. Now, calculate distances. ---

    if width > 0 and height > 0:
        px_per_cm_x = width / REAL_WORLD_WIDTH_CM
        px_per_cm_y = height / REAL_WORLD_HEIGHT_CM
    else:
        px_per_cm_x, px_per_cm_y = 1, 1

    grand_total_distance_cm = 0.0
    for track_id, points in track_history.items():
        distance_for_track_cm = 0.0
        for i in range(1, len(points)):
            p1, p2 = points[i - 1], points[i]
            delta_x_cm = (p2[0] - p1[0]) / px_per_cm_x
            delta_y_cm = (p2[1] - p1[1]) / px_per_cm_y
            distance_for_track_cm += math.sqrt(delta_x_cm ** 2 + delta_y_cm ** 2)
        grand_total_distance_cm += distance_for_track_cm

    # --- Release all resources ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # --- NEW: Save the summary to a dedicated text file ---
    with open(summary_txt_path, 'w') as f:
        f.write(f"Source Video: {video_path.name}\n")
        f.write(f"Total Frames Processed: {frame_idx}\n")
        f.write(f"Total Distance Traveled (cm): {grand_total_distance_cm:.2f}\n")

    # --- Final Printout ---
    print(f"\nFinished processing. Tracked video saved to: {output_video_path}")
    print(f"Detailed path data saved to: {csv_path}")
    print(f"Summary data saved to: {summary_txt_path}")  # Inform the user of the new file

    print("\n" + "=" * 30)
    print(f"Total Distance Traveled by all Rats: {grand_total_distance_cm:.2f} cm")
    print("=" * 30)


if __name__ == '__main__':
    main()