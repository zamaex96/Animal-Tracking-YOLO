# scripts/plot_smooth_paths.py
import cv2
import numpy as np
from pathlib import Path
import csv
from collections import defaultdict
from scipy.interpolate import splprep, splev


def main():
    """
    Reads a tracking log, fits smooth B-spline curves to the paths,
    and plots them on a static background.
    """
    # === 1. CONFIGURATION ===
    # (Configuration section remains the same)
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / 'rat_path_log/log_total_distance2.csv'
    output_image_path = project_root / 'outputs/rat_paths_plot_smooth.png'
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    VIDEO_WIDTH = 1280
    VIDEO_HEIGHT = 980

    SMOOTHNESS_FACTOR = 0.5
    TRAIL_COLOR = (255, 0, 0)
    BACKGROUND_COLOR = (255, 255, 255)
    LINE_THICKNESS = 2

    # === 2. READ AND PROCESS CSV DATA ===
    # (Data reading section remains the same)
    print(f"Reading tracking data from '{csv_path}'...")
    track_history = defaultdict(list)
    try:
        with open(str(csv_path), 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)
            for row in csv_reader:
                track_id, x_center, y_center = int(row[1]), int(row[2]), int(row[3])
                track_history[track_id].append((x_center, y_center))
    except FileNotFoundError:
        print(f"Error: Log file not found at '{csv_path}'");
        return
    print(f"Found {len(track_history)} unique tracks. Ready to plot smooth paths.")

    # === 3. CREATE CANVAS AND DRAW SMOOTH PATHS ===
    canvas = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)

    for track_id, points in track_history.items():
        # --- NEW: PRE-PROCESSING STEP TO REMOVE CONSECUTIVE DUPLICATES ---
        if not points:
            continue  # Skip if for some reason a track has no points

        cleaned_points = [points[0]]  # Start with the first point
        for i in range(1, len(points)):
            # Only add the next point if it's different from the last one added
            if points[i] != cleaned_points[-1]:
                cleaned_points.append(points[i])

        # Now use the 'cleaned_points' list for spline fitting
        # Spline fitting requires at least 4 points for a cubic spline (k=3)
        if len(cleaned_points) < 4:
            # For very short tracks, just draw straight lines
            for i in range(1, len(cleaned_points)):
                cv2.line(canvas, cleaned_points[i - 1], cleaned_points[i], TRAIL_COLOR, LINE_THICKNESS)
            continue

        # --- CORE SPLINE LOGIC (now using cleaned_points) ---
        points_array = np.array(cleaned_points)
        x_coords = points_array[:, 0]
        y_coords = points_array[:, 1]

        smoothness = SMOOTHNESS_FACTOR * len(cleaned_points)
        try:
            tck, u = splprep([x_coords, y_coords], s=smoothness, k=3)
        except ValueError as e:
            # Add a catch for any other potential spline errors with this specific track
            print(f"Warning: Could not fit spline for track_id {track_id}. Skipping. Error: {e}")
            continue

        u_new = np.linspace(u.min(), u.max(), len(cleaned_points) * 10)
        x_smooth, y_smooth = splev(u_new, tck)

        smooth_points = np.vstack((x_smooth, y_smooth)).T.astype(np.int32)
        smooth_points = smooth_points.reshape((-1, 1, 2))

        cv2.polylines(canvas, [smooth_points], isClosed=False, color=TRAIL_COLOR, thickness=LINE_THICKNESS)

    # === 4. SAVE AND DISPLAY THE FINAL IMAGE ===
    # (This section remains the same)
    cv2.imwrite(str(output_image_path), canvas)
    print(f"\nSuccessfully plotted smooth paths. Image saved to:\n{output_image_path}")
    cv2.imshow(f"Smooth Rat Paths Plot", canvas)
    print("\nPress any key to close the image window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()