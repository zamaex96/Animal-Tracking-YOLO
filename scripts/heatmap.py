
import cv2
import numpy as np
from pathlib import Path
import csv


def draw_color_bar(height, width, colormap):
    """Creates a color bar legend for the heatmap."""
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        # The color is determined by the vertical position
        color_val = int((i / height) * 255)
        # Apply the colormap to the inverted value (so hot is at the top)
        color = cv2.applyColorMap(np.array([[255 - color_val]], dtype=np.uint8), colormap)[0][0]
        cv2.line(bar, (0, i), (width, i), color.tolist(), 1)

    # Add text labels
    cv2.putText(bar, "Hot", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(bar, "Cool", (5, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return bar


def main():
    """
    Generates a high-quality heatmap with clearer, larger hotspots by drawing
    "heat disks" and blending them with the original video's background.
    """
    # === 1. CONFIGURATION ===

    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / 'rat_path_log/log_total_distance.csv'
    original_video_path = project_root / 'dataset/test1_out.avi'  # Needed for background
    output_image_path = project_root / 'outputs/rat_residence_heatmap_v2.png'
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    VIDEO_WIDTH = 1280
    VIDEO_HEIGHT = 980

    # --- NEW & IMPROVED HEATMAP SETTINGS ---
    # Radius of the "heat disk" drawn for each point. This is the primary control for hotspot size.
    HEAT_DISK_RADIUS = 20

    # Milder blur to smooth the edges of the disks. Keep this relatively small. MUST BE ODD.
    BLUR_KERNEL_SIZE = (31, 31)

    COLORMAP_TO_USE = cv2.COLORMAP_JET

    # Alpha blending weight (0.0 = transparent, 1.0 = opaque heatmap)
    HEATMAP_OPACITY = 0.7

    # === 2. READ ALL POINTS FROM CSV ===
    print(f"Reading tracking data from '{csv_path}'...")
    all_points = []
    # (This section is the same as before)
    try:
        with open(str(csv_path), 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)
            for row in csv_reader:
                x_center, y_center = int(row[2]), int(row[3])
                all_points.append((x_center, y_center))
    except FileNotFoundError:
        print(f"Error: Log file not found at '{csv_path}'");
        return
    if not all_points:
        print("No data points found. Exiting.");
        return
    print(f"Loaded {len(all_points)} data points. Generating heatmap...")

    # === 3. GENERATE THE HEATMAP USING "HEAT DISKS" ===

    heatmap_accumulator = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.float32)

    # --- THIS IS THE KEY CHANGE ---
    # Instead of incrementing a single pixel, draw a filled circle (a disk) for each point.
    for x, y in all_points:
        if 0 <= x < VIDEO_WIDTH and 0 <= y < VIDEO_HEIGHT:
            cv2.circle(heatmap_accumulator, (x, y), HEAT_DISK_RADIUS, 1, -1)  # Add 'heat' in a radius

    # Now, apply a *milder* blur to smooth the disks
    blurred_heatmap = cv2.GaussianBlur(heatmap_accumulator, BLUR_KERNEL_SIZE, 0)
    normalized_heatmap = cv2.normalize(blurred_heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colored_heatmap = cv2.applyColorMap(normalized_heatmap, COLORMAP_TO_USE)

    # === 4. BLEND HEATMAP WITH BACKGROUND AND SAVE ===

    # Read the first frame of the video to use as a background
    #cap = cv2.VideoCapture(str(original_video_path))
    cap = cv2.VideoCapture(str(original_video_path), cv2.CAP_MSMF)
    ret, background_frame = cap.read()
    cap.release()

    if not ret:
        print("Warning: Could not read background frame. Using a black background.")
        background_frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)

    # Resize background if it doesn't match the specified dimensions
    background_frame = cv2.resize(background_frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

    # Blend the heatmap onto the background frame
    # final_image = background * (1 - alpha) + heatmap * alpha
    blended_image = cv2.addWeighted(background_frame, 1 - HEATMAP_OPACITY, colored_heatmap, HEATMAP_OPACITY, 0)

    # --- Add a color bar legend for context ---
    color_bar = draw_color_bar(height=300, width=50, colormap=COLORMAP_TO_USE)
    # Place the color bar on the top-right corner of the image
    bar_h, bar_w, _ = color_bar.shape
    blended_image[10:10 + bar_h, VIDEO_WIDTH - 10 - bar_w:VIDEO_WIDTH - 10] = color_bar

    cv2.imwrite(str(output_image_path), blended_image)
    print(f"\nSuccessfully generated heatmap. Image saved to:\n{output_image_path}")

    cv2.imshow("High-Quality Heatmap", blended_image)
    print("\nPress any key to close the image window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()