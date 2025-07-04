
import cv2
import numpy as np
from pathlib import Path
import csv
from collections import defaultdict


    # === 1. CONFIGURATION ===

    # --- INPUT: Path to the CSV log file ---
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / 'rat_path_log/log_total_distance.csv'  # <--- MAKE SURE THIS IS YOUR LOG FILE

    # --- OUTPUT: Path for the final plotted image ---
    output_image_path = project_root / 'outputs/rat_paths_plot.png'

    # --- CRUCIAL: Set the dimensions of the original video ---
    # This is needed to create a correctly sized canvas for the plot.
    # Find these values from the console output of the tracking script or by checking video properties.
    VIDEO_WIDTH = 1280  # <--- SET YOUR VIDEO'S WIDTH IN PIXELS
    VIDEO_HEIGHT = 980  # <--- SET YOUR VIDEO'S HEIGHT IN PIXELS

    # --- VISUALIZATION SETTINGS ---
    TRAIL_COLOR = (255, 0, 0)  # Blue in BGR format
    BACKGROUND_COLOR = (255, 255, 255)  # White in BGR format
    LINE_THICKNESS = 2

    # === 2. READ AND PROCESS CSV DATA ===
    print(f"Reading tracking data from '{csv_path}'...")

    # Group all points by their track_id
    # Structure: {track_id: [(x1, y1), (x2, y2), ...], ...}
    track_history = defaultdict(list)

    try:
        with open(str(csv_path), 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)  # Skip the header

            for row in csv_reader:
                track_id = int(row[1])
                x_center = int(row[2])
                y_center = int(row[3])
                track_history[track_id].append((x_center, y_center))
    except FileNotFoundError:
        print(f"Error: The log file was not found at '{csv_path}'")
        return
    except (ValueError, IndexError) as e:
        print(f"Error reading the CSV file. Please ensure it's formatted correctly. Details: {e}")
        return

    if not track_history:
        print("No tracking data found in the CSV file. Exiting.")
        return

    print(f"Found {len(track_history)} unique tracks. Ready to plot.")

    # === 3. CREATE BLANK CANVAS AND DRAW PATHS ===

    # Create a blank white canvas with the specified dimensions
    # np.full is a fast way to create an array filled with a specific value.
    # The dtype=np.uint8 is essential for image data.
    canvas = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)

    # Iterate through each track and draw its path
    for track_id, points in track_history.items():
        # Loop through the points to draw line segments
        for i in range(1, len(points)):
            p1 = points[i - 1]
            p2 = points[i]
            # Draw a line from the previous point to the current one
            cv2.line(canvas, p1, p2, TRAIL_COLOR, LINE_THICKNESS)

    # === 4. SAVE AND DISPLAY THE FINAL IMAGE ===

    # Save the final image to a file
    cv2.imwrite(str(output_image_path), canvas)
    print(f"\nSuccessfully plotted paths. Image saved to:\n{output_image_path}")

    # Display the image in a window
    cv2.imshow(f"Rat Paths Plot ({len(track_history)} tracks)", canvas)

    # Wait indefinitely until a key is pressed, then close the window
    print("\nPress any key to close the image window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

