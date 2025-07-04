import cv2
import os
import argparse
import sys

# --- Configuration: Set these paths to run the script with these defaults ---
# If SCRIPT_VIDEO_PATH is None and not provided via command line, the script will exit.
# SCRIPT_OUTPUT_FOLDER has a default value if not set here or via command line.

SCRIPT_VIDEO_PATH = r"/dataset/test1_out.avi"  # <<< CHANGE THIS to your video file path, or None
SCRIPT_OUTPUT_FOLDER = r"C:\BULabAssets\BULabProjects\RatDetectandTrack\dataset\frames" # <<< CHANGE THIS to your desired default output folder

# -----------------------------------------------------------------------------------------------

def video_to_frames(video_path, output_folder, image_format="png", skip_frames=0):
    """
    Extracts frames from a video file and saves them as images.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where frames will be saved.
        image_format (str): Format for the output images (e.g., 'png', 'jpg').
        skip_frames (int): Number of frames to skip between saves (0 means save all).
                           skip_frames=1 saves every 2nd frame, etc.
    """
    # --- 1. Input Validation ---
    if not video_path: # Added check for empty or None video_path
        print(f"Error: Video path is not provided.")
        sys.exit(1)
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        sys.exit(1) # Exit with an error code

    if image_format.lower() not in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
        print(f"Warning: Unsupported image format '{image_format}'. Defaulting to 'png'.")
        image_format = "png"

    if skip_frames < 0:
        print("Warning: skip_frames cannot be negative. Setting to 0.")
        skip_frames = 0

    # --- 2. Setup Output Directory ---
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output directory: '{output_folder}'")
        else:
            print(f"Output directory already exists: '{output_folder}'")
    except OSError as e:
        print(f"Error creating output directory '{output_folder}': {e}")
        sys.exit(1)

    # --- 3. Open Video File ---
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        sys.exit(1)

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(f"Video Info: Total Frames = {total_frames}, FPS = {fps:.2f}")

    # --- 4. Frame Extraction Loop ---
    frame_count = 0
    saved_count = 0
    success = True

    print("Starting frame extraction...")
    while success:
        # Read the next frame
        success, image = vidcap.read()

        if not success:
            break # End of video

        if frame_count % (skip_frames + 1) == 0:
            filename = f"frame_{str(saved_count).zfill(6)}.{image_format}"
            output_path = os.path.join(output_folder, filename)

            try:
                cv2.imwrite(output_path, image)
                saved_count += 1
                if saved_count % 100 == 0: # Print progress every 100 saved frames
                     print(f"Saved {saved_count} frames...")
            except Exception as e:
                print(f"Error writing frame {saved_count} to '{output_path}': {e}")

        frame_count += 1

    # --- 5. Cleanup ---
    vidcap.release()
    print("-" * 30)
    print(f"Frame extraction complete.")
    print(f"Total frames read: {frame_count}")
    print(f"Frames saved ({image_format}): {saved_count}")
    print(f"Output folder: '{os.path.abspath(output_folder)}'")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a video file into image frames. "
                    "Paths can be set in the script or overridden by command line."
    )

    # --- Positional Argument for video_file ---
    # nargs='?' makes it optional. If not provided on CLI, it takes the default value.
    parser.add_argument(
        "video_file",
        nargs='?',
        default=SCRIPT_VIDEO_PATH,
        help=f"Path to the input video file. "
             f"Defaults to SCRIPT_VIDEO_PATH in script (currently: '{SCRIPT_VIDEO_PATH}')."
    )

    # --- Optional Arguments ---
    parser.add_argument(
        "-o", "--output",
        default=SCRIPT_OUTPUT_FOLDER,
        help=f"Path to the folder where output frames will be saved. "
             f"Defaults to SCRIPT_OUTPUT_FOLDER in script (default: '{SCRIPT_OUTPUT_FOLDER}')."
    )

    parser.add_argument(
        "-f", "--format",
        default="png",
        choices=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Output image format (default: png)."
    )

    parser.add_argument(
        "-s", "--skip",
        type=int,
        default=0,
        help="Number of frames to skip between saves (0=save all, 1=save every 2nd, etc. default: 0)."
    )

    args = parser.parse_args()

    # Critical check: Ensure video_file is actually set either from script or CLI
    if not args.video_file:
        parser.error(
            "The video_file argument is required. "
            "Provide it on the command line or set SCRIPT_VIDEO_PATH in the script."
        )
        # parser.error() exits the script

    # Call the main function
    video_to_frames(
        video_path=args.video_file,
        output_folder=args.output,
        image_format=args.format,
        skip_frames=args.skip
    )