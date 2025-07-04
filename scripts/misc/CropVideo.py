
import subprocess
import os
import sys


def crop_video_ffmpeg(input_path, output_path, width, height, x_start, y_start):
    """
    Crops a video using a direct FFmpeg command. The output video will have the
    exact dimensions of the crop, with no black bars.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the cropped video file.
        width (int): The width of the cropped video.
        height (int): The height of the cropped video.
        x_start (int): The x-coordinate of the top-left corner of the crop.
        y_start (int): The y-coordinate of the top-left corner of the crop.
    """
    # Check if the input file exists before proceeding
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    try:
        # Construct the FFmpeg command
        # -y: Overwrite output file if it exists
        # -i: Specify the input file
        # -filter:v crop=...: The core crop filter. Syntax is width:height:x:y
        # -c:a copy: Copies the audio stream without re-encoding. This is fast and preserves quality.
        # -c:v libx264: A high-quality and widely compatible video codec (same as moviepy default)
        # -preset veryfast: A good balance of speed and file size for encoding
        command = [
            'ffmpeg',
            '-y',
           # '-c:v', 'h264',
            '-i', input_path,
            '-filter:v', f'crop={width}:{height}:{x_start}:{y_start}',
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-c:a', 'copy',
            output_path
        ]

        print("--- FFmpeg Command ---")
        # Join for display purposes, but the list is used for execution
        print(' '.join(command))
        print("----------------------\n")
        print("Starting video crop... This may take a moment.")

        # Execute the command.
        # We use Popen to potentially show real-time progress from FFmpeg, which prints to stderr.
        process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True, universal_newlines=True)

        # Print FFmpeg's output line-by-line to the console
        for line in process.stderr:
            sys.stderr.write(line)

        # Wait for the process to complete
        process.wait()

        if process.returncode == 0:
            print(f"\nDone! Cropped video saved successfully to '{output_path}'.")
        else:
            # The loop above will have already printed the error details from FFmpeg
            print(f"\nFFmpeg failed with return code {process.returncode}. Please check the output above for errors.")

    except FileNotFoundError:
        print("\nError: 'ffmpeg' command not found.")
        print("Please ensure FFmpeg is installed and accessible in your system's PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# This is the main part of the script that runs when you execute the file
if __name__ == "__main__":
    # === USER-DEFINED PARAMETERS ===
    # <<< IMPORTANT: Change these values to match your video and desired crop >>>

    # 1. Define the input and output file paths
    #    Use raw strings (r"...") on Windows to handle backslashes correctly
    input_video_path = r"C:\Users\user\Downloads\test1.avi"
    output_video_path = r"C:\Users\user\Downloads\test1_out.avi"

    # 2. Define the desired resolution of the cropped video
    crop_width = 1280
    crop_height = 980

    # 3. Define the starting point (top-left corner) of the crop
    #    (0, 0) is the top-left corner of the original video.
    start_x = 360
    start_y = 110

    # --- Example: To center a 1280x720 crop in a 1920x1080 video ---
    # original_w, original_h = 1920, 1080
    # crop_w, crop_h = 1280, 720
    # start_x = (original_w - crop_w) // 2  # This would be (1920 - 1280) / 2 = 320
    # start_y = (original_h - crop_h) // 2  # This would be (1080 - 720) / 2 = 180

    # ===============================

    # Call the function with the parameters defined above
    crop_video_ffmpeg(
        input_path=input_video_path,
        output_path=output_video_path,
        width=crop_width,
        height=crop_height,
        x_start=start_x,
        y_start=start_y
    )