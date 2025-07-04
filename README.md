# YOLOv8 Real-Time Rat Detection and Tracking

This project uses a custom-trained YOLOv8 model to detect and track rats in video streams, calculate distance traveled, and generate path visualizations and heatmaps.

## Features
- Custom object detection with YOLOv8
- Real-time object tracking with path drawing
- Post-processing scripts to generate path plots and heatmaps from log files
- Calculation of real-world distance traveled

## Setup

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/your-username/yolo-rat-tracking.git
   cd yolo-rat-tracking
   \`\`\`

2. **Install Git LFS and pull the model:**
   \`\`\`bash
   git lfs install
   git lfs pull
   \`\`\`

3. **Create and activate the Conda environment:**
   \`\`\`bash
   conda create --name yolo_rats python=3.9 -y
   conda activate yolo_rats
   \`\`\`

4. **Install dependencies:**
   \`\`\`bash
   pip install ultralytics opencv-python numpy scipy
   \`\`\`

## Usage

**Run the main tracking script:**
\`\`\`bash
python scripts/track_video.py
\`\`\`

**Generate visualizations from the log file:**
\`\`\`bash
python scripts/plot_smooth_paths.py
python scripts/generate_heatmap_v2.py
\`\`\`
