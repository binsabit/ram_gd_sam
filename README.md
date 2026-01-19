# RGBD-Tracker: RAM + GroundingDINO + SAM 2

**RGBD-Tracker** is an automated computer vision pipeline that integrates three state-of-the-art foundation models to achieve zero-shot detection and segmentation using an Intel RealSense camera.

Instead of training on specific object classes, this system "looks" at the video feed, recognizes objects dynamically, and segments them in real-time.

## 📝 How It Works

The pipeline chains together three powerful models:

1.  **RAM (Recognize Anything Model):** Scans the image and generates text labels for every object it detects (e.g., "person", "robot", "bottle").
2.  **GroundingDINO:** Takes the text labels from RAM and locates them in the image, generating bounding boxes.
3.  **SAM 2 (Segment Anything Model 2):** Takes the bounding boxes and generates precise segmentation masks for specific objects.

## 📂 Project Structure

```text
RAM_SAM_STARTER/
├── config/              # Configuration files (settings.yaml, model configs)
├── data/                # Data storage
├── src/                 # Source code
│   ├── inputs/          # Camera/Video input handlers
│   ├── models/          # Model wrappers (RAM, SAM, GroundingDINO)
│   ├── util/            # Utilities
│   └── main.py          # Entry point
├── weights/             # Model checkpoints (.pt, .pth)
├── install.sh           # Installation script
└── requirements.txt     # Python dependencies
```

## 🚀 Getting Started
Prerequisites

    OS: Linux (Recommended)

    GPU: NVIDIA GPU with CUDA support (Required for real-time inference)

    Hardware: Intel RealSense Depth Camera (D400 series recommended)

## Installation

    Clone the repository:

```bash
git clone <repository_url>
cd RAM_SAM_STARTER
```

Run the installation script: We have provided a script to set up the environment and dependencies automatically.

```bash
install.sh
```

Download Model Weights: Ensure the following weight files are placed inside the weights/ directory. You may need to download these from their respective model repositories if the install script does not fetch them automatically:

    groundingdino_swint_ogc.pth

    ram_plus_swin_large_14m.pth

    sam2_hiera_large.pt

## 💻 Usage

To start the tracking pipeline, run the main script from the root directory:

```bash
python src/main.py
```

## Controls

The system will display a window named annotated.

    Press q to stop the program.

Upon exit, the annotated video will be saved as annotated_video.mp4