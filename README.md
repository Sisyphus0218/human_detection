# Human Detection

A person-following pipeline for target tracking, pose and 3D position estimation, and robot velocity control. Register a target from reference images and track them in video or RGB-D camera streams.

## Getting Started

### Installation

1. Clone the repository.

   ```bash
   git clone https://github.com/Sisyphus0218/human_detection.git
   cd human_detection
   ```

2. Create an environment and install dependencies.

   ```bash
   # Create the environment
   uv venv --python 3.12
   
   # Activate the environment (choose one)
   source .venv/bin/activate       # Ubuntu (Bash)
   .\.venv\Scripts\Activate.ps1    # Windows (PowerShell)
   
   # Install PyTorch; choose a CUDA build compatible with your GPU and driver
   uv pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
   
   # Install project dependencies
   uv pip install -r requirements.txt
   ```

   For PrimeSense camera input, first download and install the OpenNI2 SDK for your platform from [the OpenNI2 download page](https://structure.io/openni/). Then install the Python bindings:

   ```bash
   uv pip install primesense
   ```

3. Install an FFmpeg build with the `libx264` encoder and add its executable directory to your system `PATH`. Verify that FFmpeg is available from the terminal you use to run tracking:

   ```bash
   ffmpeg -version
   ffmpeg -encoders
   ```
   
   Confirm that `libx264` appears in the encoder list. After tracking finishes, the program uses FFmpeg to convert the temporary video to the final `tracking.mp4`. If FFmpeg is unavailable, this final conversion fails. You can skip this dependency when running with `video_enabled=false`.

### Download Checkpoints

Run the following command from the project root to download the models listed below:

```bash
python scripts/download_models.py
```

| Model | Purpose | Source |
| --- | --- | --- |
| YOLO26-M | Person detection and tracking | [Ultralytics](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt) |
| YOLO26-L | Person detection and tracking | [Ultralytics](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt) |
| OSNet-AIN x1.0, trained on MSMT17 | Appearance feature extraction | [OSNet](https://huggingface.co/kaiyangzhou/osnet/tree/main) |
| ViTPose Base Simple | Pose estimation | [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple/tree/main) |
| MediaPipe Pose Landmarker Full | Pose estimation | [MediaPipe](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task) |

The models are saved in the following locations under the project root:

```text
models/
├── yolo/
│   ├── yolo26m.pt
│   └── yolo26l.pt
├── osnet/
│   └── osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth
├── vitpose/
│   └── models--usyd-community--vitpose-base-simple/
│       ├── refs/
│       │   └── main
│       └── snapshots/
│           └── <commit-hash>/
│               ├── config.json
│               ├── preprocessor_config.json
│               └── model.safetensors
└── mediapipe/
    └── pose_landmarker.task
```

`<commit-hash>` identifies the downloaded Hugging Face repository revision and is filled in automatically by the downloader. Auxiliary cache files are omitted from the tree.

You can also download the files manually from the source links above and place them in the corresponding locations, using the filenames shown in the tree.

### Run Tracking

1. Prepare target images and place them under `inputs/target/<target-name>/`, for example, `inputs/target/person_a/`:

   ```text
   inputs/
   └── target/
       └── person_a/
           ├── 01.jpg
           └── 02.jpg
   ```

   Use images containing only the intended target, ideally from different viewpoints. At least one image must contain a detectable person.

2. If you use a video as input, place it under `inputs/video/`:

   ```text
   inputs/
   └── video/
       └── demo.mp4
   ```

   Run tracking with the video filename without its `.mp4` extension:

   ```bash
   python -m human_detection.main target.name=person_a source=video source.name=demo
   ```

   Ordinary RGB video supports person tracking and 2D pose estimation, but does not provide the depth required for 3D position estimation or robot following.

3. If you use a PrimeSense camera as input, install the camera dependencies described in [Installation](#installation), then connect the camera. The default OpenNI2 runtime directory is `C:\Program Files\OpenNI2\Redist`; update `openni2_redist_path` in [the camera configuration](configs/source/primesense_camera.yaml) if your installation uses another location.

   ```bash
   python -m human_detection.main target.name=person_a source=primesense_camera
   ```

   Configure the camera intrinsics and camera-to-robot extrinsics in [configs/source/primesense_camera.yaml](configs/source/primesense_camera.yaml). The camera preset uses 320 x 240 frames at 30 FPS. For robot following, provide velocity feedback through the `velocity_feedback_provider` argument to `TrackingPipeline.run()`.

4. To replay an Intel RealSense recording, pass its `.bag` or SQLite-backed
   `.db3` path through the `realsense_bag` source:

   ```bash
   python -m human_detection.main target.name=person_a source=realsense_bag source.bag_path='C:\path\to\recording.db3'
   ```

   Playback runs as fast as inference allows by default. Set
   `source.real_time=true` to preserve the recording pace. Color and depth are
   aligned by librealsense before each frame enters the tracking pipeline, and
   the recorded color-camera intrinsics are used automatically for 3D position
   estimation unless `source.intrinsics` is explicitly overridden.

5. The program first registers the target from the reference images, then tracks them in the selected video or camera stream. It displays annotated frames and saves the output video. Press **Q** or **Esc** while the display window is focused to stop.

6. Results are saved in the following directory, where `<source-name>` is the video name (for example, `demo`) or `camera`:

   ```text
   results/
   └── <timestamp>-<source-name>-<target-name>/
       ├── crops/          # Person crops used for registration
       └── tracking.mp4    # Annotated tracking video
   ```


## Configuration

### Common Settings

Edit common settings in [configs/config.yaml](configs/config.yaml), or specify them in the command line using the examples below. Command-line arguments take precedence over YAML settings.

| Override | Description | Default | Example |
| --- | --- | --- | --- |
| `target.name` | Target reference-image folder name; required for both input types. | Required | `target.name=person_a` |
| `source` | Input source: `video`, `primesense_camera`, or `realsense_bag`. | `video` | `source=realsense_bag` |
| `source.name` | Video filename without `.mp4`; the camera preset supplies its own name. | Required for video; `camera` for camera input | `source.name=demo` |
| `device` | Device for inference; use `cpu` if CUDA is unavailable. | `cuda:0` | `device=cpu` |
| `display_enabled` | Show the tracking window. | `true` | `display_enabled=false` |
| `video_enabled` | Save the tracking video. | `true` | `video_enabled=false` |
| `results.directory` | Output directory for registration crops and video. | Timestamped directory under `results/` | `results.directory=results/demo` |

For example, run video tracking on the CPU without a display window:

```bash
python -m human_detection.main target.name=person_a source=video source.name=demo device=cpu display_enabled=false
```

### Camera Settings

For PrimeSense input, edit [configs/source/primesense_camera.yaml](configs/source/primesense_camera.yaml). These settings are not needed for ordinary video input.

| Setting | Description |
| --- | --- |
| `openni2_redist_path` | Local OpenNI2 runtime directory. |
| `width`, `height`, `fps` | Supported camera capture resolution and frame rate; defaults are 320 × 240 at 30 FPS. |
| `intrinsics` | Calibrated 3 × 3 color-camera intrinsic matrix for the capture resolution. With `null`, the estimator approximates it from frame dimensions. |
| `camera_to_robot_rotation` | 3 × 3 rotation from camera coordinates to robot coordinates; replace the example with your installation calibration for robot following. |
| `camera_to_robot_translation_mm` | Camera-to-robot translation in millimeters; configure for your installation when using robot following. |

### Model Selection

The default models are YOLO26-L for detection and tracking, OSNet-AIN x1.0 for appearance features, and ViTPose for pose estimation. Change the following presets only when you want a different model configuration:

| Override | Description | Default | Example |
| --- | --- | --- | --- |
| `person_detector` | Registration detector: `accurate` uses YOLO26-L; `fast` uses YOLO26-M. | `accurate` | `person_detector=fast` |
| `person_tracker` | Tracker: `accurate` uses YOLO26-L at input size 1280; `fast` uses YOLO26-M at 320. | `accurate` | `person_tracker=fast` |
| `pose_estimator` | Pose backend: `vitpose` or `mediapipe`. | `vitpose` | `pose_estimator=mediapipe` |

To use YOLO26-M for both registration and tracking:

```bash
python -m human_detection.main target.name=person_a source=video source.name=demo person_detector=fast person_tracker=fast
```

To use MediaPipe instead of ViTPose:

```bash
python -m human_detection.main target.name=person_a source=video source.name=demo pose_estimator=mediapipe
```

## Acknowledgements

This project uses the following open-source projects and pretrained models:

- [Ultralytics](https://github.com/ultralytics/ultralytics) for person detection and tracking.
- [Torchreid / OSNet](https://github.com/KaiyangZhou/deep-person-reid) for appearance feature extraction.
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) and [Transformers](https://github.com/huggingface/transformers) for pose estimation.
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) for the alternative pose backend.
- [Hydra](https://github.com/facebookresearch/hydra) for configuration management.
