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

3. If you use a PrimeSense camera as input, install the camera dependencies described in [Installation](#installation), then connect the camera. The default OpenNI2 runtime directory is `C:\Program Files\OpenNI2\Redist`; update `openni2_redist_path` in [the camera configuration](configs/source/primesense_camera.yaml) if your installation uses another location.

   ```bash
   python -m human_detection.main target.name=person_a source=primesense_camera
   ```

   The camera preset uses 320 x 240 frames at 30 FPS.

4. The program first registers the target from the reference images, then tracks them in the selected video or camera stream. It displays annotated frames and saves the output video. Press **Q** or **Esc** while the display window is focused to stop.

5. Results are saved in the following directory, where `<source-name>` is the video name (for example, `demo`) or `camera`:

   ```text
   results/
   └── <timestamp>-<source-name>-<target-name>/
       ├── crops/          # Person crops used for registration
       └── tracking.mp4    # Annotated tracking video
   ```


## Configuration

Configuration is managed with Hydra. Defaults are defined in [configs/config.yaml](configs/config.yaml); configuration groups can be selected or individual values overridden from the command line.

Append the example arguments below to a tracking command to override the defaults.

| Override | Description | Default | Example |
| --- | --- | --- | --- |
| `device` | Device for YOLO, OSNet, and ViTPose; use `cpu` for CPU inference. | `cuda:0` | `device=cpu` |
| `person_detector` | Registration detector preset: `accurate` or `fast`. | `accurate` | `person_detector=fast` |
| `person_tracker` | Video tracker preset: `accurate` or `fast`. | `accurate` | `person_tracker=fast` |
| `pose_estimator` | Pose backend: `vitpose` or `mediapipe`. | `vitpose` | `pose_estimator=mediapipe` |
| `source` | Input source: `video` or `primesense_camera`. | `video` | `source=primesense_camera` |
| `results.directory` | Output directory for registration crops and video. | Timestamped directory under `results/` | `results.directory=results/demo` |

For example, run video tracking on the CPU:

```bash
python -m human_detection.main target.name=person_a source=video source.name=demo device=cpu
```

To inspect the resolved configuration without loading models or running inference:

```bash
python -m human_detection.main target.name=person_a source.name=demo --cfg job --resolve
```

## Acknowledgements

This project uses the following open-source projects and pretrained models:

- [Ultralytics](https://github.com/ultralytics/ultralytics) for person detection and tracking.
- [Torchreid / OSNet](https://github.com/KaiyangZhou/deep-person-reid) for appearance feature extraction.
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) and [Transformers](https://github.com/huggingface/transformers) for pose estimation.
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) for the alternative pose backend.
- [Hydra](https://github.com/facebookresearch/hydra) for configuration management.
