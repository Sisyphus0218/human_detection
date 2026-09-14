"""Download all project models into the fixed models/ directory."""

from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoImageProcessor, VitPoseForPoseEstimation
from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    print(f"Models directory: {MODELS_DIR}")

    # YOLO
    print("Downloading YOLO26-M and YOLO26-L...")
    (MODELS_DIR / "yolo").mkdir(parents=True, exist_ok=True)
    YOLO(MODELS_DIR / "yolo/yolo26m.pt")
    YOLO(MODELS_DIR / "yolo/yolo26l.pt")

    # OSNet
    print("\nDownloading OSNet...")
    hf_hub_download(
        repo_id="kaiyangzhou/osnet",
        filename=(
            "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_"
            "softmax_labsmth_flip_jitter.pth"
        ),
        local_dir=MODELS_DIR / "osnet",
    )

    # ViTPose
    print("\nDownloading ViTPose...")
    AutoImageProcessor.from_pretrained(
        "usyd-community/vitpose-base-simple",
        cache_dir=MODELS_DIR / "vitpose",
    )
    VitPoseForPoseEstimation.from_pretrained(
        "usyd-community/vitpose-base-simple",
        cache_dir=MODELS_DIR / "vitpose",
    )

    # MediaPipe: download the official Pose Landmarker Full bundle.
    print("\nDownloading MediaPipe...")
    mediapipe_path = MODELS_DIR / "mediapipe/pose_landmarker.task"
    mediapipe_path.parent.mkdir(parents=True, exist_ok=True)
    if not mediapipe_path.is_file():
        partial = mediapipe_path.with_suffix(".part")
        with tqdm(desc="MediaPipe", unit="B", unit_scale=True, unit_divisor=1024) as progress:
            def report_progress(block_count: int, block_size: int, total_size: int) -> None:
                if total_size > 0:
                    progress.total = total_size
                downloaded = block_count * block_size
                if total_size > 0:
                    downloaded = min(downloaded, total_size)
                progress.update(downloaded - progress.n)

            urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_full/float16/1/pose_landmarker_full.task",
                partial,
                reporthook=report_progress,
            )
        partial.replace(mediapipe_path)

    print("\nAll models are ready.")


if __name__ == "__main__":
    main()
