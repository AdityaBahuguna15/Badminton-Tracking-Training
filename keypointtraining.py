from roboflow import Roboflow
import os
import shutil
from datetime import datetime
from ultralytics import YOLO
import torch

# ---- GPU Check ----
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Using: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ---- Paths ----
PROJECT_DIR = 'C:/Users/Aditya/BadmintonML'
RUNS_DIR = f'{PROJECT_DIR}/runs'
RUN_NAME = 'court_keypoints_v2'
WEIGHTS_DIR = f'{RUNS_DIR}/{RUN_NAME}/weights'
BACKUP_DIR = f'{PROJECT_DIR}/backups'

# ---- Backup Function ----
def backup_weights(weights_dir, backup_dir):
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backed_up = []

    for weight_file in ['best.pt', 'last.pt']:
        src = os.path.join(weights_dir, weight_file)
        if os.path.exists(src):
            dst = os.path.join(backup_dir, f"{timestamp}_keypoint_{weight_file}")
            shutil.copy2(src, dst)
            backed_up.append(dst)
            print(f"Backed up: {dst}")

    if not backed_up:
        print("No weight files found to backup")

if __name__ == '__main__':

    # GPU Check
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"Using: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Path to keypoint dataset
    rf = Roboflow(api_key="JK3XRYll8vdLCszAp83e")
    project = rf.workspace("adityas-workspace-0eca5").project("badmintoncourtdetectionoffical-b3hl9-dloud")
    version = project.version(1)
    dataset = version.download("yolov8")
    print(f"Dataset location: {dataset.location}")

    # Pose model — different from detection model
    # yolo11s-pose.pt is pretrained on human pose but transfers well to court keypoints
    model = YOLO('yolo11s-pose.pt')

    print("\nStarting keypoint training...")
    try:
        model.train(
            data=f"{dataset.location}/data.yaml",
            epochs=100,
            imgsz=960,
            batch=4,
            device=0,
            cache='disk',
            optimizer='AdamW',
            lr0 = 0.001,
            workers=2,
            patience=20,        # more patience — 1000 images is small
            save=True,
            save_period=5,
            plots=True,
            project=RUNS_DIR,
            name=RUN_NAME,
            exist_ok=True,
            amp=True,

            # Loss weights — important for keypoint accuracy
            box=7.5,            # bounding box loss
            cls=0.5,            # classification loss — only 1 class so keep low
            pose=12.0,          # keypoint loss weight — higher = more focus on keypoint accuracy
            kobj=2.0,           # keypoint objectness — confidence that keypoint exists
            
            # Safe Augmentation for keypoints
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            fliplr=0.5,       # high impact
            scale=0.1,        # medium impact
            hsv_v=0.2,        # medium impact
            hsv_s=0.1,        # minor
        )
        print("\nTraining complete")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    except Exception as e:
        print(f"\nTraining crashed: {e}")

    finally:
        print("\nSaving backup...")
        backup_weights(WEIGHTS_DIR, BACKUP_DIR)

# # Now read the data.yaml
# data_yaml_path = os.path.join(dataset.location, 'data.yaml')

# with open(data_yaml_path) as f:
#     print(f.read())