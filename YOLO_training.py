import shutil
import os
import zipfile
from datetime import datetime
from roboflow import Roboflow
from ultralytics import YOLO
import torch

# ---- GPU Check ----
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Using: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ---- Paths ----
PROJECT_DIR = 'C:/Users/Aditya/BadmintonML'
RUNS_DIR = f'{PROJECT_DIR}/runs'
RUN_NAME = 'badminton_detector'
WEIGHTS_DIR = f'{RUNS_DIR}/badminton_detector/weights'
BACKUP_DIR = f'{PROJECT_DIR}/backups'

# ---- Backup Function ----
def backup_weights(weights_dir, backup_dir):
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backed_up = []

    for weight_file in ['best.pt', 'last.pt']:
        src = os.path.join(weights_dir, weight_file)
        if os.path.exists(src):
            dst = os.path.join(backup_dir, f"{timestamp}_{weight_file}")
            shutil.copy2(src, dst)
            backed_up.append(dst)
            print(f"Backed up: {dst}")

    if not backed_up:
        print("No weight files found to backup")

# ---- Everything inside main guard ----
if __name__ == '__main__':
    
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"Using: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Download dataset from Roboflow
    rf = Roboflow(api_key="JK3XRYll8vdLCszAp83e")
    project = rf.workspace("adityas-workspace-0eca5").project("badminton-detection-0e58z")
    version = project.version(2)
    dataset = version.download("yolov11") 
    print(f"Dataset location: {dataset.location}")
    
    # Load Model
    # model = YOLO('yolo11s.pt')
    model = YOLO("runs/badminton_v2/weights/last.pt")
    # Train
    print("\nStarting training...")
    try:
        model.train(resume=True)
        # model.train(
        #     data=f"{dataset.location}/data.yaml",
        #     epochs=30,
        #     imgsz=640,
        #     batch=16,
        #     device=0,
        #     cache='disk',
        #     workers=1,
        #     patience=15,
        #     save=True,
        #     save_period=5,
        #     fraction=1.0,
        #     plots=True,
        #     project=RUNS_DIR,
        #     name='badminton_v2',
        #     exist_ok=True,
        #     amp=True,
        #     close_mosaic=10,
        #     # Class weighting — shuttle penalized more when missed
        #     cls=3.0,            # increases overall classification loss weight
        #     box=8.0,          # increases box regression weight — better localization
        #     # Augmentation for small objects
        #     copy_paste=0.0,     # paste shuttle instances into more images
        #     mixup=0.0,         # blend images
        # )
        print("\nTraining complete")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    except Exception as e:
        print(f"\nTraining crashed: {e}")

    finally:
        print("\nSaving backup...")
        backup_weights(WEIGHTS_DIR, BACKUP_DIR)



# # ---- Load Model ----
# model = YOLO('yolo11l.pt')

# # ---- Train ----
# print("\nStarting training...")
# try:
#     model.train(
#         data=f"{dataset.location}\data.yaml",
#         epochs=50,
#         imgsz=416,
#         batch=8,
#         device=0,
#         cache=False,
#         workers=2,
#         patience=10,
#         save=True,
#         save_period=1,
#         plots=True,
#         project=RUNS_DIR,
#         name=RUN_NAME,
#         exist_ok=True
#     )
#     print("\nTraining complete")

# except KeyboardInterrupt:
#     print("\nTraining interrupted by user")

# except Exception as e:
#     print(f"\nTraining crashed: {e}")

# finally:
#     print("\nSaving backup...")
#     backup_weights(WEIGHTS_DIR, BACKUP_DIR)