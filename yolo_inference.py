# from ultralytics import YOLO
# import cv2

# shuttle_model = YOLO('Models/last_shuttle.pt')
# #model = YOLO('yolo11s.pt')
# court_model = YOLO('Models/best_court.pt')
# #result = model.track('PictureVid\LZJvsVA_Vid.mp4', conf = 0.2, save=True)
# # results = model.track(
# #     source="PictureVid/LZJvsVA_Vid.mp4",
# #     classes=[0],              # person
# #     conf=0.3,
# #     tracker="bytetrack.yaml",
# #     persist=True,
# #     save=True
# # )
# #result = model.predict('PictureVid\LZJvsVA_Vid.mp4', conf = 0.05, save=True)
# # print(result)
# # print("boxes:")
# # for box in result[0].boxes:
# #     print(box)

# cap = cv2.VideoCapture("PictureVid/LZJvsVA_Vid.mp4")

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Shuttle detection
#     shuttle_results = shuttle_model.predict(frame, conf=0.2)

#     # Court keypoints
#     court_results = court_model.predict(frame, conf=0.25)

#     # Draw detections
#     annotated = shuttle_results[0].plot()

#     # Draw keypoints
#     keypoints = court_results[0].keypoints.xy

#     for kp in keypoints[0]:
#         x, y = int(kp[0]), int(kp[1])
#         cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)

#     cv2.imshow("Result", annotated)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2
import os

# ---- Models ----
shuttle_model = YOLO('Models/last_shuttle.pt')
court_model = YOLO('Models/best_court.pt')
player_model = YOLO('yolo11s.pt')

# ---- Paths ----
INPUT_VIDEO = 'Input_PicVids/LZJvsVA_Vid.mp4'
video_name = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]  # → 'LZJvsVA_Vid'
OUTPUT_VIDEO = f'runs/detect/{video_name}3_analyzed.mp4'
os.makedirs('runs/detect', exist_ok=True)

# ---- Video Setup ----
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height} @ {fps}fps — {total_frames} frames")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

frame_count = 0
cached_court = None  # initialize before loop

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- Player Detection ----
    player_results = player_model.track(
        frame,
        classes=[0],        # person only
        conf=0.3,
        tracker='bytetrack.yaml',
        persist=True,
        verbose=False,
        device=0
    )

    # ---- Shuttle Detection ----
    shuttle_results = shuttle_model.track(
        frame,
        classes=[0],
        conf=0.2,
        tracker='bytetrack.yaml',
        persist=True,
        verbose=False,
        device=0
    )

    # ---- Court Keypoints ----
    if frame_count % 30 == 0 or cached_court is None:
        cached_court = court_model.predict(
            frame,
            conf=0.25,
            verbose=False,
            device=0
    )

    # ---- Draw Everything ----
    # ---- Draw Everything ----
    # Layer 1 — players
    annotated = player_results[0].plot()

    # Layer 2 — shuttle on top
    annotated = shuttle_results[0].plot(img=annotated)

    # Layer 3 — court keypoints on top
    annotated = cached_court[0].plot(img=annotated)

    # ---- Frame Counter Overlay ----
    cv2.putText(annotated,
               f'Frame: {frame_count}/{total_frames}',
               (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.7, (255, 255, 255), 2)

    out.write(annotated)
    frame_count += 1

    if frame_count % 100 == 0:
        print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")

cap.release()
out.release()
print(f"\nDone — saved to {OUTPUT_VIDEO}")