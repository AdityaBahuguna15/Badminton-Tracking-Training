from ultralytics import YOLO
import cv2
import pickle

class CourtDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.cached_keypoints = None  # cache for static court

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        court_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                court_detections = pickle.load(f)
            return court_detections

        for i, frame in enumerate(frames):
            # Only re-detect every 30 frames since court doesn't move
            if i % 30 == 0 or self.cached_keypoints is None:
                self.cached_keypoints = self.detect_frame(frame)
            court_detections.append(self.cached_keypoints)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(court_detections, f)

        return court_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.25, verbose=False)[0]

        keypoints_dict = {}
        if results.keypoints is not None and len(results.keypoints) > 0:
            kps = results.keypoints.xy[0].tolist()  # list of [x, y] for each keypoint
            for idx, kp in enumerate(kps):
                keypoints_dict[idx] = kp  # {0: [x,y], 1: [x,y], ...}

        return keypoints_dict

    def draw_keypoints(self, video_frames, court_detections):
        output_video_frames = []
        for frame, keypoints_dict in zip(video_frames, court_detections):
            for idx, (x, y) in keypoints_dict.items():
                if x == 0 and y == 0:  # skip undetected keypoints
                    continue
                cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)
                cv2.putText(frame, str(idx),
                            (int(x) + 6, int(y) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            output_video_frames.append(frame)

        return output_video_frames