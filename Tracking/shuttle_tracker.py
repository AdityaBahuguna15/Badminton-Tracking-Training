from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class ShuttleTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
    
    def interpolate_shuttle_position(self, shuttle_positions):
        shuttle_positions = [x.get(1, []) for x in shuttle_positions]
        # convert list into pandas dataframe
        df_shuttle_positions = pd.DataFrame(shuttle_positions, columns=['x1', 'y1', 'x2', 'y2'])

        #interpolate missing values
        df_shuttle_positions = df_shuttle_positions.interpolate(limit=2, limit_direction='both')
        # df_shuttle_positions = df_shuttle_positions.bfill()

        #shuttle_positions = [{1:x} for x in df_shuttle_positions.to_numpy().tolist()]
        shuttle_positions = [
        {1: x.tolist()} if not any(pd.isna(x)) else {}
        for x in df_shuttle_positions.to_numpy()
        ]

        return shuttle_positions

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        shuttle_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                shuttle_detections = pickle.load(f)
            return shuttle_detections
        
        for frame in frames:
            shuttle_dict = self.detect_frame(frame)
            shuttle_detections.append(shuttle_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(shuttle_detections, f)

        return shuttle_detections

    def detect_frame(self,frame):
        results = self.model.predict(frame, conf= 0.2, verbose=False)[0] # persists allows to remember the tracking from before
        id_name_dict = results.names

        shuttle_dict = {}
        best_conf = 0

        for box in results.boxes:
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]

            if object_cls_name == "Shuttle":
                conf = box.conf.tolist()[0]
                if conf > best_conf:
                    best_conf = conf
                    shuttle_dict = {1: box.xyxy.tolist()[0]}  # always ID 1
        
        return shuttle_dict
    
    def draw_bboxes(self,video_frames, shuttle_detections):
        output_video_frames = []
        for frame, shuttle_dict in zip(video_frames, shuttle_detections):
            # Draw Bounding Boxes
            for track_id, bbox in shuttle_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Shuttle",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.rectangle(frame, 
                              (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
