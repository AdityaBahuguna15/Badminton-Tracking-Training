from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
    
    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, player_detections_first_frame)

        # Assign top/bottom role from frame 1
        first_frame_bboxes = player_detections_first_frame
        player_positions = {tid: get_center_of_bbox(bbox) for tid, bbox in first_frame_bboxes.items() if tid in chosen_players}
        sorted_by_y = sorted(player_positions.items(), key=lambda x: x[1][1])
        top_player_id = sorted_by_y[0][0]
        bot_player_id = sorted_by_y[1][0]

        # Net y position — average of left and right net posts
        net_y = (court_keypoints[8][1] + court_keypoints[9][1]) / 2

        # Horizontal court bounds — outermost doubles corners
        # keypoint 0 = top-left doubles, 3 = top-right doubles
        # keypoint 14 = bottom-left doubles, 17 = bottom-right doubles
        min_court_x = min(court_keypoints[0][0], court_keypoints[14][0])
        max_court_x = max(court_keypoints[3][0], court_keypoints[17][0])

        # Service line midpoints
        top_service_mid = (
            (court_keypoints[6][0] + court_keypoints[7][0]) / 2,
            (court_keypoints[6][1] + court_keypoints[7][1]) / 2
        )
        bot_service_mid = (
            (court_keypoints[10][0] + court_keypoints[11][0]) / 2,
            (court_keypoints[10][1] + court_keypoints[11][1]) / 2
        )

        filtered_player_detections = []
        for player_dict in player_detections:
            frame_result = {}

             # Apply all filters: feet side of net, min height, within court x bounds
            top_candidates = {tid: bbox for tid, bbox in player_dict.items()
                            if bbox[3] < net_y
                            and (bbox[3] - bbox[1]) > 150
                            and get_center_of_bbox(bbox)[0] > min_court_x
                            and get_center_of_bbox(bbox)[0] < max_court_x}

            bot_candidates = {tid: bbox for tid, bbox in player_dict.items()
                            if bbox[3] >= net_y
                            and (bbox[3] - bbox[1]) > 150
                            and get_center_of_bbox(bbox)[0] > min_court_x
                            and get_center_of_bbox(bbox)[0] < max_court_x}

            if top_candidates:
                best_top = min(top_candidates.items(),
                            key=lambda x: measure_distance(get_center_of_bbox(x[1]), top_service_mid))
                frame_result[top_player_id] = best_top[1]

            if bot_candidates:
                best_bot = min(bot_candidates.items(),
                            key=lambda x: measure_distance(get_center_of_bbox(x[1]), bot_service_mid))
                frame_result[bot_player_id] = best_bot[1]

            filtered_player_detections.append(frame_result)

        return filtered_player_detections
    
    def choose_players(self, court_keypoints, player_dict):
        # Reconstruct the 2 service line midpoints from 18 keypoints
        # New idx 6 = service line top-left, 7 = service line top-right
        # New idx 10 = service line bottom-left, 11 = service line bottom-right
        top_service_mid = (
            (court_keypoints[6][0] + court_keypoints[7][0]) / 2,
            (court_keypoints[6][1] + court_keypoints[7][1]) / 2
        )
        bot_service_mid = (
            (court_keypoints[10][0] + court_keypoints[11][0]) / 2,
            (court_keypoints[10][1] + court_keypoints[11][1]) / 2
        )
        reference_points = [top_service_mid, bot_service_mid]

        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            # Distance to nearest service line midpoint
            min_distance = min(measure_distance(player_center, ref) for ref in reference_points)
            distances.append((track_id, min_distance))

        distances.sort(key=lambda x: x[1])
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
        
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0] # persists allows to remember the tracking from before
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict
    
    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
