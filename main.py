from utils import (read_video, save_video)
from Tracking import PlayerTracker
from Tracking import ShuttleTracker
from Court_line_detect import CourtDetector
import cv2
def main():
    # Read Videos
    input_video_path = "Input_PicVids/GinVsMom_Vid.mp4"
    video_frames, fps = read_video(input_video_path)

    # Detect Players and Shuttle
    player_tracker = PlayerTracker(model_path='yolo11m.pt')
    shuttle_tracker = ShuttleTracker(model_path= 'Models/last_shuttle.pt')
    court_detector = CourtDetector(model_path='Models/best_court.pt')
    player_detections = player_tracker.detect_frames(video_frames, 
                                                     read_from_stub=False, 
                                                     stub_path="Tracker_stubs/LAplayer_detection.pk1"
                                                     )
    shuttle_detections = shuttle_tracker.detect_frames(video_frames, 
                                                     read_from_stub=False, 
                                                     stub_path="Tracker_stubs/LAshuttle_detection.pk1"
                                                     )
    court_detections = court_detector.detect_frames(video_frames,
                                                    read_from_stub=False,
                                                    stub_path="Tracker_stubs/LAcourt_detection.pk1")
    
    shuttle_detections = shuttle_tracker.interpolate_shuttle_position(shuttle_detections)
    
    player_detections = player_tracker.choose_and_filter_players(court_detections[0], player_detections)
    
    # Draw Ouput

    ## Draw Player Bounding Boxes
    output_video_frames = court_detector.draw_keypoints(video_frames, court_detections)
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = shuttle_tracker.draw_bboxes(video_frames, shuttle_detections)

    # Draw frame on top left corner:
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (frame.shape[1] - 250,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (57,255,20), 2)

    save_video(output_video_frames, "Output/GM1_inter_2playerfps.avi", fps)

if __name__ == "__main__":
    main()