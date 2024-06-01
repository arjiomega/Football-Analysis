import os
import pickle
import logging

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from src.utils import get_bounding_box_width, get_center_of_bounding_box


class Tracker:
    def __init__(self, model_path) -> None:
        logging.info("Initializing Tracker with model path: %s", model_path)
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        BATCH_SIZE = 20
        detections = []

        logging.info(
            "Starting detection on %d frames with batch size %d",
            len(frames),
            BATCH_SIZE,
        )

        for i in range(0, len(frames), BATCH_SIZE):
            logging.debug(
                "Processing batch from frame %d to %d",
                i,
                min(i + BATCH_SIZE, len(frames)),
            )
            batch_detections = self.model.predict(frames[i : i + BATCH_SIZE], conf=0.1)
            detections += batch_detections
            logging.debug(
                "Batch detection completed, total detections: %d", len(detections)
            )

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            logging.info("Reading tracks from stub file: %s", stub_path)
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        logging.info("Detecting frames for tracking")
        detections = self.detect_frames(frames)

        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inverse = {value: key for key, value in class_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            for object_idx, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_idx] = class_names_inverse[
                        "player"
                    ]

            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            # [
            #   {0: {bounding_box:[0,0,0,0]}, 1: {bounding_box:[0,0,0,0]}, ...},
            #   {0: {bounding_box:[0,0,0,0]}, 1: {bounding_box:[0,0,0,0]}, ...}
            # ]
            # FOR EACH FRAME a new blank dict is added (CONCERN RESOLVED)
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            BOUNDING_BOX_IDX = 0
            CLASS_ID_IDX = 3
            TRACK_ID_IDX = 4

            for frame_detection in detection_with_tracks:
                bounding_box = frame_detection[BOUNDING_BOX_IDX].tolist()
                class_id = frame_detection[CLASS_ID_IDX]
                track_id = frame_detection[TRACK_ID_IDX]

                if class_id == class_names_inverse["player"]:
                    tracks["players"][frame_num][track_id] = {
                        "bounding_box": bounding_box
                    }
                if class_id == class_names_inverse["referee"]:
                    tracks["referees"][frame_num][track_id] = {
                        "bounding_box": bounding_box
                    }

            for frame_detection in detection_supervision:
                bounding_box = frame_detection[BOUNDING_BOX_IDX].tolist()
                class_id = frame_detection[CLASS_ID_IDX]
                track_id = frame_detection[TRACK_ID_IDX]

                if class_id == class_names_inverse["ball"]:
                    tracks["ball"][frame_num][1] = {"bounding_box": bounding_box}

        if stub_path is not None:
            logging.info("Saving tracks to stub file: %s", stub_path)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bounding_box, color, track_id=None):
        x1, y1, x2, y2 = bounding_box

        x_center, _ = get_center_of_bounding_box(bounding_box)
        width = get_bounding_box_width(bounding_box)

        cv2.ellipse(
            frame,
            center=(x_center, int(y2)),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        RECTANGLE_WIDTH = 40
        RECTANGLE_HEIGHT = 20

        x1_rect = x_center - (RECTANGLE_WIDTH // 2)
        x2_rect = x_center + (RECTANGLE_WIDTH // 2)

        y1_rect = (y2 - RECTANGLE_HEIGHT // 2) + 15
        y2_rect = (y2 + RECTANGLE_HEIGHT // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                -1,
            )

            x1_text = x1_rect + 12

            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_triangle(self, frame, bounding_box, color):
        x1, y1, x2, y2 = bounding_box

        y = int(y1)
        x, _ = get_center_of_bounding_box(bounding_box)

        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])

        BLACK = (0, 0, 0)
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, BLACK, 2)

        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []

        for (
            frame_num,
            frame,
        ) in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw players
            RED = (0, 0, 255)

            for track_id, player in player_dict.items():
                team_color = player.get("team_color", RED)
                frame = self.draw_ellipse(
                    frame, player["bounding_box"], team_color, track_id
                )

            # Draw referees
            YELLOW = (0, 255, 255)

            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bounding_box"], YELLOW)

            output_video_frames.append(frame)

            # Draw ball
            GREEN = (0, 255, 0)

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bounding_box"], GREEN)

        return output_video_frames
