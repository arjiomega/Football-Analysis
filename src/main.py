import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from config import config
from src.trackers import Tracker
from src.utils import save_video, read_video
from src.team_classifier import TeamClassifier
from src.view_transformer import ViewTransformer
from src.player_ball_assigner import PlayerBallAssigner
from src.camera_movement_estimator import CameraMovementEstimator


# Configure logging
def setup_logging(level=logging.DEBUG):
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def main(model_name: str = "best.pt", video_path: str = config.SAMPLE_VID):

    report_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = Path(config.REPORTS_DIR, report_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Read Video
    video_frames = read_video(video_path)

    # Initialize Tracker
    model_path = Path(config.MODELS_DIR, model_name)
    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=Path(config.STUB_DIR, "track_stubs.pkl"),
    )

    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Camera Movement Estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path=Path(config.STUB_DIR, "camera_movement_stubs.pkl"),
    )

    #
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement_per_frame
    )

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign Player Teams
    team_classifier = TeamClassifier()
    team_classifier.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_classifier.get_player_team(
                video_frames[frame_num], track["bounding_box"], player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_classifier.team_colors[team]
            )

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bounding_box = tracks["ball"][frame_num][1]["bounding_box"]
        assigned_player = player_assigner.assign_ball_to_player(
            player_track, ball_bounding_box
        )

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(
                tracks["players"][frame_num][assigned_player]["team"]
            )
        else:
            team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)

    # Draw output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control
    )

    ## Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )

    # Save Video
    video_name = "output"
    video_format = "avi"
    save_video_as = Path(save_dir, f"{video_name}.{video_format}")

    save_video(output_video_frames, str(save_video_as))


if __name__ == "__main__":
    setup_logging(level=logging.DEBUG)
    main()
