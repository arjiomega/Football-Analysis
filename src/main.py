import logging
from pathlib import Path
from datetime import datetime

import cv2

from config import config
from src.trackers import Tracker
from src.utils import save_video, read_video
from src.team_classifier import TeamClassifier


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

    # Draw output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save Video
    video_name = "output"
    video_format = "avi"
    save_video_as = Path(save_dir, f"{video_name}.{video_format}")

    save_video(output_video_frames, str(save_video_as))


if __name__ == "__main__":
    setup_logging(level=logging.DEBUG)
    main()
