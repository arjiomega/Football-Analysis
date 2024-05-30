import logging
from pathlib import Path
from datetime import datetime

from config import config
from src.trackers import Tracker
from src.utils import save_video, read_video


# Configure logging
def setup_logging(level=logging.DEBUG):
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def main(model_name: str = "best.pt", video_path: str = config.SAMPLE_VID):

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

    # Draw output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save Video
    report_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = Path(config.REPORTS_DIR, report_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    video_name = "output"
    video_format = "avi"
    save_video_as = Path(save_dir, f"{video_name}.{video_format}")

    save_video(output_video_frames, str(save_video_as))


if __name__ == "__main__":
    setup_logging(level=logging.DEBUG)
    main()
