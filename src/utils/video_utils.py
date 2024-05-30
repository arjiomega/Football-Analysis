import cv2


def read_video(video_path: str):
    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)
    return frames


def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter.fourcc(*"XVID")

    WIDTH = output_video_frames[0].shape[1]
    HEIGHT = output_video_frames[0].shape[0]

    FPS = 24

    out = cv2.VideoWriter(output_video_path, fourcc, FPS, (WIDTH, HEIGHT))

    for frame in output_video_frames:
        out.write(frame)
    out.release()
