# Imports
import os
import cv2
import uuid
import yt_dlp
import configparser
from pathlib import Path

def download_video(url: str, output_dir: Path) -> tuple[uuid.UUID, str| None]:

    video_uuid = uuid.uuid4()
    ydl_opts: dict = {
        "format": "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": f"{output_dir}/{video_uuid}.%(ext)s",
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(params=ydl_opts) as ydl:
        # Gather video info
        info = ydl.extract_info(url, download=False)
        title = info.get("title", "Unknown Title")
        if isinstance(title, str):
            title = title.replace(" ", "_")  # reassign the result

        # Download video
        ydl.download([url])

    return video_uuid, title

def get_video_properties(video_path: str) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(video_path, cv2.CAP_ANY)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps   = int(cap.get(cv2.CAP_PROP_FPS)) or int(30)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return (fps, vid_w, vid_h)

def read_video(video_path: str) -> list:
    frames = []
    cap = cv2.VideoCapture(video_path, cv2.CAP_ANY)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)

    return frames

def save_video(video_output_path: str, frames: list, fps: int, vid_w: int, vid_h: int) -> None:
    out = cv2.VideoWriter(
        video_output_path,
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (vid_w, vid_h),
    )
    
    if not out.isOpened():
        raise IOError(f"Could not open VideoWriter at: {video_output_path}")
    
    for frame in frames:
        out.write(frame)

    out.release()


def convert_video_to_images(video_path: str, interval_seconds: float, output_dir: str = "screenshots") -> None:
    """
    Extract a screenshot from a video every X seconds.

    Args:
        video_path:       Path to the input video file.
        interval_seconds: Time interval (in seconds) between screenshots.
        output_dir:       Directory where screenshots will be saved.

    Returns:
        List of file paths to the saved screenshots.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError:        If interval_seconds is not positive.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be a positive number.")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval_seconds)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    saved_paths = []
    frame_number = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        if not success:
            break

        timestamp = frame_number / fps
        filename = f"{video_name}_{timestamp:.2f}s".replace(".","p").replace(" ","_")+".jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, frame)
        saved_paths.append(filepath)
        print(f"  Saved: {filepath}  (t={timestamp:.2f}s)")

        frame_number += frame_interval
        if frame_number >= total_frames:
            break

    cap.release()
    print(f"\nDone — {len(saved_paths)} screenshot(s) saved to '{output_dir}/'")

def convert_images_video(image_directory: str, video_output_path: str) -> None:
    """
    Creates a video from an image sequence using a seqinfo.ini config file.
 
    Args:
        image_directory: Path to the directory containing seqinfo.ini and the image folder.
        video_output_path: Output path for the video file.
    """
    sequence_dir = Path(image_directory)
    ini_path = sequence_dir / "seqinfo.ini"
 
    if not ini_path.exists():
        raise FileNotFoundError(f"seqinfo.ini not found in: {sequence_dir}")
 
    # Parse seqinfo.ini
    config = configparser.ConfigParser()
    config.read(ini_path)
    seq = config["Sequence"]
 
    name       = seq["name"]
    im_dir     = seq["imDir"]
    frame_rate = int(seq["frameRate"])
    seq_length = int(seq["seqLength"])
    width      = int(seq["imWidth"])
    height     = int(seq["imHeight"])
    ext        = seq["imExt"]
 
    img_dir = sequence_dir / im_dir
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
 
    # Collect and sort image files
    images = sorted(img_dir.glob(f"*{ext}"))
    if not images:
        raise ValueError(f"No '{ext}' images found in: {img_dir}")
 
    print(f"Found {len(images)} images (expected {seq_length})")
 
    # Determine output path — always .mp4
    output_path = Path(video_output_path) / f"{name}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    # Set up VideoWriter — try avc1 (H.264) first, fall back to mp4v
    writer = None
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter.fourcc(*codec)  # Fix: was missing the * unpack
        w = cv2.VideoWriter(str(output_path), fourcc, frame_rate, (width, height))
        if w.isOpened():
            writer = w
            print(f"Using codec: {codec}")
            break
        w.release()
 
    if writer is None:
        raise RuntimeError("Failed to open VideoWriter. Check codec support.")
 
    for i, img_path in enumerate(images, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  Warning: could not read {img_path.name}, skipping.")
            continue
 
        # Resize if frame dimensions don't match (safety net)
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
 
        writer.write(frame)
 
        if i % 100 == 0 or i == len(images):
            print(f"  Processed {i}/{len(images)} frames...")
 
    writer.release()
    print(f"\nVideo saved to: {output_path}")

