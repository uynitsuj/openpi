import argparse
from pathlib import Path
from torchvision.io import read_video_timestamps
from tqdm import tqdm


def get_fps_from_video(video_path: Path):
    """Compute FPS, frame count, and duration for a single video."""
    try:
        pts, _ = read_video_timestamps(str(video_path), pts_unit='sec')
        if len(pts) < 2:
            return (video_path, 0.0, len(pts), 0.0)

        pts_float = [float(p) for p in pts]
        time_diffs = [t2 - t1 for t1, t2 in zip(pts_float[:-1], pts_float[1:])]
        avg_time_diff = sum(time_diffs) / len(time_diffs)
        fps = 1.0 / avg_time_diff if avg_time_diff > 0 else 0.0
        duration = pts_float[-1] - pts_float[0]
        return (video_path, fps, len(pts), duration)
    except Exception as e:
        return (video_path, "ERROR", str(e), "")


def scan_and_check_videos(root_dir: Path):
    """Recursively find videos and compute FPS statistics."""
    results = []
    video_paths = list(root_dir.rglob("top_camera-images-rgb.mp4"))
    for video_path in tqdm(video_paths, desc="Scanning videos", unit="video"):
        result = get_fps_from_video(video_path)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute average FPS of videos under a directory.")
    parser.add_argument("--yam_data_path", type=str, required=True,
                        help="Root path containing video files (e.g., top_camera-images-rgb.mp4)")
    args = parser.parse_args()

    root = Path(args.yam_data_path)
    if not root.exists():
        print(f"Error: Provided path '{root}' does not exist.")
        return

    results = scan_and_check_videos(root)

    # Filter out only valid numeric FPS values
    fps_values = [fps for _, fps, _, _ in results if isinstance(fps, float)]
    mean_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0

    print(f"\nFound {len(results)} videos")
    print(f"video_mean_fps: {mean_fps:.2f}")

    

if __name__ == "__main__":
    main()
