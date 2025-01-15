import subprocess

from script_generator.video.video_info import VideoInfo
from params.config import ffprobe_path


def get_video_info(video_path):
    try:
        cmd = [
            ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,width,height,codec_name,nb_frames,duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]

        output = subprocess.check_output(cmd).decode("utf-8").splitlines()

        # Ensure the output has the correct number of fields
        if len(output) < 6:
            raise ValueError("FFProbe output is missing required fields.")

        # Parse metadata
        codec_name = output[0]
        width = int(output[1])
        height = int(output[2])
        r_frame_rate = output[3]
        duration = float(output[4])
        total_frames = int(output[5])

        # Calculate FPS
        num, den = map(int, r_frame_rate.split('/'))  # Split numerator and denominator
        fps = num / den  # Calculate FPS

        return VideoInfo(video_path, codec_name, width, height, duration, total_frames, fps)
    except Exception as e:
        print(f"Error initializing video info: {e}")
        raise