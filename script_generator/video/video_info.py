from dataclasses import dataclass


@dataclass
class VideoInfo:
    path: str
    codec_name: str
    width: int
    height: int
    duration: float
    total_frames: int
    fps: float