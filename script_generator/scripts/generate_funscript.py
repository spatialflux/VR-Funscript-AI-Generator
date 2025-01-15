import queue
import string

from script_generator.scripts.analyse_video import analyse_video


def generate_funscript(video_path: string):
    result_q = queue.Queue(maxsize=0)
    analyse_video(video_path, result_q)