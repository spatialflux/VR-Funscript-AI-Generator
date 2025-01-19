import os

from script_generator.gui.controller.debug_video import debug_video
from script_generator.object_detection.analye_tracking_results import analyze_tracking_results
from script_generator.object_detection.object_detection_result import ObjectDetectionResult
from script_generator.object_detection.utils import make_data_boxes, parse_yolo_data_looking_for_penis
from script_generator.utils.file import load_json_from_file, get_output_file_path
from script_generator.video.video_info import get_cropped_dimensions
from utils.lib_FunscriptHandler import FunscriptGenerator


def tracking_analysis(state):
    # Load YOLO detection results from file
    raw_yolo_path, _ = get_output_file_path(state.video_path, "_rawyolo.json")
    if not os.path.exists(raw_yolo_path):
        print(f"Raw yolo json not found in {raw_yolo_path}")
        return

    yolo_data = load_json_from_file(raw_yolo_path)
    state.set_video_info()
    video_info = state.video_info
    width, height = get_cropped_dimensions(state.video_info)

    results = make_data_boxes(yolo_data)

    # Looking for the first instance of penis within the YOLO results
    first_penis_frame = parse_yolo_data_looking_for_penis(yolo_data, 0)

    if first_penis_frame is None:
        print(f"No penis found in video: {state.video_path}")
        first_penis_frame = 0

    # Deciding whether we start from there or from a user-specified later frame
    state.frame_start = max(max(first_penis_frame - int(video_info.fps), state.frame_start - int(video_info.fps)), 0)

    state.frame_end = state.video_info.total_frames if not state.frame_end else state.frame_end

    print(f"Frame Start adjusted to: {state.frame_start}")

    # Performing the tracking part and generation of the raw funscript data
    state.funscript_data = analyze_tracking_results(state, results, progress_callback=lambda: debug_video(state))

    state.debugger.save_logs()

    funscript_handler = FunscriptGenerator()

    # Simplifying the funscript data and generating the file
    funscript_handler.generate(state)

    # Optionally, compare generated funscript with reference funscript if specified, or a simple generic report
    funscript_handler.create_report_funscripts(state)

    print(f"Finished processing video: {state.video_path}")
