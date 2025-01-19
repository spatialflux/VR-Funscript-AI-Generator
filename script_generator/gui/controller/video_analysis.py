import threading
from tkinter import messagebox

from script_generator.gui.controller.tracking_analysis import tracking_analysis
from script_generator.object_detection.utils import check_skip_object_detection
from script_generator.scripts.analyse_video import analyse_video
from utils.lib_Debugger import Debugger


def video_analysis(state):
    if not state.video_path:
        messagebox.showerror("Error", "Please select a video file.")
        return

    print(f"Processing video: {state.video_path}")
    print(f"Video Reader: {state.video_reader}")
    print(f"Debug Mode: {state.debug_mode}")
    print(f"Live Display Mode: {state.life_display_mode}")
    print(f"Frame Start: {state.frame_start}")
    print(f"Frame End: {state.frame_end}")
    print(f"Reference Script: {state.reference_script}")

    def run():
        # Initialize the debugger
        state.debugger = Debugger(state.video_path, output_dir=state.video_path[:-4])

        skip_detection = check_skip_object_detection(state)

        if not skip_detection:
            analyse_video(state)

        tracking_analysis(state)

    processing_thread = threading.Thread(target=run)
    processing_thread.start()
    # processing_thread.join()