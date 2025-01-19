import os
from tkinter import messagebox

from script_generator.state.app_state import AppState
from script_generator.utils.file import get_output_file_path
from utils.lib_Debugger import Debugger


def debug_video(state: AppState):
    if not state.video_path:
        messagebox.showerror("Error", "Please select a video file.")
        return

    state.debugger = Debugger(state.video_path, state.video_info.is_vr, state.video_reader, output_dir=state.video_path[:-4])  # Initialize the debugger

    # if the debug_logs.json file exists, load it
    logs_path, _ = get_output_file_path(state.video_path, "_debug_logs.json")
    if os.path.exists(logs_path):
        state.debugger.load_logs()

        state.debugger.play_video(
            start_frame=state.frame_start,
            duration=state.debug_record_duration if state.debug_record_mode else 0,
            record=state.debug_record_mode,
            downsize_ratio=2
        )
    else:
        messagebox.showinfo("Info", f"Debug logs file not found: {state.video_path[:-4] + f'_debug_logs.json'}")
