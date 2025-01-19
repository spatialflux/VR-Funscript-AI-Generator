import tkinter as tk
from tkinter import ttk

from script_generator.gui.controller.regenerate_funscript import regenerate_funscript
from script_generator.gui.controller.video_analysis import video_analysis
from script_generator.gui.messages.messages import UIMessage, ProgressMessage
from script_generator.gui.utils.widgets import Widgets
from script_generator.state.app_state import AppState
from script_generator.utils.helpers import is_mac


class FunscriptGeneratorPage(tk.Frame):
    def __init__(self, parent, controller):
        #region SETUP
        super().__init__(parent)
        self.controller = controller
        self.state: AppState = controller.state
        state: AppState = controller.state

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        wrapper = Widgets.frame(self, title=None, sticky="nsew")
        #endregion

        #region VIDEO SELECTION
        video_selection = Widgets.frame(wrapper, title="Video Selection", main_section=True)

        Widgets.file_selection(
            attr="video_path",
            parent=video_selection,
            label_text="Video",
            button_text="Browse",
            file_selector_title="Choose a File",
            file_types=[("Text Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")],
            state=state
        )

        Widgets.dropdown(
            attr="video_reader",
            parent=video_selection,
            label_text="Video Reader",
            options=["FFmpeg", *([] if is_mac() else ["FFmpeg + OpenGL (Windows)"])],
            default_value=state.video_reader,
            tooltip_text=("On Mac only FFmpeg is supported" if is_mac() else "FFmpeg + OpenGL is usually about 30% faster on a good GPU."),
            state=state,
            row=1
        )
        #endregion

        #region OPTIONAL SETTINGS
        optional_settings = Widgets.frame(wrapper, title="Optional settings", main_section=True, row=2)

        Widgets.input(optional_settings, "Frame Start", state=state, attr="frame_start")
        Widgets.input(optional_settings, "Frame End", state=state, attr="frame_start", row=1)
        #endregion

        #region PROCESSING
        processing = Widgets.frame(wrapper, title="Processing", main_section=True, row=3)
        yolo_p_container, yolo_p, yolo_p_label, yolo_p_perc = Widgets.labeled_progress(processing, "YOLO Detection", row=0)
        track_p_container, track_p, track_p_label, track_p_perc = Widgets.labeled_progress(processing, "Tracking Analysis", row=1)
        Widgets.button(processing, "Start processing", lambda: video_analysis(state), row=2)
        #endregion

        #region FUNSCRIPT TWEAKING
        tweaking = Widgets.frame(wrapper, title="Funscript", main_section=True, row=4)
        # tweaking.grid_rowconfigure(1, weight=10)
        tweaking_container = ttk.Frame(tweaking)
        tweaking_container.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        boost_frame = ttk.LabelFrame(tweaking_container, text="Boost Settings", padding=(10, 5))
        boost_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        Widgets.checkbox(boost_frame, "Boost Settings", state, "boost_enabled", False)
        Widgets.range_selector(
            parent=boost_frame,
            label_text="Boost Up %",
            state=self.state,
            attr='boost_up_percent',
            values=[str(i) for i in range(0, 21)],
            row=1
        )
        Widgets.range_selector(
            parent=boost_frame,
            label_text="Reduce Down %",
            state=self.state,
            attr='boost_down_percent',
            values=[str(i) for i in range(0, 21)],
            row=2
        )

        # Threshold Settings
        threshold_frame = ttk.LabelFrame(tweaking_container, text="Threshold Settings", padding=(10, 5))
        threshold_frame.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        Widgets.checkbox(threshold_frame, "Enable Threshold", state, "threshold_enabled", False)
        Widgets.range_selector(
            parent=threshold_frame,
            label_text="0 Threshold",
            state=self.state,
            attr='threshold_low',
            values=[str(i) for i in range(0, 16)],
            row=1
        )
        Widgets.range_selector(
            parent=threshold_frame,
            label_text="100 Threshold",
            state=self.state,
            attr='threshold_high',
            values=[str(i) for i in range(80, 101)],
            row=2
        )

        # Simplification Settings
        vw_frame = ttk.LabelFrame(tweaking_container, text="Simplification", padding=(10, 5))
        vw_frame.grid(row=1, column=3, padx=5, pady=5, sticky="ew")

        Widgets.checkbox(vw_frame, "Enable Simplification", state, "vw_simplification_enabled", False)
        Widgets.range_selector(
            parent=vw_frame,
            label_text="VW Factor",
            state=self.state,
            attr='vw_factor',
            values=[i / 5 for i in range(10, 51)],
            row=1
        )
        Widgets.range_selector(
            parent=vw_frame,
            label_text="Rounding",
            state=self.state,
            attr='rounding',
            values=[5, 10],
            row=2
        )

        # Regenerate Funscript Button
        Widgets.button(tweaking_container, "Regenerate Funscript", lambda: regenerate_funscript(self.state), row=2)

        #endregion

        #region DEBUGGING
        debugging = Widgets.frame(wrapper, title="Debugging", main_section=True, row=5)
        general = Widgets.frame(debugging, title="General", row=0)
        Widgets.checkbox(general, "Logging for debug", state, "debug_mode")

        script_compare = Widgets.frame(debugging, title="Script compare", row=1)
        Widgets.file_selection(
            attr="reference_script",
            parent=script_compare,
            label_text="Reference Script",
            button_text="Browse",
            file_selector_title="Choose a File",
            file_types=[("Funscript Files", "*.funscript"), ("All Files", "*.*")],
            state=state,
            row=0
        )

        object_detection = Widgets.frame(debugging, title="Object detection", row=2)

        Widgets.checkbox(object_detection, "Live display mode", state, "life_display_mode",tooltip_text="Will show a live preview of the object detection.", row=3)
        Widgets.checkbox(object_detection, "Save debugging video", state=state, attr="save_debug_as_video", row=5)
        Widgets.dropdown(
            attr="debug_record_duration_var",
            parent=object_detection,
            label_text="save a frame every",
            options=[5, 10, 20],
            default_value=5,
            state=state,
            col=2,
            row=5
        )
        Widgets.label(object_detection, text="seconds", row=5, column=3)
        Widgets.button(object_detection, "Play debug video (q to quit)", None, row=6)
        #endregion

        #region FOOTER
        Widgets.disclaimer(wrapper)
        #endregion

        #region UI UPDATE CALLBACK
        def update_ui(msg: UIMessage):
            """Handle UI updates using a switch-like statement."""

            def process_update():
                handlers = {
                    ProgressMessage: handle_progress_message
                }

                handler = handlers.get(type(msg))
                if handler:
                    handler(msg)
                else:
                    print(f"Unhandled message type: {type(msg)}")

            def handle_progress_message(progress_msg: ProgressMessage):
                if progress_msg.process == "OBJECT_DETECTION":
                    yolo_p["value"] = progress_msg.frames_processed
                    yolo_p["maximum"] = progress_msg.total_frames
                    percentage = (progress_msg.frames_processed / progress_msg.total_frames) * 100 if progress_msg.total_frames > 0 else 0
                    yolo_p_perc.config(text=f"{percentage:.0f}% - ETA: {progress_msg.eta}")


            # Schedule the update on the main thread
            self.controller.after(0, process_update)

        self.state.update_ui = update_ui
        #endregion