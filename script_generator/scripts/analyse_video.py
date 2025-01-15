import queue
import string
import threading
import time

from tqdm import tqdm
from colorama import Fore

from script_generator.config import QUEUE_MAXSIZE, SEQUENTIAL_MODE, PROGRESS_BAR, UPDATE_PROGRESS_INTERVAL
from script_generator.video_conversion.vr_to_2d_task_processor import VrTo2DTaskProcessor
from script_generator.tasks.abstract_task_processor import TaskProcessorTypes
from script_generator.tasks.tasks import AnalyseVideoTask
from script_generator.video.ffmpeg import get_video_info
from script_generator.video.video_task_processor import VideoTaskProcessor
from script_generator.yolo.yolo import YoloTaskProcessor


def analyse_video(video_path: string, result_queue: queue.Queue):
    print(f"[OBJECT DETECTION] Starting up pipeline with profiling in {'sequential mode' if SEQUENTIAL_MODE else 'parallel mode'}...")

    # Initialize batch task
    video = get_video_info(video_path)
    batch_task = AnalyseVideoTask(video)

    # Create queues
    opengl_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    yolo_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    # yolo_analysis_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

    # create threads
    decode_thread = VideoTaskProcessor(batch_task=batch_task, output_queue=opengl_queue)
    opengl_thread = VrTo2DTaskProcessor(batch_task=batch_task, input_queue=opengl_queue, output_queue=yolo_queue)
    yolo_thread = YoloTaskProcessor(batch_task=batch_task, input_queue=yolo_queue, output_queue=result_queue)
    # yolo_analysis_thread = YoloAnalysisTaskProcessor(batch_task=batch_task, input_queue=yolo_analysis_queue, output_queue=result_queue)

    # Start logging thread
    log_thread_stop_event = threading.Event()
    queue_logging_thread = threading.Thread(
        target=log_progress,
        args=(batch_task, opengl_queue, yolo_queue, result_queue, log_thread_stop_event),
        daemon=True,
    )
    queue_logging_thread.start()

    # Sequential mode can be used to determine performance bottlenecks on very short videos
    if SEQUENTIAL_MODE:
        def run_thread(thread, thread_name, out_queue):
            start_time = time.time()
            thread.start()
            thread.join()
            out_queue.put(None)
            print(f"[OBJECT DETECTION] {thread_name} thread done in {time.time() - start_time} s")

        run_thread(decode_thread, TaskProcessorTypes.VIDEO, opengl_queue)
        run_thread(opengl_thread, TaskProcessorTypes.OPENGL, yolo_queue)
        run_thread(yolo_thread, TaskProcessorTypes.YOLO_ANALYSIS, result_queue)
        # run_thread(yolo_analysis_thread, "YOLO analysis")
    else:
        threads = [decode_thread, opengl_thread, yolo_thread]  # , yolo_analysis_thread
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    batch_task.end_time = time.time()

    # wait for update progress thread to close
    time.sleep(UPDATE_PROGRESS_INTERVAL)

    log_performance(batch_task=batch_task, results_queue=result_queue)


def log_progress(batch_task, opengl_q, yolo_q, results_q, stop_event):
    total_frames = batch_task.video.total_frames

    if PROGRESS_BAR:
        with tqdm(
                total=total_frames,
                desc="Analysing video",
                unit="frames",
                position=0,
                unit_scale=False,
                unit_divisor=1,
                ncols=130
        ) as progress_bar:
            while not stop_event.is_set():
                opengl_size = opengl_q.qsize()
                yolo_size = yolo_q.qsize()
                results_size = results_q.qsize()

                progress_bar.n = results_size
                progress_bar.set_postfix_str(
                    f"Queues: OpenGL: {opengl_size:>3}, YOLO: {yolo_size:>3}"
                )
                progress_bar.refresh()

                if results_size >= total_frames:
                    stop_event.set()

                time.sleep(UPDATE_PROGRESS_INTERVAL)
    else:
        while not stop_event.is_set():
            opengl_size = opengl_q.qsize()
            yolo_size = yolo_q.qsize()
            results_size = results_q.qsize()
            print(f"Queues: OpenGL: {opengl_size:>3}, YOLO: {yolo_size:>3}, DONE: {results_size:>3}")

            if results_size >= total_frames:
                stop_event.set()

            time.sleep(0.5)



def log_performance(batch_task, results_queue):
    tasks = [task for task in results_queue.queue if task is not None and hasattr(task, 'profile')]
    total_frames = len(tasks)

    total_pipeline_time = batch_task.end_time - batch_task.start_time
    video_duration = total_frames / batch_task.video.fps
    avg_processing_fps = total_frames / total_pipeline_time
    realtime_percentage = (avg_processing_fps / 60.0) * 100.0

    log_message = (
        f"\n{'-' * 60}"
        f"\n OBJECT DETECTION COMPLETED {'(sequential mode)' if SEQUENTIAL_MODE else '(parallel mode)'}\n"
        f"\n Video stats\n"
        f"  - Total Frames               : {total_frames}\n"
        f"  - Video Duration             : {video_duration:.2f} s\n"
    )

    if SEQUENTIAL_MODE:
        log_message += f"\n Sequential Queue statistics\n"
        for key, total_time in batch_task.profile.items():
            if key.endswith("_duration"):  # Only include duration metrics
                avg_time = total_time / total_frames if total_frames > 0 else 0.0
                stage_name = key.replace("_duration", "").capitalize()
                log_message += (
                    f"  - {stage_name:<27}: {avg_time * 1000:.0f} ms | "
                    f"{(1 / avg_time if avg_time > 0 else 0):.0f} fps\n"
                )
    else:
        log_message += (
            f"\n Performance stats\n"
            f"  - Average Processing         : {avg_processing_fps:.2f} fps\n"
            f"  - Real-time Processing       : {realtime_percentage:.2f} %\n"
            f"  - Total Pipeline Runtime (s) : {total_pipeline_time:.2f} s\n"
        )
        log_message += f"\n Task Average Times (while running in parallel)\n"

        aggregated_times = {}
        for task in tasks:
            for key, total_time in task.profile.items():
                if key.endswith("_duration"):
                    if key not in aggregated_times:
                        aggregated_times[key] = {"total_time": 0, "task_count": 0}
                    aggregated_times[key]["total_time"] += total_time
                    aggregated_times[key]["task_count"] += 1

        # Calculate and format averages for each key
        for key, data in aggregated_times.items():
            avg_time = data["total_time"] / total_frames if total_frames > 0 else 0.0
            stage_name = key.replace("_duration", "").replace("_", " ").capitalize()
            log_message += (
                f"  - {stage_name:<27}: {avg_time * 1000:.0f} ms | "
                f"{(1 / avg_time if avg_time > 0 else 0):.0f} fps\n"
            )

    log_message += f"{'-' * 60}\n"

    print(Fore.LIGHTBLUE_EX + log_message)
