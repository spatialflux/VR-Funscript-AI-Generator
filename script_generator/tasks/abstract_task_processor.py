import queue
import threading
from typing import Generator, Optional
from script_generator.tasks.tasks import AnalyseVideoTask, ProcessFrameTask


class AbstractTaskProcessor(threading.Thread):
    process_type = ""

    def __init__(self, batch_task: AnalyseVideoTask, output_queue: queue.Queue, input_queue: Optional[queue.Queue] = None):
        """
        Abstract thread class to handle lifecycle management and task handling boilerplate.

        :param input_queue: Queue to consume tasks from.
        :param output_queue: Queue to produce processed tasks.
        """
        super().__init__()
        self.batch_task = batch_task
        self.input_queue = input_queue
        self.output_queue = output_queue
        self._stop_event = threading.Event()

    def log(self, message):
        """
        Unified logging for the thread.
        :param message: Message to log.
        """
        thread_name = threading.current_thread().name
        print(f"[{self.__class__.__name__}-{thread_name}] {message}")

    def get_task(self) -> Generator[ProcessFrameTask, None, None]:
        """
        Generator for retrieving tasks from the input queue.
        Yields tasks until a sentinel (None) is encountered or the thread is stopped.
        Logs and tracks the time taken to retrieve tasks.
        """
        if self.input_queue is None:
            raise ValueError("Input queue is None. An input queue must be provided to use get_task().")

        while not self._stop_event.is_set():
            try:
                task = self.input_queue.get(timeout=1)

                if task is None:  # Sentinel for termination
                    self.batch_task.end(self.process_type)
                    self.on_last_item()
                    self.finish_task(None)
                    break
                yield task
            except queue.Empty:
                continue

    def finish_task(self, task):
        """
        Finalizes the task by placing it in the output queue.
        If the queue is full, waits until a spot is available.

        :param task: The task to place in the output queue.
        """
        while not self._stop_event.is_set():
            try:
                self.output_queue.put(task, timeout=1)
                break
            except queue.Full:
                return

    def run(self):
        """
        Main thread entry point. Executes the `task_logic` method.
        """
        try:
            self.batch_task.start(self.process_type)
            self.task_logic()
        except Exception as e:
            print(f"An error occurred during task execution on thread {self.process_type}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            return

    def task_logic(self):
        """
        Abstract method for setup, task processing, and cleanup.
        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement task_logic")

    def stop_process(self):
        self.batch_task.end(self.process_type)
        self.on_last_item()
        # Propagate sentinel to the output queue
        self.output_queue.put(None)

    def on_last_item(self):
        return

from enum import Enum

class TaskProcessorTypes(Enum):
    VIDEO = "Video processing"
    OPENGL = "3d to 2D"
    YOLO = "YOLO inference"
    YOLO_ANALYSIS = "YOLO analysis"

    def __str__(self):
        return self.value