import time

from script_generator.config import YOLO_CONF, YOLO_BATCH_SIZE, YOLO_MODEL
from script_generator.tasks.abstract_task_processor import AbstractTaskProcessor, TaskProcessorTypes

class YoloTaskProcessor(AbstractTaskProcessor):
    process_type = TaskProcessorTypes.YOLO

    def task_logic(self):
        batch = []
        tasks = []

        for task in self.get_task():
            if task.rendered_frame is not None:
                batch.append(task.rendered_frame)
                tasks.append(task)

                # If batch is ready, process it
                if len(batch) >= YOLO_BATCH_SIZE:
                    self.process_batch(batch, tasks)
                    batch = []
                    tasks = []

        # Process any remaining tasks in the batch
        if batch:
            self.process_batch(batch, tasks)

    def process_batch(self, batch, tasks):
        start_time = time.time()
        detections = yolo_inference_batch(batch, conf_threshold=YOLO_CONF)
        avg_time = (time.time() - start_time) / len(tasks)

        for t, det in zip(tasks, detections):
            t.detections = det
            t.duration(str(self.process_type), avg_time)
            self.finish_task(t)

    def on_last_item(self):
        #TODO this should be after every stage in the batch task is complete for now it ends after Yolo inference but later after screen analysis
        self.batch_task.end_time = time.time()


def yolo_inference_batch(frames: list, conf_threshold=0.3):
    """
    Run YOLO inference on a batch of images. Returns a list of detections for each image.
    """
    results = YOLO_MODEL(frames, conf=conf_threshold, verbose=False)

    batch_detections = []
    for frame_results in results:
        out_detections = []
        for detection in frame_results.boxes:
            box = detection.xyxy[0].cpu().numpy().astype(int)
            cls = int(detection.cls[0])
            conf = float(detection.conf[0])
            out_detections.append((cls, conf, box))
        batch_detections.append(out_detections)

    return batch_detections