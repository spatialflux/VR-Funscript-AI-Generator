from script_generator.tasks.abstract_task_processor import AbstractTaskProcessor, TaskProcessorTypes

LiveDisplayMode = False

# TODO Implement all tracking logic here
class YoloAnalysisTaskProcessor(AbstractTaskProcessor):
    process_type = TaskProcessorTypes.YOLO_ANALYSIS

    def task_logic(self):

        records = []

        for task in self.get_task():
            detections = task.detections
            frame_pos = task.frame
            if detections[0].boxes.id is None:  # Skip if no tracks are found
                continue

            if len(detections[0].boxes) == 0 and not LiveDisplayMode:  # Skip if no boxes are detected
                continue

            ### DETECTION of BODY PARTS
            # Extract track IDs, boxes, classes, and confidence scores
            track_ids = detections[0].boxes.id.cpu().tolist()
            boxes = detections[0].boxes.xywh.cpu()
            classes = detections[0].boxes.cls.cpu().tolist()
            confs = detections[0].boxes.conf.cpu().tolist()

            # Process each detection
            for track_id, cls, conf, box in zip(track_ids, classes, confs, boxes):
                track_id = int(track_id)
                x, y, w, h = box.int().tolist()
                x1 = x - w // 2
                y1 = y - h // 2
                x2 = x + w // 2
                y2 = y + h // 2
                # Create a detection record
                record = [frame_pos, int(cls), round(conf, 1), x1, y1, x2, y2, track_id]
                records.append(record)

        # write_dataset(os.path.join(DEBUG_PATH, "rawyolo.json"), records)