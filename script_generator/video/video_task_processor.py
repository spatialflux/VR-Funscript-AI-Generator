import av

from script_generator.config import TEXTURE_RESOLUTION
from script_generator.tasks.abstract_task_processor import AbstractTaskProcessor, TaskProcessorTypes
from script_generator.tasks.tasks import ProcessFrameTask


class VideoTaskProcessor(AbstractTaskProcessor):
    process_type = TaskProcessorTypes.VIDEO

    def task_logic(self):
        container = av.open(self.batch_task.video.path)  #
        video_stream = container.streams.video[0]
        video_stream.thread_type = "AUTO"

        frame_count = 0

        for frame in container.decode(video_stream):
            task = ProcessFrameTask(frame=frame_count)

            # Preprocess
            task.start(str(self.process_type))
            image = frame.reformat(width=TEXTURE_RESOLUTION * 2, height=TEXTURE_RESOLUTION).to_ndarray(format="rgb24")
            h, w, _ = image.shape
            image = image[:, :w // 2, :]
            task.preprocessed_frame = image
            task.end(str(self.process_type))

            self.finish_task(task)  # May block if queue is full

            frame_count += 1

        self.stop_process()