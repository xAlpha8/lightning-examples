import os
import shutil
import tempfile

import gradio as gr
import cv2
import supervision as sv

from inference.models.yolo_world.yolo_world import YOLOWorld

model = YOLOWorld(model_id="yolo_world/l")
BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=2)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    text_thickness=2, text_scale=1, text_color=sv.Color.BLACK
)
cur_dir = os.path.dirname(os.path.abspath(__file__))


def save_video_to_local(video_path):
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "temp",
        next(tempfile._get_candidate_names()) + ".mp4",
    )
    shutil.copyfile(video_path, filename)
    return filename


def image_generator(video, textbox_in):
    video = video if video else "none"
    filename = save_video_to_local(video)
    generator = sv.get_video_frames_generator(filename)
    model.set_classes(textbox_in)
    for frame in generator:
        results = model.infer(frame, confidence=0.002)
        detections = sv.Detections.from_inference(results).with_nms(threshold=0.1)
        annotated_image = frame.copy()
        annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
        annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
        yield annotated_image


video = gr.Video(label="Input Video", sources=["upload"])
textbox = gr.Textbox(label="Enter text and press ENTER")

demo = gr.Interface(
    fn=image_generator,
    inputs=[video, textbox],
    outputs=[gr.Image(label="Output Video")],
    title="Prompt-based Object Detection Demo",
    examples=[[f"{cur_dir}/examples/sample_1.mp4", "Detect all boats"]],
)


demo.launch(share=True)
