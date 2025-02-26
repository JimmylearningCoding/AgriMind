import gradio as gr
from models.vision_model import VisionModel

class VisionInterface:
    def __init__(self):
        self.vision_model = VisionModel()

    def qvq_interface(self, question, image_path):
        return self.vision_model.query(question, image_path)