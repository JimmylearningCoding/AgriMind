# app.py
import gradio as gr
from inference import InferenceEngine

class GradioApp:
    def __init__(self):
        self.inference = InferenceEngine()

    def launch(self):
        with gr.Blocks(
            css="""
            .header { text-align: center; font-size: 2.5rem; color: #333; margin-bottom: 20px; }
            .button { background-color: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 5px; font-size: 1.2rem; transition: background-color 0.3s ease; }
            .button:hover { background-color: #45a049; }
            .output { background-color: #fff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); font-size: 1.1rem; margin-top: 20px; color: #333; }
            .footer { text-align: center; color: #777; margin-top: 30px; font-size: 0.9rem; }
            }
        """) as demo:
            gr.Markdown("<div class = 'header'> 🌱 AgriMind - 智能农业平台")

            with gr.Tabs():
                with gr.TabItem("🤖 直接聊天"):
                    chat_input = gr.Textbox(label="输入你的问题", placeholder="请输入你的问题", lines=3)
                    chat_button = gr.Button("生成答案",elem_classes="button")
                    chat_output = gr.Textbox(label="模型回答", lines=5, interactive=False)
                    chat_button.click(self.inference.chat, inputs=chat_input, outputs=chat_output)

                with gr.TabItem("📄 文档问答"):
                    doc_input = gr.Textbox(label="输入你的问题", placeholder="请输入你的问题", lines=3)
                    doc_button = gr.Button("生成答案", elem_classes="button")
                    doc_output = gr.Textbox(label="模型回答", lines=5, interactive=False)
                    doc_button.click(self.inference.qa, inputs=doc_input, outputs=doc_output)

                with gr.TabItem("🖼️ 图像问答"):
                    image_input = gr.Image(label="上传图片", type="filepath")
                    image_q_input = gr.Textbox(label="输入问题", lines=3)
                    image_q_button = gr.Button("生成回答", elem_classes="button")
                    image_q_output = gr.Textbox(label="模型回答", lines=5, interactive=False)
                    image_q_button.click(self.inference.image_question_answer, inputs=[image_q_input, image_input], outputs=image_q_output)

            gr.Markdown("<div class='footer'>由 Deepseek Langchain Gradio 提供支持</div>")

        demo.launch(server_name="0.0.0.0", server_port=5000, share=True)

if __name__ == "__main__":
    app = GradioApp()
    app.launch()
