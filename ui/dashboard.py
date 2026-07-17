"""
dashboard.py

Builds the main PerceptionLab dashboard.
"""

import gradio as gr
from services.pipeline import run_pipeline
from config import APP_TITLE, APP_SUBTITLE


def build_dashboard():

    with gr.Blocks(title="PerceptionLab") as demo:

        gr.Markdown(
            f"""
# 🔬 {APP_TITLE}

### {APP_SUBTITLE}
"""
        )

        with gr.Row():

            # ---------------- Sidebar ----------------
            with gr.Column(scale=1):

                gr.Markdown("## Pipeline")

                detection = gr.Checkbox(value=True, label="Detection")
                segmentation = gr.Checkbox(value=True, label="Segmentation")
                depth = gr.Checkbox(value=True, label="Depth")
                caption = gr.Checkbox(value=True, label="Caption")
                scene = gr.Checkbox(value=True, label="Scene Graph")
                vqa = gr.Checkbox(value=True, label="Visual QA")

                run = gr.Button(
                    "Run Pipeline",
                    variant="primary"
                )

            # ---------------- Images ----------------
            with gr.Column(scale=2):

                input_image = gr.Image(
                    type="numpy",
                    label="Input Image"
                )

            with gr.Column(scale=2):

                output_image = gr.Image(
                    label="Output Preview"
                )

        with gr.Tabs():

            with gr.Tab("Detection"):
                detection_box = gr.Textbox(lines=8)

            with gr.Tab("Segmentation"):
                segmentation_box = gr.Textbox(lines=8)

            with gr.Tab("Depth"):
                depth_box = gr.Textbox(lines=8)

            with gr.Tab("Scene Graph"):
                scene_box = gr.Textbox(lines=8)

            with gr.Tab("Caption"):
                caption_box = gr.Textbox(lines=8)

            with gr.Tab("Visual QA"):
                vqa_box = gr.Textbox(lines=8)

            with gr.Tab("Benchmark"):
                benchmark_box = gr.Textbox(lines=8)

        run.click(
            fn=run_pipeline,
            inputs=input_image,
            outputs=[
                output_image,
                detection_box,
                segmentation_box,
                depth_box,
                scene_box,
                caption_box,
                vqa_box,
                benchmark_box,
            ],
        )

    return demo