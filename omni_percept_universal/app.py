import gradio as gr
from pipeline_runner import execute_pipeline

with gr.Blocks(theme=gr.themes.Soft(), title="Omni-Percept Universal Pipeline") as app:
    gr.Markdown("# 👁️ Omni-Percept Universal Pipeline")
    gr.Markdown("Defense-grade, memory-safe, neuro-symbolic Vision Proof-of-Concept.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Upload Image")
            critical_question = gr.Textbox(
                label="Critical Question", 
                placeholder="e.g., Is the truck facing the tree?"
            )
            run_btn = gr.Button("Run Pipeline", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("1. X-Ray Vision"):
                    xray_output = gr.Image(type="filepath", label="Annotated Image (Grounding & Masks)", interactive=False)
                with gr.TabItem("2. Z3 Prover Terminal"):
                    z3_output = gr.Textbox(label="Z3 Mathematical Logic", lines=15, max_lines=20, interactive=False)
                with gr.TabItem("3. RAG Context"):
                    rag_output = gr.Textbox(label="Retrieved Spatial Context", lines=10, interactive=False)
                with gr.TabItem("4. Final Intelligence Report"):
                    report_output = gr.Markdown(value="*Waiting for pipeline execution...*")
                    
    # Wire the pipeline sequentially updating the UI
    run_btn.click(
        fn=execute_pipeline,
        inputs=[image_input, critical_question],
        outputs=[xray_output, z3_output, rag_output, report_output]
    )

if __name__ == "__main__":
    import os
    output_dir = os.path.join(os.path.dirname(__file__), "data", "temp")
    app.launch(share=False, allowed_paths=[output_dir])
