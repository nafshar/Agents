import gradio as gr
import pandas as pd


# Function to load CSV and display the head for confirmation
def load_csv(file):
    if file is None:
        return "No file uploaded.", None

    # Read the entire CSV file
    df = pd.read_csv(file.name)

    # Display the first 5 rows for confirmation
    return df.head(), df  # First output for preview, second for full DataFrame


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("### 📊 Upload a CSV File to View and Process")

    file_input = gr.File(file_types=['.csv'], label="Upload CSV")

    # Outputs: One for preview and one for full DataFrame (hidden)
    preview_output = gr.Dataframe(label="Preview (First 5 Rows)")
    full_data_output = gr.Dataframe(visible=False)  # Hidden but exists in memory

    # Trigger load_csv on file upload
    file_input.change(load_csv, inputs=file_input, outputs=[preview_output, full_data_output])

demo.launch()
