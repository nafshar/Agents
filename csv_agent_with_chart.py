import os
import re
import tempfile
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from PIL import Image
from langchain_cohere import ChatCohere, create_csv_agent

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
os.environ['COHERE_API_KEY'] = COHERE_API_KEY

# Initialize the Cohere LLM
llm = ChatCohere(cohere_api_key=COHERE_API_KEY,
                 model="command-r-plus-08-2024",
                 temperature=0)

# Placeholders
agent_executor = None
uploaded_df = None

# Upload CSV
def upload_csv(file):
    global agent_executor, uploaded_df
    try:
        uploaded_df = pd.read_csv(file.name)
        temp_csv_path = os.path.join(tempfile.gettempdir(), "temp.csv")
        uploaded_df.to_csv(temp_csv_path, index=False)
        agent_executor = create_csv_agent(llm, temp_csv_path)
        return "✅ CSV uploaded successfully!", uploaded_df.head(), gr.update(visible=False)
    except Exception as e:
        return f"❌ Error uploading CSV: {e}", None, gr.update(visible=False)

# Handle User Questions
def ask_question(question):
    global agent_executor

    if not agent_executor:
        return "❗ Please upload a CSV file first.", gr.update(visible=False)

    try:
        # Agent Response
        response = agent_executor.invoke({"input": question})
        response_message = response.get("output")

        # Detect Markdown-style image reference
        image_match = re.search(r'!\[.*?\]\("(?P<filename>[^"]+\.png)"\)', response_message)
        if image_match:
            image_path = image_match.group("filename")

            # Remove the Markdown image reference from the response
            response_message = re.sub(r'!\[.*?\]\("[^"]+\.png"\)', '', response_message).strip()

            # Check if the image exists and load it
            if os.path.exists(image_path):
                return response_message, gr.update(value=image_path, visible=True)

        return response_message, gr.update(visible=False)

    except Exception as e:
        return f"⚠️ Failed to process the question: {e}", gr.update(visible=False)

# Reset Agent
def reset_agent():
    global agent_executor, uploaded_df
    agent_executor = None
    uploaded_df = None
    return gr.update(value="🔄 Agent reset. You can upload a new CSV."), None, gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

# Gradio Interface
with gr.Blocks(css="""
    .gr-input, .gr-output, textarea, .gr-dataframe, .gr-image {
        background-color: #e6f7ff !important;
        border: 2px solid #007acc !important;
        padding: 10px !important;
        border-radius: 8px !important;
    }
    label, .gr-box label {
        color: #003366 !important;
        font-weight: bold !important;
        font-size: 14px !important;
    }
""") as demo:
    gr.Markdown("# 📊 CSV Agent with Cohere LLM \n ### Nader Afshar")

    # File Upload Section
    with gr.Row():
        file_input = gr.File(label="Upload CSV", file_types=['.csv'])
        upload_button = gr.Button("Upload")

    upload_status = gr.Textbox(label="Upload Status", interactive=False)
    df_head_output = gr.Dataframe(label="CSV Preview (Head)", interactive=False)

    # Question Section
    question_input = gr.Textbox(label="Ask a Question", placeholder="Type your question here...")
    submit_button = gr.Button("Submit Question")

    # Output Section
    response_output = gr.Textbox(label="Response", interactive=False)
    image_output = gr.Image(label="Generated Chart", visible=False)

    # Reset Button
    reset_button = gr.Button("Reset Agent")

    # Event Handlers
    upload_button.click(upload_csv, inputs=file_input, outputs=[upload_status, df_head_output, image_output])
    submit_button.click(ask_question, inputs=question_input, outputs=[response_output, image_output])
    reset_button.click(reset_agent, outputs=[upload_status, df_head_output, image_output, file_input, upload_button])

# Launch the Gradio App
demo.launch()
