import os
import re
import tempfile
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from io import StringIO
from langchain_cohere import ChatCohere, create_csv_agent

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
os.environ['COHERE_API_KEY'] = COHERE_API_KEY

# Initialize the Cohere LLM
llm = ChatCohere(cohere_api_key=COHERE_API_KEY,
                 model="command-r-plus-08-2024",
                 temperature=0)

# Placeholder for agent, generated image, and dataframe
agent_executor = None
generated_image_path = None
uploaded_df = None

# Function to handle CSV upload
def upload_csv(file):
    global agent_executor, uploaded_df

    try:
        uploaded_df = pd.read_csv(file.name)
        temp_csv_path = os.path.join(tempfile.gettempdir(), "temp.csv")
        uploaded_df.to_csv(temp_csv_path, index=False)

        # Create the CSV agent
        agent_executor = create_csv_agent(llm, temp_csv_path)
        return "✅ CSV uploaded successfully!", uploaded_df.head(), None
    except Exception as e:
        return f"❌ Error uploading CSV: {e}", None, None

# Function to handle user questions
def ask_question(question):
    global agent_executor, generated_image_path

    if not agent_executor:
        return "❗ Please upload a CSV file first.", None

    try:
        response = agent_executor.invoke({"input": question})
        response_message = response.get("output")

        # Extract image path if a chart was generated
        image_match = re.search(r'\("(?P<filename>[^\"]+\.png)"\)', response_message)
        generated_image_path = image_match.group("filename") if image_match else None

        return response_message, generated_image_path
    except Exception as e:
        return f"⚠️ Failed to process the question: {e}", None

# Function to reset the agent
def reset_agent():
    global agent_executor, generated_image_path, uploaded_df
    agent_executor = None
    generated_image_path = None
    uploaded_df = None
    return "🔄 Agent reset. You can upload a new CSV.", None, None

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 📊 CSV Agent with Cohere AI")

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
    reset_button.click(reset_agent, outputs=[upload_status, df_head_output, image_output])

# Launch the Gradio App
demo.launch()