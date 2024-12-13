import os
import tempfile
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from langchain_cohere import ChatCohere, create_csv_agent
from dotenv import load_dotenv
from io import StringIO
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
os.environ['COHERE_API_KEY'] = COHERE_API_KEY

# Initialize the Cohere LLM
llm = ChatCohere(cohere_api_key=COHERE_API_KEY,
                 model="command-r-plus-08-2024",
                 temperature=0)

# Placeholder empty CSV for initial agent_executor setup
empty_df = pd.DataFrame()  # Placeholder empty DataFrame

# Get a temporary directory that works on all platforms
temp_dir = tempfile.gettempdir()
temp_csv_path = os.path.join(temp_dir, "temp.csv")
empty_df.to_csv(temp_csv_path, index=False)  # Create an empty CSV for initialization
agent_executor = None

# Store question-answer history
history = []

# Initialize FastAPI app
app = FastAPI()

# Define the HTML template for the web interface
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>CSV Agent</title>
</head>
<body>
    <h1>CSV Agent</h1>
    {success_message}
    <form action="/upload-csv" enctype="multipart/form-data" method="post">
        <label for="file">Upload CSV:</label><br>
        <input type="file" id="file" name="file"><br><br>
        <button type="submit">Upload</button>
    </form>
    <br>
    <form action="/ask-question" method="post">
        <label for="question">Ask a Question:</label><br>
        <input type="text" id="question" name="question"><br><br>
        <button type="submit">Submit</button>
    </form>
    <br>
    <form action="/quit" method="get">
        <button type="submit">Quit</button>
    </form>
    <br>
    {response_message}
    <br><br>
    <h3>Previous Questions and Answers:</h3>
    <ul>
    {history}
    </ul>
    <br>
    {chart}
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def read_root(success: bool = False, response_message: str = None, chart: str = None):
    success_message = """
    <p style="color: green;">CSV file uploaded successfully! You can now ask questions.</p>
    """ if success else ""

    # Render the response message if provided
    if response_message:
        response_message_html = f"<p><strong>Response:</strong> {response_message}</p>"
    else:
        response_message_html = ""

    # Render the question-answer history
    history_html = "".join([f"<li><strong>{q}</strong> - {a}</li>" for q, a in history])

    # Render chart if available
    chart_html = f'<img src="data:image/png;base64,{chart}" alt="Chart">' if chart else ""

    html_content = html_template.format(
        success_message=success_message,
        response_message=response_message_html,
        history=history_html,
        chart=chart_html
    )
    return HTMLResponse(content=html_content)


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    global agent_executor

    try:
        # Read the uploaded file into a pandas DataFrame
        contents = await file.read()
        try:
            df = pd.read_csv(StringIO(contents.decode('utf-8')))
        except Exception as e:
            return {"error": f"Failed to read the CSV file. Error: {e}"}

        # Save the DataFrame to a temporary CSV file
        try:
            df.to_csv(temp_csv_path, index=False)
        except Exception as e:
            return {"error": f"Failed to save the CSV file. Error: {e}"}

        # Create the agent for the uploaded CSV
        try:
            agent_executor = create_csv_agent(llm, temp_csv_path)
        except Exception as e:
            return {"error": f"Failed to initialize the agent. Error: {e}"}

        # Redirect back to the main page with success status
        return RedirectResponse(url="/?success=true", status_code=303)

    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}


@app.post("/ask-question")
async def ask_question(question: str = Form(...)):
    global agent_executor
    if not agent_executor:
        return {"error": "No CSV file uploaded yet. Please upload a file first."}

    # Use the agent to process the question
    response = agent_executor.invoke({"input": question})
    response_message = response.get("output")

    # Save the question and answer to history
    history.append((question, response_message))

    # Generate a chart only if the question is related to chart generation
    chart_base64 = None
    if "chart" in question.lower():  # Simple check for a chart-related question
        chart_base64 = generate_chart()

    # Redirect back to the homepage with the response and chart (if any)
    return RedirectResponse(
        url=f"/?response_message={response_message}&chart={chart_base64}" if chart_base64 else f"/?response_message={response_message}",
        status_code=303)


@app.get("/quit")
async def quit_app():
    global agent_executor
    agent_executor = None
    return {"message": "Agent reset. You can upload a new CSV file."}


def generate_chart():
    # Example function to generate a simple chart and return it as a base64 string
    plt.figure(figsize=(6, 4))
    data = [10, 20, 30, 40]
    labels = ['A', 'B', 'C', 'D']
    plt.bar(labels, data)
    plt.title('Example Chart')

    # Save the plot to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Convert the buffer to a base64 string
    chart_base64 = base64.b64encode(buf.read()).decode('utf-8')

    # Close the plot to free memory
    plt.close()

    return chart_base64
