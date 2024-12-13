import os
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_cohere import ChatCohere, create_csv_agent
from dotenv import load_dotenv
from io import StringIO

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
empty_csv_path = "/tmp/empty.csv"
empty_df.to_csv(empty_csv_path, index=False)  # Create an empty CSV for initialization
agent_executor = create_csv_agent(llm, empty_csv_path)  # Initialize with the empty CSV

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
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTMLResponse(content=html_template)


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    global agent_executor

    # Read the uploaded file into a pandas DataFrame
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    # Save the DataFrame to a temporary CSV file
    temp_file_path = "evaluation_results.csv"
    df.to_csv(temp_file_path, index=False)

    # Create the agent for the uploaded CSV
    agent_executor = create_csv_agent(llm, temp_file_path)

    return {"message": "CSV file uploaded and agent initialized successfully!"}


@app.post("/ask-question")
async def ask_question(question: str = Form(...)):
    global agent_executor
    if not agent_executor:
        return {"error": "No CSV file uploaded yet. Please upload a file first."}

    # Use the agent to process the question
    response = agent_executor.invoke({"input": question})
    return {"question": question, "answer": response.get("output")}


@app.get("/quit")
async def quit_app():
    global agent_executor
    agent_executor = None
    return {"message": "Agent reset. You can upload a new CSV file."}