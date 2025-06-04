import os
import re
import tempfile
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
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
empty_df = pd.DataFrame()
temp_dir = tempfile.gettempdir()  # Get a temporary directory for saving files
temp_csv_path = os.path.join(temp_dir, "temp.csv")
empty_df.to_csv(temp_csv_path, index=False)  # Create an empty CSV for initialization
agent_executor = None  # Placeholder for agent
generated_image_path = None  # Placeholder for generated chart image path

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
        <input type="text" id="question" name="question" style="width: 400px;"><br><br>
        <button type="submit">Submit</button>
    </form>
    {response_message}
    {image_display}
    <br>
    <form action="/quit" method="get">
        <button type="submit">Reset Agent</button>
    </form>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def read_root(success: bool = False, response_message: str = None, image: str = None):
    success_message = """
    <p style="color: green;">CSV file uploaded successfully! You can now ask questions.</p>
    """ if success else ""

    # Render the response message if provided
    response_message_html = f"<p><strong>Response:</strong> {response_message}</p>" if response_message else ""

    # Display image if available
    image_display_html = f"""<br><p><strong>Generated Chart:</strong></p>
                             <img src="/images/{image}" alt="Chart Image" style="max-width:500px;"/>""" if image else ""

    html_content = html_template.format(success_message=success_message,
                                        response_message=response_message_html,
                                        image_display=image_display_html)
    return HTMLResponse(content=html_content)

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    global agent_executor

    try:
        # Read the uploaded file into a pandas DataFrame
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))

        # Save the DataFrame to a temporary CSV file
        df.to_csv(temp_csv_path, index=False)

        # Create the agent for the uploaded CSV
        agent_executor = create_csv_agent(llm, temp_csv_path)

        # Redirect back to the main page with success status
        return RedirectResponse(url="/?success=true", status_code=303)

    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

@app.post("/ask-question")
async def ask_question(question: str = Form(...)):
    global agent_executor, generated_image_path
    if not agent_executor:
        return {"error": "No CSV file uploaded yet. Please upload a file first."}

    try:
        # Use the agent to process the question
        response = agent_executor.invoke({"input": question})
        response_message = response.get("output")

        # Extract the image file name (if a chart was created)
        image_match = re.search(r'\("(?P<filename>[^"]+\.png)"\)', response_message)
        if image_match:
            generated_image_path = image_match.group("filename")

        # Redirect back to the homepage with the response
        return RedirectResponse(
            url=f"/?response_message={response_message}&image={generated_image_path}",
            status_code=303
        )

    except Exception as e:
        return {"error": f"Failed to process the question. Error: {e}"}

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    """Serve the generated image file."""
    image_file_path = os.path.join(os.getcwd(), image_name)
    if os.path.exists(image_file_path):
        return FileResponse(image_file_path)
    return {"error": "Image not found."}

@app.get("/quit")
async def quit_app():
    global agent_executor, generated_image_path
    agent_executor = None
    generated_image_path = None
    return RedirectResponse(url="/", status_code=303)
