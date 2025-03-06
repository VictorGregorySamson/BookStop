import gradio as gr
import base64
import requests
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from src.style.styles import CSS_STYLE  # Import the CSS styles
from src.core.chatbot import chatbot  # Import chatbot instance

# Cloudstaff branding
CLOUDSTAFF_URL = "https://www.cloudstaff.com"
LOGO_PATH = os.path.join("src", "assets", "logo.png")

# Function to encode the logo image
def encode_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found.")
        return None

LOGO_BASE64 = encode_image(LOGO_PATH)

session_id = "user1_session"

# Function to process chatbot responses
def respond(message, chat_history):
    # Call book_task_pipeline with the required parameters:
    response = requests.post(
                "http://localhost:8000/chat",
                json={
                    'query': message,
                    'session_id': session_id,
                    "model": "gemini-2.0-flash",
                    "temperature": 0.7,
                    "top_p": 0.9
                })
    
    if response.status_code == 200:
        return response.json()['response']
    #return response["response"]

# Function to create and return the Gradio UI
def create_gradio_ui():
    with gr.Blocks(title="Cloudstaff Library Assistant", theme=gr.themes.Base(primary_hue="blue")) as demo:
        gr.HTML(CSS_STYLE)  # Apply the CSS styling

        # Header
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"""
                <div id="header">
                    <h1>Cloudstaff Library Assistant 📚</h1>
                    <p>Your AI-powered library assistant for book recommendations, availability checks, discussions, and more!</p>
                    <div id="logo-container">
                        <a href="{CLOUDSTAFF_URL}" target="_blank">
                            <img src="data:image/png;base64,{LOGO_BASE64}" id="logo" alt="Cloudstaff Logo">
                        </a>
                    </div>
                </div>
                """)

        # Chatbox
        with gr.Group(elem_id="chatbox"):
            chatbot_ui = gr.ChatInterface(
                fn=respond,
                examples=[
                    "What books are available?",
                    "Can you recommend a book?",
                    "Recommend me business and management books",
                    "Let's talk about Harry Potter",
                    "Is 'Atomic Habits' available?",
                    "I like One Piece",
                    "Do you have romance books?",
                    "How to return a book"
                ],
                cache_examples=False,
                type="messages",
                fill_width=True,
            )

        # Footer
        gr.HTML("""
        <div id="footer">
            <p>📚 Powered by Cloudstaff | <a href="https://www.libib.com/u/cloudstaff" target="_blank">Visit Library</a></p>
        </div>
        """)

    return demo

# Launch Gradio UI
if __name__ == "__main__":
    ui = create_gradio_ui()
    ui.launch()