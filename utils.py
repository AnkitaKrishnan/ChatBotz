import yaml
import fitz
import gradio as gr
from PIL import Image

def load_config(file_path):
    """
    Load configuration from a YAML file.

    Parameters:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration as a dictionary.
    """
    try:
        with open(file_path, 'r') as stream:
            config = yaml.safe_load(stream)
            if not config:
                raise ValueError("Empty configuration file.")
            return config
    except (yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"Error loading configuration: {exc}")

def add_text(history, text):
    """
    Add user-entered text to the chat history.

    Parameters:
        history (list): List of chat history tuples.
        text (str): User-entered text.

    Returns:
        list: Updated chat history.
    """
    if not text:
        raise gr.Error('Enter text')
    history.append((text, ''))
    return history

def render_file(file, page):
    """
    Renders a specific page of a PDF file as an image.

    Parameters:
        file (FileStorage): The PDF file.
        page (int): Page number to render.

    Returns:
        PIL.Image.Image: The rendered page as an image.
    """
    doc = fitz.open(stream=file.read(), filetype="pdf")
    page = doc.load_page(page - 1)  # Pages are zero-indexed
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image
