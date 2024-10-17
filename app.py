import gradio as gr
from chatbot import PDFChatBot

bot = PDFChatBot()

def update_file(file):
    bot.process_file(file)
    return "File processed successfully"

def respond(history, query, file):
    if history is None:
        history = []
    history, _ = bot.generate_response(history, query, file)
    image = bot.render_file(file)
    return history, image

# Define the layout
with gr.Blocks() as demo:
    gr.Markdown("# 📝 PDF ChatBot")
    gr.Markdown("Upload a PDF, ask questions about its content, and get answers along with the specific page rendered.")
    
    with gr.Row():
        with gr.Column(scale=2):
            file_upload = gr.File(label="Upload a PDF")
            chatbot = gr.Chatbot()
            textbox = gr.Textbox(placeholder="Enter your query and press enter")
            submit_button = gr.Button("Submit")
        
        with gr.Column(scale=1):
            pdf_page = gr.Image(label="PDF Page")
    
    # Set up the Gradio interface with interactions
    submit_button.click(respond, inputs=[chatbot, textbox, file_upload], outputs=[chatbot, pdf_page])
    textbox.submit(respond, inputs=[chatbot, textbox, file_upload], outputs=[chatbot, pdf_page])

demo.launch(share=True)
