import re
import fitz
import torch
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gradio as gr
import os

from utils import load_config, add_text, render_file

class PDFChatBot:
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the PDFChatBot instance.

        Parameters:
            config_path (str): Path to the configuration file (default is "config.yaml").
        """
        self.processed = False
        self.page = 0
        self.chat_history = []
        self.config = load_config(config_path)
        self.prompt = None
        self.documents = None
        self.embeddings = None
        self.vectordb = None
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.chain = None

    def create_prompt_template(self):
        """
        Create a prompt template for the chatbot.
        """
        template = (
            "You are an assistant that provides detailed answers based on the given context. "
            "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
            "{context}\n\n"
            "Question: {question}\n"
            "Helpful Answer:"
        )
        self.prompt = PromptTemplate.from_template(template)

    def load_embeddings(self):
        """
        Load embeddings from Hugging Face and set in the config file.
        """
        model_name = self.config.get("modelEmbeddings")
        if not model_name:
            raise ValueError("modelEmbeddings not specified in the config file.")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def load_vectordb(self):
        """
        Load the vector database from the documents and embeddings.
        """
        self.vectordb = Chroma.from_documents(self.documents, self.embeddings)

    import os

    def load_tokenizer(self):
        tokenizer_path = self.config.get("autoTokenizer")
        if not tokenizer_path:
            raise ValueError("autoTokenizer not specified in the config file.")
        
        if not os.path.isdir(tokenizer_path):
            raise ValueError(f"Tokenization directory '{tokenizer_path}' does not exist or is not a directory.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    def load_model(self):
        local_model_path = self.config.get("localModelPath")
        if not local_model_path:
            raise ValueError("localModelPath not specified in the config file.")
        
        if not os.path.isdir(local_model_path):
            raise ValueError(f"Model directory '{local_model_path}' does not exist or is not a directory.")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            device_map='auto',
            torch_dtype=torch.float32,
            load_in_8bit=False
        )


    def create_pipeline(self):
        """
        Create a pipeline for text generation using the loaded model and tokenizer.
        """
        pipe = pipeline(
            model=self.model,
            task='text-generation',
            tokenizer=self.tokenizer,
            max_new_tokens=200
        )
        self.pipeline = HuggingFacePipeline(pipeline=pipe)

    def create_chain(self):
        """
        Create a Conversational Retrieval Chain
        """
        self.chain = ConversationalRetrievalChain.from_llm(
            self.pipeline,
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(search_kwargs={"k": 1}),
            condense_question_prompt=self.prompt,
            return_source_documents=True
        )

    def process_file(self, file):
        """
        Process the uploaded PDF file and initialize necessary components: Tokenizer, VectorDB, and LLM.

        Parameters:
            file (FileStorage): The uploaded PDF file.
        """
        if hasattr(file, 'name'):  # Check if the file has a 'name' attribute
            file_path = file.name  # Get the path of the uploaded file
        else:
            raise ValueError("Invalid file object. Cannot process the file.")

        self.create_prompt_template()
        self.documents = PyPDFLoader(file_path).load()  # Load the PDF from the file path
        self.load_embeddings()
        self.load_vectordb()
        self.load_tokenizer()
        self.load_model()
        self.create_pipeline()
        self.create_chain()


    def generate_response(self, history, query, file):
        """
        Generate a response based on user query and chat history.

        Parameters:
            history (list): List of chat history tuples.
            query (str): User's query.
            file (FileStorage): The uploaded PDF file.

        Returns:
            string: Helpful Answer to the asked user query.
        """
        if not query:
            raise gr.Error(message='Submit a question')
        if not file:
            raise gr.Error(message='Upload a PDF')
        if not self.processed:
            self.process_file(file)
            self.processed = True

        result = self.chain.invoke({"question": query, 'chat_history': self.chat_history})
        self.chat_history.append((query, result["answer"]))
        self.page = list(result['source_documents'][0])[1][1]['page']

        if history:
            history.append((query, result["answer"]))
        else:
            history = [(query, result["answer"])]

        return history, " "


    
    def render_file(self, file):
        """
        Renders a specific page of a PDF file as an image.

        Parameters:
            file (FileStorage): The PDF file.

        Returns:
            PIL.Image.Image: The rendered page as an image.
        """
        if hasattr(file, 'name'):  # Check if the file has a 'name' attribute
            file_path = file.name  # Get the path of the uploaded file
        else:
            raise ValueError("Invalid file object. Cannot render the file.")

        doc = fitz.open(file_path)
        page = doc.load_page(self.page - 1)  # Pages are zero-indexed
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image
