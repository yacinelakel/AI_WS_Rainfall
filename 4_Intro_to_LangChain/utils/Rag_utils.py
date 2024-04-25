import fitz
import gradio as gr
from PIL import Image
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class PDFChatBot:
    def __init__(self, config_path="../config.yaml"):
        """
        Initialize the PDFChatBot instance.
        """
        self.prompt = ChatPromptTemplate.from_template(
            """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. If you get greeting questions respond accordingly.

            Question: {question} 

            Context: {context} 

            Answer:
            """
        )

        self.documents = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.embeddings = OpenAIEmbeddings()
        self.docsearch = None
        self.page = 0
        self.processed = False
        self.chat_history = []
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.rag_chain_from_docs = (
                RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                | self.prompt
                | self.llm
                | StrOutputParser()
        )


    def add_text(self, history, text):
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

    def load_vectordb(self):
        """
        Load the vector database from the documents and embeddings.
        """
        self.vectordb = Chroma.from_documents(self.splits, self.embeddings)

        self.retriever = self.vectordb.as_retriever()

        self.rag_chain_with_source = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | RunnablePassthrough().assign(answer=self.rag_chain_from_docs)
        )

    def process_file(self, file):
        """
        Process the uploaded PDF file and initialize necessary components: Tokenizer, VectorDB and LLM.

        Parameters:
            file (FileStorage): The uploaded PDF file.
        """
        self.documents = PyPDFLoader('resources/Noria_eBook_The_Insurance_Industry_2025_171205_v8.pdf').load()
        # Split the document into chunks
        self.splits = self.text_splitter.split_documents(self.documents)
        self.load_vectordb()
        self.processed = True

    def generate_response(self, history, query, file):
        """
        Generate a response based on user query and chat history.

        Parameters:
            history (list): List of chat history tuples.
            query (str): User's query.
            file (FileStorage): The uploaded PDF file.

        Returns:
            tuple: Updated chat history and a space.
        """
        if not query:
            raise gr.Error(message='Submit a question')
        if not file:
            raise gr.Error(message='Upload a PDF')
        if not self.processed:
            self.process_file(file)
            self.processed = True

        result = self.rag_chain_with_source.invoke(query)
        self.chat_history.append((query, result["answer"]))

        self.page = result['context'][0].metadata['page']

        for char in result['answer']:
            history[-1][-1] += char
        return history, " "

    def render_file(self, file):
        """
        Renders a specific page of a PDF file as an image.

        Parameters:
            file (FileStorage): The PDF file.

        Returns:
            PIL.Image.Image: The rendered page as an image.
        """
        doc = fitz.open('resources/Noria_eBook_The_Insurance_Industry_2025_171205_v8.pdf')
        page = doc[self.page]
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image



def rag_chat_bot():
    # Create PDFChatBot instance
    pdf_chatbot = PDFChatBot()

    with gr.Blocks(title="RAG Chatbot Q&A", ) as demo:
        with gr.Column():
            with gr.Row():
                chat_history = gr.Chatbot(value=[], elem_id='chatbot')
                show_img = gr.Image(label='Overview')

        with gr.Row():
            with gr.Column(scale=0.60):
                text_input = gr.Textbox(
                    show_label=False,
                    placeholder="Type here to ask your PDF",
                    container=False)

            with gr.Column(scale=0.20):
                submit_button = gr.Button('Send')

            with gr.Column(scale=0.20):
                uploaded_pdf = gr.Button("📁 load PDF")

        # Event handler for uploading a PDF
        uploaded_pdf.click(pdf_chatbot.render_file, inputs=[uploaded_pdf], outputs=[show_img])

        # Event handler for submitting text and generating response
        submit_button.click(pdf_chatbot.add_text, inputs=[chat_history, text_input], outputs=[chat_history],
                            queue=False). \
            success(pdf_chatbot.generate_response, inputs=[chat_history, text_input, uploaded_pdf],
                    outputs=[chat_history, text_input]). \
            success(pdf_chatbot.render_file, inputs=[uploaded_pdf], outputs=[show_img])

    return demo