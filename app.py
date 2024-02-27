import panel as pn
import param
from pathlib import Path
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import (
    HuggingFaceBgeEmbeddings,
    SentenceTransformerEmbeddings,
)

pn.extension()

# Define parameters
MODEL_NAME = "TheBloke/Mistral-7B-v0.1-GGUF"
BASE_URL = "http://localhost:1234/v1"
OPENAI_API_KEY = "not-needed"
PARENT_FOLDER_PATH = Path(__file__).parent
EMBEDDINS_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
BASIC_INPUT_PATH = "basic_test_input.txt"


class Chatbot:
    model_name: str = param.String(default=MODEL_NAME)
    base_url: str = param.String(default=BASE_URL)
    openai_api_key: str = param.String(default=OPENAI_API_KEY)
    emdeddings: HuggingFaceBgeEmbeddings
    documents: list[Document]

    def __init__(
        self,
        is_locally: bool = True,
        model_name: str = None,
        base_url: str = None,
        openai_api_key: str = None,
    ) -> None:
        self.embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDINS_MODEL_NAME)
        self.documents = []
        self.__create_chat_interface()

        if not is_locally:
            if not (base_url and openai_api_key):
                raise Exception(
                    "In order to initialize chatbot using remote endpoint please provide 'base_url' and 'openai_api_key'"
                )
            elif not model_name:
                raise Warning(
                    f"Since 'model_name' is not provided, default one will be used, equals to {self.model_name}"
                )
            else:
                self.model_name = model_name

            self.base_url = base_url
            self.openai_api_key = openai_api_key
        else:
            self._load_text_documents(BASIC_INPUT_PATH)

        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url=self.base_url,
            openai_api_key=self.openai_api_key,
        )

    def _load_text_documents(self, file_paths: str | list[str]) -> None:
        file_paths = [file_paths] if type(file_paths) is str else file_paths
        for file_path in file_paths:
            file_path = PARENT_FOLDER_PATH / file_path
            print(file_path)
            text_loader = TextLoader(file_path)
            self.documents.extend(text_loader.load())

        self.__initialize_retriever()

    def _load_pdf_documents(self, file_paths: str | list[str]) -> None:
        file_paths = [file_paths] if type(file_paths) is str else file_paths
        for file_path in file_paths:
            file_path = PARENT_FOLDER_PATH / file_path
            with open(file_path, "rb") as file:
                pdf_loader = PdfReader(file)
                for page in pdf_loader.pages:
                    if page_content := page.extract_text():
                        self.documents.append(Document(page_content=page_content))

        self.__initialize_retriever()

    def __initialize_retriever(self):
        print(
            f"Number of Document objects before recursive splitting: {len(self.documents)}"
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        self.documents = text_splitter.split_documents(self.documents)
        vector_store = FAISS.from_documents(self.documents, self.embeddings)
        self.retriever = vector_store.as_retriever(search_type="similarity")
        print(
            f"Number of Document objects after recursive splitting: {len(self.documents)}"
        )

    def __create_conversational_chain(self) -> None:
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.llm_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=memory,
            verbose=True,
        )

    def __create_retrieval_chain(self) -> None:
        self.llm_chain = RetrievalQA.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            verbose=True,
        )

    def __create_chat_interface(self) -> None:
        self.chat_input = pn.widgets.TextInput(placeholder="Ask questions here!")
        self.chat_interface = pn.chat.ChatInterface(
            callback=self.__chat_respond_callback,
            sizing_mode="stretch_width",
            widgets=[self.chat_input],
            callback_exception="verbose",
        )
        self.chat_interface.send(
            "Send a message to start chatting!", user="System", respond=False
        )
        self.chat_interface.servable()

    async def __chat_respond_callback(
        self, contents: str, user: str, instance: pn.chat.ChatInterface
    ):
        output_key = "answer" if self.is_conversational else "result"
        response = await self.llm_chain.ainvoke(input=contents)

        yield {"user": MODEL_NAME, "value": response[output_key]}

    def run(self, is_conversational: bool = False) -> None:
        self.is_conversational = is_conversational
        if self.is_conversational:
            self.__create_conversational_chain()
        else:
            self.__create_retrieval_chain()
        self.chat_interface.show()


if __name__ == "__main__":
    app = Chatbot()
    app.run(is_conversational=True)
