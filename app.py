import panel as pn
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Define parameters
MODEL_NAME = "Mistral-7B-Instruct-v0.1"
BASE_URL = "http://localhost:1234/v1"
OPENAI_API_KEY = "not-needed"

# Load Panel extension
pn.extension()


class Chatbot:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        base_url: str = BASE_URL,
        openai_api_key: str = OPENAI_API_KEY,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.openai_api_key = openai_api_key
        self.__create_llm_chain()
        self.__create_chat_interface()

    def __create_llm_chain(self) -> None:
        """
        Create LLM chain for chatbot.
        """
        TEMPLATE = """<s>[INST] You are a friendly chat bot who's willing to help answer user:
        {user_input} [/INST] </s>
        """
        prompt = PromptTemplate(template=TEMPLATE, input_variables=["user_input"])
        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url=self.base_url,
            openai_api_key=self.openai_api_key,
        )
        self.llm_chain = LLMChain(llm=self.llm, prompt=prompt)

    def __create_chat_interface(self) -> None:
        """
        Create a chat interface using Panel's chat components.
        """
        self.chat_interface = pn.chat.ChatInterface(callback=self.callback)

        self.chat_interface.send(
            "Send a message to receive an echo!", user="System", respond=False
        )
        self.chat_interface.servable()

    async def callback(
        self, contents: str, user: str, instance: pn.chat.ChatInterface
    ) -> None:
        """
        Callback function to handle user input.

        Args:
            contents (str): User's message.
            user (str): User's name.
            instance (pn.chat.ChatInterface): Chat interface instance.
        """
        instance.send(
            await self.llm_chain.apredict(user_input=contents),
            user=self.llm.model_name,
            respond=False,
        )

    def run(self) -> None:
        """
        Run chatbot.
        """
        self.chat_interface.show()


if __name__ == "__main__":
    # Create and run chatbot
    app = Chatbot()
    app.run()
