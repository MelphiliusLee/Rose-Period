import os
from typing import List, TypedDict
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import hub
from langchain_community.llms import Tongyi
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# Set the API key if not already set
load_dotenv()
api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("DASHSCOPE_API_KEY is not set in the .env file or environment variables")

class State(TypedDict):
    question: str
    context: List[Document]
    histroy: List[str]
    answer: str


class RAGModel:
    def __init__(self, vectorstore_path: str, embedding_model: str = "moka-ai/m3e-base", temperature: float = 0.5):
        """
        Initialize the RAGModel class.

        Args:
            vectorstore_path (str): Path to the FAISS vectorstore.
            embedding_model (str): HuggingFace model name for embeddings.
            temperature (float): Temperature for the language model.
        """

        self.vector_store = FAISS.load_local(
            vectorstore_path,
            embeddings=HuggingFaceEmbeddings(model_name=embedding_model),
            allow_dangerous_deserialization=True
        )
        self.prompt = PromptTemplate(template="""
        You are a highly knowledgeable and empathetic assistant specializing in women's menstrual health. 
        Use the following context and conversation history to answer the user's question.
        
        Context:
        {context}
        
        Conversation History:
        {history}
        
        Current Question:
        {question}
        
        
        Guidelines:
        1. Use the knowledge base to inform your answer.
        2. The context part is not part of the dialogue.
        Do not explicitly mention or reveal the existence of the context part, as knowledge base.
        3. Provide a clear, concise, and empathetic answer to the question.
        Answer the question clearly and concisely:
        """)
        self.llm = Tongyi(temperature=temperature)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build and compile the retrieval and generation graph.

        Returns:
            StateGraph: The compiled StateGraph.
        """
        # Define application steps
        def retrieve(state: State):
            retrieved_docs = self.vector_store.similarity_search(state["question"])
            return {"context": retrieved_docs}

        def generate(state: State):
            history_content = "\n".join(state.get("history", []))
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = self.prompt.invoke({"history": history_content,"question": state["question"], "context": docs_content})
            response = self.llm.invoke(messages)
            if "history" not in state:
                state["history"] = []
            state["history"].append(f"Q: {state['question']}\nA: {response}")
            return {"answer": response}

        # Build and compile the graph
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()

    def get_answer(self, question: str) -> str:
        """
        Process the input question and return the answer.

        Args:
            question (str): The user's question.

        Returns:
            str: The generated answer.
        """
        if not hasattr(self, "history"):
            self.history = []  # Initialize conversation history

        # Invoke the graph with history
        state = {
            "question": question,
            "context": [],
            "history": self.history,
            "answer": ""
        }
        response = self.graph.invoke(state)
        return response["answer"]