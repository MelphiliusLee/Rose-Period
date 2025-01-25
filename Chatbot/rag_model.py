import os
from typing import List, TypedDict
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import hub
from langchain_community.llms import Tongyi
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document

# Set the API key if not already set
if not os.environ.get("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = "sk-8e24269478904f5683d6998e2ed637c8"


class State(TypedDict):
    question: str
    context: List[Document]
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
        self.prompt = hub.pull("rlm/rag-prompt")
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
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
            response = self.llm.invoke(messages)
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
        response = self.graph.invoke({"question": question})
        return response["answer"]
