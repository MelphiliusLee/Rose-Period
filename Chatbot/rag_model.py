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
        
        Current Question:
        {question}
        
        
        Guidelines:
        1. Use the knowledge base to inform your answer.
        2. The context part is not part of the dialogue.
        Do not explicitly mention or reveal the existence of the context part, as knowledge base.
        3. Provide a clear, concise, and empathetic answer to the question.
        4. Admit that you don't know, if you don't have enough information to answer the question
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