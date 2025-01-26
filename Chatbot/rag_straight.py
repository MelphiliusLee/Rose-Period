import os
from operator import itemgetter
from typing import List, Tuple


from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_community.llms import Tongyi
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("DASHSCOPE_API_KEY is not set in the .env file or environment variables")



vector_store = FAISS.load_local("faiss_index",embeddings=HuggingFaceEmbeddings(model_name="moka-ai/m3e-base"),allow_dangerous_deserialization=True)
prompt = hub.pull("rlm/rag-prompt")
llm = Tongyi(temperature=0.5)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response0 = graph.invoke({"question": "What the length of Menstrual Phase?"})
print(response0["answer"])
response1 = graph.invoke({"question": "What is the phases in Menstrual Cycle?"})
print(response1["answer"])
response2 = graph.invoke({"question": "Do Cai Xu Kun play basketball?"})
print(response2["answer"])