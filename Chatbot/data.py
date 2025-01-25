import os
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

class data_saver:
    def save_data_to_vector(directory_folder):
        def load_multi_documents(directory_path):
            data = []
            for filename in os.listdir(directory_path):
                if filename.endswith(".txt"):
                    loader = TextLoader(directory_path+"/"+filename)
                    temp = loader.load()
                    data += temp
                elif filename.endswith(".doc") or filename.endswith(".docx"):
                    loader = Docx2txtLoader(directory_path+"/"+filename)
                    temp = loader.load()
                    data+=temp
                elif filename.endswith(".pdf"):
                    loader = PyPDFLoader(directory_path+"/"+filename)
                    temp = loader.load()
                    data+=temp
            return data
        documents = load_multi_documents(directory_folder)
        # "/Users/xuwei/Desktop/AI_master/Hackerson/data"
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(
            docs, embedding=HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
        )

        vectorstore.save_local("faiss_index")