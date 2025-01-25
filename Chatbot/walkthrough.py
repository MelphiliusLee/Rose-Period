from rag_model import RAGModel
from data import data_saver

data_saver.save_data_to_vector("/Users/xuwei/Desktop/AI_master/Rose-Period/data")
rag_model = RAGModel("faiss_index")
print(rag_model.get_answer("What the length of Menstrual Phase?"))
print(rag_model.get_answer("What is the phases in Menstrual Cycle?"))
print(rag_model.get_answer("Do Cai Xu Kun play basketball?"))