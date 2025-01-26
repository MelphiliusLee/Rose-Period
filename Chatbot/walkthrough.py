
data_saver.save_data_to_vector("../data")
rag_model = RAGModel("faiss_index")
print(rag_model.get_answer("Who are you?"))
print(rag_model.get_answer("What the length of Menstrual Phase?"))
print(rag_model.get_answer("What is the phases in Menstrual Cycle?"))
print(rag_model.get_answer("Do Cai Xu Kun play basketball?"))