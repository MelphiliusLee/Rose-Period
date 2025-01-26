
from rag_histroy import RAGModel
from data import data_saver
data_saver.save_data_to_vector("../data")
rag_model = RAGModel(vectorstore_path="faiss_index")

# Ask a question
response0 = rag_model.get_answer("Who are you?")
print("Response 0:", response0)

response1 = rag_model.get_answer("What are the phases of the menstrual cycle?")
print("Response 1:", response1)

# Follow up with another question
response2 = rag_model.get_answer("How long does the menstrual phase typically last?")
print("Response 2:", response2)

# Memory ensures the agent remembers previous questions
response3 = rag_model.get_answer("Can you summarize everything we've discussed?")
print("Response 3:", response3)

response4 = rag_model.get_answer("Do Cai Xu Kun play basketball?")
print("Response 4:", response4)



