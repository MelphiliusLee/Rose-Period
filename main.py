from typing import Union
from fastapi import FastAPI, Request
from Chatbot.rag_model import RAGModel
from Chatbot.data import data_saver

data_saver.save_data_to_vector("data")
rag_model = RAGModel("Chatbot/faiss_index")

app = FastAPI()

@app.get("/chatbot")
def get_answer_from_chatbot(request: Request):
    query = request.query_params.get("query")
    if not query:
        return {"success": 0, "message": "Please provide a question."}

    try:
        # 调用模型获取答案
        answer = rag_model.get_answer(query)
        return {"success": 1, "message": answer}
    except Exception as e:
        # 捕获并返回可能发生的任何异常
        return {"success": 0, "message": f"An error occurred while processing the request: {str(e)}"}
