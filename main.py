from typing import Union
from fastapi import FastAPI, Request
from Chatbot.rag_model import RAGModel
from Chatbot.data import data_saver
from Predict_model.lango_models import LangoModels

data_saver.save_data_to_vector("data")
rag_model = RAGModel("Chatbot/faiss_index")
lango_model = LangoModels("Predict_model/FedCycleData071012 (2) (1).csv")

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


@app.get("/predict_period")
def get_answer_from_lango(request: Request):
    days_to_next_period = request.query_params.get("DaysToNextPeriod")
    try:
        days_to_next_period = int(days_to_next_period)
    except Exception as e:
        return {"success": 0, "message": f"An error occurred while converting request body parameter: {str(e)}"}

    length_of_next_period = request.query_params.get("LengthToNextPeriod")
    try:
        length_of_next_period = int(length_of_next_period)
    except Exception as e:
        return {"success": 0, "message": f"An error occurred while converting request body parameter: {str(e)}"}

    has_unusual_bleeding = request.query_params.get("HasUnusualBleeding")
    try:
        has_unusual_bleeding = int(has_unusual_bleeding)
    except Exception as e:
        return {"success": 0, "message": f"An error occurred while converting request body parameter: {str(e)}"}

    try:
        length_of_cycle = lango_model.predict("model_LengthofCycle", length_of_next_period)
        length_of_menses = lango_model.predict("model_LengthofMenses", days_to_next_period)
        unusual_bleeding = lango_model.predict("model_UnusualBleeding", has_unusual_bleeding)
        return {
            "success": 1,
            "length_of_cycle": length_of_cycle,
            "length_of_menses": length_of_menses,
            "unusual_bleeding": unusual_bleeding
        }
    except Exception as e:
        return {"success": 0, "message": f"An error occurred while processing the request: {str(e)}"}
