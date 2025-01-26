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
    age = request.query_params.get("Age")
    height = request.query_params.get("Height")
    weight = request.query_params.get("Weight")
    mean_bleeding_intensity = request.query_params.get("Mean Bleeding Intensity")
    number_of_days_intercourse = request.query_params.get("Number of Days of Intercourse")
    years_married = request.query_params.get("Years of Married")
    mean_cycle_length = request.query_params.get("MeanCycleLength")

    # body = request.json()
    # mean_bleeding_intensity = body.get("Mean Bleeding Intensity")

    try:
        age = int(age)
        height = int(height)
        weight = int(weight)
        mean_bleeding_intensity = int(mean_bleeding_intensity)
        number_of_days_intercourse = int(number_of_days_intercourse)
        years_married = int(years_married)
        mean_cycle_length = int(mean_cycle_length)
    except Exception as e:
        return {"success": 0, "message": f"An error occurred while converting parameters: {str(e)}"}

    X = [age, height, weight, mean_bleeding_intensity, number_of_days_intercourse, years_married, mean_cycle_length]

    try:
        length_of_cycle = lango_model.predict("model_LengthofCycle", X)
        length_of_menses = lango_model.predict("model_LengthofMenses", X)
        unusual_bleeding = lango_model.predict("model_UnusualBleeding", X)
        return {
            "success": 1,
            "length_of_cycle": length_of_cycle,
            "length_of_menses": length_of_menses,
            "unusual_bleeding": unusual_bleeding
        }
    except Exception as e:
        return {"success": 0, "message": f"An error occurred while processing the request: {str(e)}"}
