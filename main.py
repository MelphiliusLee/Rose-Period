from fastapi import FastAPI, Request
from Chatbot.rag_model import RAGModel
from Chatbot.data import data_saver
from Predict_model.lango_models import LangoModels
import pandas as pd

data_saver.save_data_to_vector("data")
rag_model = RAGModel("Chatbot/faiss_index")
lango_model = LangoModels("Predict_model/FedCycleData071012 (2) (1).csv")
lango_model.train_and_evaluate()
app = FastAPI()

@app.get("/chatbot")
def get_answer_from_chatbot(request: Request):
    query = request.query_params.get("query")
    if not query:
        return {"success": 0, "message": "Please provide a question."}

    try:
        # fetch answer from rag_model
        answer = rag_model.get_answer(query)
        return {"success": 1, "message": answer}
    except Exception as e:
        # catch any error
        return {"success": 0, "message": f"An error occurred while processing the request: {str(e)}"}


@app.get("/predict_period")
def get_answer_from_lango(request: Request):
    age = request.query_params.get("Age")
    height = request.query_params.get("Height")
    weight = request.query_params.get("Weight")
    mean_bleeding_intensity = request.query_params.get("MeanBleedingIntensity")
    number_of_days_intercourse = request.query_params.get("NumberOfDaysOfIntercourse")
    years_married = request.query_params.get("YearOfMarried")
    mean_cycle_length = request.query_params.get("MeanCycleLength")

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

    data = [age, height, weight, mean_bleeding_intensity, number_of_days_intercourse, years_married, mean_cycle_length]
    columns = ["Age", "Height", "Weight", "MeanBleedingIntensity",
               "NumberofDaysofIntercourse", "Yearsmarried", "MeanCycleLength"]

    # Create DataFrame
    dataframe = pd.DataFrame([data], columns=columns, index=[1])

    try:
        print(data)
        length_of_cycle = lango_model.predict("model_LengthofCycle", dataframe).tolist()
        print("success getting length_of_cycle:", length_of_cycle)
        length_of_menses = lango_model.predict("model_LengthofMenses", dataframe).tolist()
        print("success getting length_of_menses:", length_of_menses)
        unusual_bleeding = lango_model.predict("model_UnusualBleeding", dataframe).tolist()
        print("success getting unusual_bleeding:", unusual_bleeding)

        # Convert the first two to integers
        length_of_cycle = int(length_of_cycle[0])
        length_of_menses = int(length_of_menses[0])
        unusual_bleeding = unusual_bleeding[0]

        # Validate unusual_bleeding is either 0 or 1
        if unusual_bleeding not in [0, 1]:
            return {"success": 0, "message": "Failed to predict unusual bleeding."}

        return {
            "success": 1,
            "length_of_cycle": length_of_cycle,
            "length_of_menses": length_of_menses,
            "unusual_bleeding": unusual_bleeding
        }

    except Exception as e:
        return {"success": 0, "message": f"An error occurred while processing the request: {str(e)}"}
