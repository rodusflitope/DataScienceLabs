import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class WaterMeasurement(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


app = FastAPI()

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.get("/")
def home():
    return {
        "modelo": "XGBoost optimizado con Optuna",
        "problema": "Clasificar la potabilidad del agua de Maipú",
        "entrada": {
            "ph": "float",
            "Hardness": "float",
            "Solids": "float",
            "Chloramines": "float",
            "Sulfate": "float",
            "Conductivity": "float",
            "Organic_carbon": "float",
            "Trihalomethanes": "float",
            "Turbidity": "float"
        },
        "salida": {
            "potabilidad": "int (0: no potable, 1: potable)"
        }
    }


@app.post("/potabilidad/")
def predict_potability(measurement: WaterMeasurement):
    data = [[
        measurement.ph,
        measurement.Hardness,
        measurement.Solids,
        measurement.Chloramines,
        measurement.Sulfate,
        measurement.Conductivity,
        measurement.Organic_carbon,
        measurement.Trihalomethanes,
        measurement.Turbidity
    ]]
    prediction = int(model.predict(data)[0])
    return {"potabilidad": prediction}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
