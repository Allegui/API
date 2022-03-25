import numpy as np
import pandas as pd
from fastapi import FastAPI
# Initiate app instance
app = FastAPI()

@app.get("/predict/{name}")
async def predict(name: str):
    return f"My name is {name}"
