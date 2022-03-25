import uvicorn
import numpy as np
import pandas as pd
import pickle
# FastAPI libray
from fastapi import FastAPI


# test_X = pd.read_csv('test_X.csv')
# test_X = test_X.drop(["Unnamed: 0"], axis=1)

# SK_ID_CURR_test_X = pd.read_csv('SK_ID_CURR_test_X.csv')
# SK_ID_CURR_test_X = SK_ID_CURR_test_X.drop(["Unnamed: 0"], axis=1)


# Initiate app instance
app = FastAPI()

# Import Model
pickle_in = open("xgb_cl_undersampling.pkl","rb")
xgb_cl_undersampling = pickle.load(pickle_in)



@app.get("/predict/{name}")
async def predict(name: int):
  
    return f"My name is {name}"










# ML API endpoint for making prediction aganist the request received from client

  
# @app.get("/predict/{identifiant}")
# async def predict(identifiant: int):
    
#     data_df = test_X.loc[test_X.index==np.asscalar(SK_ID_CURR_test_X.loc[SK_ID_CURR_test_X['SK_ID_CURR']==identifiant].index),:]
    
#     # Create prediction
#     prediction = float(xgb_cl_undersampling.predict_proba(data_df)[0][1])
#     return {"prediction": prediction}
