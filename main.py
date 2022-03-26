import uvicorn
import numpy as np
import pandas as pd
import joblib
import xgboost
# FastAPI libray
from fastapi import FastAPI


test_X = pd.read_csv('test_X.csv')
test_X = test_X.drop(["Unnamed: 0"], axis=1)

SK_ID_CURR_test_X = pd.read_csv('SK_ID_CURR_test_X.csv')
SK_ID_CURR_test_X = SK_ID_CURR_test_X.drop(["Unnamed: 0"], axis=1)


# Initiate app instance
app = FastAPI()

xgb_cl_undersampling = joblib.load('xgb_cl_undersampling.joblib')

# # ML API endpoint for making prediction aganist the request received from client

  
@app.get("/predict/{identifiant}")
async def predict(identifiant: int):
    
    data_df = test_X.loc[test_X.index==np.asscalar(SK_ID_CURR_test_X.loc[SK_ID_CURR_test_X['SK_ID_CURR']==identifiant].index),:]
    
    # Create prediction
    score = np.array(xgb_cl_undersampling.predict_proba(data_df),dtype='float64')[0][1]
  
#     prediction = np.array(data_df)[0][4]

    return {"prediction": str(score)}
  
  
if __name__ == '__main__' : 
    uvicorn.run(app, host="127.0.0.1",port=8000)  
