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




@app.get('/get_variable')
async def get_variable(identifiant: int) :
    resp = test_X.loc[test_X.index==np.asscalar(SK_ID_CURR_test_X.loc[SK_ID_CURR_test_X['SK_ID_CURR']==identifiant].index),:]
    return {f"valeur de la variable de l'individu {identifiant}" : np.array(resp)[0][4]}


def predict(resp) : 
    score = xgb_cl_undersampling.predict_proba(resp)[0][1]    
    return {"score" : str(score)}
    

@app.get('/get_predict')
async def get_predict(identifiant: int) :
    resp = test_X.loc[test_X.index==np.asscalar(SK_ID_CURR_test_X.loc[SK_ID_CURR_test_X['SK_ID_CURR']==identifiant].index),:]
    resp_ = 0.69
    return predict(resp_)


  
if __name__ == '__main__' : 
    uvicorn.run(app, host="127.0.0.1",port=8000)  
