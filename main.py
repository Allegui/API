from fastapi import FastAPI

app = FastAPI()

@app.get('/beginning')
async def say_hello():
    return "Hi I'm Guillaume ! I'm using FastAPI."