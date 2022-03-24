from fastapi import FastAPI

app = FastAPI()

@app.get('/beginning/{name}')
async def say_hello(name: str):
    return f"Hello, My name is {name} and I'm using Fast API."
