from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/hello")
async def hello():
    home = os.environ['MY_ENV']
    print(home)
    return {"message": "Hello World"}
