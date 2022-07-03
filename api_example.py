import fastapi
from fastapi import FastAPI
from fastapi import Request
from fastapi import UploadFile
from fastapi import File
import uvicorn

app = FastAPI()

@app.get('/example')
def dummy():
    return 'Hello'

@app.post('/example')
def dummy_post(image: UploadFile = File(...)):
    print(image)
    return 'Bye'

if __name__ == '__main__':
    uvicorn.run('api_example:app', host='0.0.0.0', port=8080)