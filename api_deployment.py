import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from processor_image import ImageProcessor
import fastapi
from fastapi import FastAPI
from fastapi import Request
from fastapi import UploadFile
from fastapi import File
import uvicorn
from fastapi.responses import JSONResponse

class ImageClassifier(nn.Module):
    def __init__(self,
                num_classes, decoder: dict = None, device='cpu'):
        super(ImageClassifier, self).__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = self.resnet50.fc.out_features
        self.linear = nn.Linear(out_features, num_classes).to(device)
        self.main = nn.Sequential(self.resnet50, self.linear).to(device)
        self.decoder = decoder
    
    def forward(self, inp):
        x = self.main(inp)
        return x
    
    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x
    
    def predict_proba(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return torch.softmax(x, dim=1)

    def predict_classes(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return self.decoder[int(torch.argmax(x, dim=1))]

image_processor = ImageProcessor()
with open('image_decoder.pkl', 'rb') as f:
    image_decoder = pickle.load(f)
img_classifer = ImageClassifier(num_classes=len(image_decoder), decoder = image_decoder)
img_classifer.load_state_dict(torch.load('image_model.pt', map_location='cpu'))

app = FastAPI()

@app.get('/example')
def dummy(x):
    print(x)
    return 'Hello'

@app.post('/example2')
def dummy_post(image: UploadFile = File(...)):
    img = Image.open(image.file)
    processed_img = image_processor(img)
    print(processed_img)
    prediction = img_classifer.predict(processed_img)
    probs = img_classifer.predict_proba(processed_img)
    classes = img_classifer.predict_classes(processed_img)
    print(prediction)
    print(probs)
    print(classes)
    return JSONResponse(status_code=200, content={'prediction': prediction.tolist(), 'probs': probs.tolesit(), 'class': classes})

if __name__ == '__main__':
    uvicorn.run('api_deployment:app', host='0.0.0.0', port=8080)

