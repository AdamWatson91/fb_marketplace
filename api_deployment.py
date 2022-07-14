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
from processor_text import TextProcessor
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

class BertCNN(torch.nn.Module):
    def __init__(self, embedding_size=768, num_classes=13, decoder: dict = None, device='cpu'):
        super(BertCNN, self).__init__()
        self.decoder = decoder
        self.layers = torch.nn.Sequential(
            nn.Conv1d(embedding_size, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Flatten(),
            nn.Linear(1152 , 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
            ).to(device)

    def forward(self, X):
        X = self.layers(X)
        return X
    
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

# Set up image processing and model
image_processor = ImageProcessor()
with open('image_decoder.pkl', 'rb') as f:
    image_decoder = pickle.load(f)
img_classifer = ImageClassifier(num_classes=len(image_decoder), decoder = image_decoder)
img_classifer.load_state_dict(torch.load('image_model.pt', map_location='cpu'))

# Set up text processing and model
text_processor = TextProcessor()
with open('bert_decoder.pkl', 'rb') as f:
    text_decoder = pickle.load(f)
text_classifer = BertCNN(num_classes=len(text_decoder), decoder = text_decoder)
text_classifer.load_state_dict(torch.load('bert_model.pt', map_location='cpu'))

app = FastAPI()

@app.post('/image_predict')
def image_predict(image: UploadFile = File(...)):
    img = Image.open(image.file)
    processed_img = image_processor(img)
    prediction = img_classifer.predict(processed_img)
    probs = img_classifer.predict_proba(processed_img)
    classes = img_classifer.predict_classes(processed_img)
    print(prediction)
    print(probs)
    print(classes)
    return JSONResponse(status_code=200, content={'prediction': prediction.tolist(), 'probs': probs.tolist(), 'class': classes})

@app.post('/text_predict')
def text_predict(text: str = Form(...)):
    text = text
    processed_text = text_processor(text)
    prediction = text_classifer.predict(processed_text)
    probs = text_classifer.predict_proba(processed_text)
    classes = text_classifer.predict_classes(processed_text)
    print(prediction)
    print(probs)
    print(classes)
    return JSONResponse(status_code=200, content={'prediction': prediction.tolist(), 'probs': probs.tolist(), 'class': classes})

if __name__ == '__main__':
    uvicorn.run('api_deployment:app', host='0.0.0.0', port=8080)

