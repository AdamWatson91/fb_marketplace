import torch.nn as nn
import torch

class ImageClassifier(nn.Module):
    """
    This classifier applies transfer learning using the resnet50 model
    for a multi-classification of image data.

    Args:
        num_classes (int): The number of potential classess in the
                            classifciation problem
    """
    def __init__(self,
                num_classes: int, 
                device: str):
        super(ImageClassifier, self).__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = self.resnet50.fc.out_features
        self.linear = nn.Linear(out_features, num_classes).to(device)
        self.main = nn.Sequential(self.resnet50, self.linear).to(device)

    def forward(self, inp):
        x = self.main(inp)
        return x
