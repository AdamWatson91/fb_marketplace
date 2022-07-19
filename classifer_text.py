import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')


class TextClassifier(nn.Module):
    """
    This classifier applies transfer learning using the bert model
    for a multi-classification of text data.

    Args:
        input_size (int): The embedding size to use in the model.
    """
    def __init__(self,
                input_size: int = 768):
        super(TextClassifier, self).__init__()
        self.main = nn.Sequential(nn.Conv1d(input_size, 256, kernel_size=3, stride=1, padding=1),
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
                                nn.Flatten(),
                                nn.Linear(192 , 128))

    def forward(self, inp):
        x = self.main(inp)
        return x
        