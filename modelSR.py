### Implementation by Sebastian Raschka at github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/basic-ml/logistic-regression.ipynb
### Comment was added to the original code

import torch
import torch.nn.functional as F

class SoftmaxRegression(torch.nn.Module):
    """
    Multinomial logistic regression model

    Parameters
    ----------
    num_features : int
      Number of input features
    num_classes : int
        Number of classes
    """

    def __init__(self, num_features, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)
        
        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()
        
    def forward(self, x):
        logits = self.linear(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas