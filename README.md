
# The Expansion Model

The expansion model is an untrained wide convolutionl neural nework with a pre-defined first layer that aims to model early to mid level visual cortex. The deeper layers of the model use a large number of randomly initialized convolution filters to sample brain-like features from a high dimensional space of representations. In every layer of the model, the follwoing operations are performed in order: convolution -> non-linear function -> pooling.

![alt text](https://github.com/akazemian/Engineered_Models/blob/master/expansion.png?style=centerme)


# Using the model

- Clone this repository to a local directrory. 
```
git clone https://github.com/akazemian/Engineered_Models.git
```

- Navigate to the local directory, import the model and initiate it.
```
from models.expansion_model import ExpansionModel

model = ExpansionModel().Build()
print(model)
```

- Pass your images (torch arrays) through the model and extract the features.
```
import torch
X = torch.rand(1,3,96,96) # N, C, W, H
model(X)
```

- Alternatively, open the call_model notebook and initialize the model uing the provided code blocks.
