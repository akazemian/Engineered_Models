# Engineered_Models

- This repository introduces the expansion model - an wide untrained convolutional neural networks that competes with standard pre-trained CNNs at predicting neural responses in primate visual cortex. 
- To use the expansion model, clone the repository to a local directrory and proceed as you would with any pytorch model. 
- The call_model notebook shows how the model can be initialized.

# Model Expansion Model Architecture

The expansion model is an untrained wide convolutionl neural nework with a pre-defined first layer that aims to model early to mid level visual cortex. The deeper layers of the model use a large number of randomly initialized convolution filters to sample brain-like features from a high dimensional space of representations. In every layer of the model, the follwoing operations are performed in order: convolution -> non-linear function -> pooling.

![alt text](https://github.com/Atlaskz/Bionic-AI-Predicting-Grasp-and-Lift-Motions/Engineered_Models/Expansion.png?style=centerme)

