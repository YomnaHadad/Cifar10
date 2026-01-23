# Cifar10
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.\
There are 50000 training images and 10000 test images.

## MLP Vs CNN Vs LeNet
**_3 models from scratch_**

**1. cifar_mlp.py file**
 
   as (MLP) model typically achieves an accuracy of around 45% to 60% on the CIFAR-10 dataset. \
   i built a mlp model with 4-hidden layers. \
   **achieving Accuracy: 0.5588 - loss: 1.2802**

**2. cifar_cnn.py file**
 
   building a simple CNN model with bunch of convolutional layes, normalization layers, and pooling layers \
   **achieving Accuracy: 0.8409 - loss: 0.5430**

**3. cifar_LeNet.py**
 
   Building the LeNet Standard Architecture \
   the typically achievable Accuracy of cifar with LeNet is around 55% to 60%  \
   **val_accuracy: 0.5704 - val_loss: 1.2523**
   
