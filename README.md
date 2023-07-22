<!-- #region -->
**This is the submission for assigment number 10 of ERA V1 course.**

**Problem Statement**
The Task given was to use CIFAR 10 data and get the custom resnet network with accuracy of minimum 90% in 24 EPOCHS. 

The architecture has to be followed as provided in the question. This architecture is same as one used by David C Page. The same is as follows:
PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
Layer1 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
Add(X, R1)
Layer 2 -
Conv 3x3 [256k]
MaxPooling2D
BN
ReLU
Layer 3 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
Add(X, R2)
MaxPooling with Kernel Size 4
FC Layer 
SoftMax

The image transformations are also specified which is as follows:
Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)

For the learning rate One Cycle LR is to be used with the following parameters:
Uses One Cycle Policy such that:
Total Epochs = 24
Max at Epoch = 5
LRMIN = FIND
LRMAX = FIND
NO Annihilation

**File Structure**
custom_resnet.py - has the customer resnet model created by me. 
era_s10_cifar.ipynb - the main file
images:
     Accuracy & Loss.jpg   -- Plot of train and test accuracy and loss with respect to epochs
     miss_classified_image.jpg  -- sample mis classified images. 
     test_dataset.jpg           -- sample test dataset
     train_dataset.jpg          -- sample train dataset after tranformation
modular:
     create_data_loader.py      -- import CIFAR dataset
     dataloader.py              -- to create train and test loader
     plots.py                   -- function to plot images
     train.py                   -- function to train model by calulating loss
     tranforms.py               -- function to transform image

The tranformation performed as as follows:

    def train_transforms(means,stds):
        transforms = A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                A.RandomCrop(height=32, width=32, always_apply=True),
                A.HorizontalFlip(),
                A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=means),
                ToTensorV2(),
            ]
        )

    def test_transforms(means,stds):
        transforms = A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                ToTensorV2(),
            ]
        )
        return transforms
        
Following are the sample images of train dataset:
<img src="https://github.com/saurabhmangal/era1_s10/blob/main/images/test_dataset.jpg" alt="alt text" width="600px">

Following are the sample imagese of the test dataset:
<img src="https://github.com/saurabhmangal/era1_s10/blob/main/images/test_dataset.jpg" alt="alt text" width="600px">




**Custom Resnet ARCHITECTURE**

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,792
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,856
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
              ReLU-7          [-1, 128, 16, 16]               0
            Conv2d-8          [-1, 128, 16, 16]         147,584
       BatchNorm2d-9          [-1, 128, 16, 16]             256
             ReLU-10          [-1, 128, 16, 16]               0
           Conv2d-11          [-1, 128, 16, 16]         147,584
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 256, 16, 16]         295,168
        MaxPool2d-15            [-1, 256, 8, 8]               0
      BatchNorm2d-16            [-1, 256, 8, 8]             512
             ReLU-17            [-1, 256, 8, 8]               0
           Conv2d-18            [-1, 512, 8, 8]       1,180,160
        MaxPool2d-19            [-1, 512, 4, 4]               0
      BatchNorm2d-20            [-1, 512, 4, 4]           1,024
             ReLU-21            [-1, 512, 4, 4]               0
           Conv2d-22            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-23            [-1, 512, 4, 4]           1,024
             ReLU-24            [-1, 512, 4, 4]               0
           Conv2d-25            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-26            [-1, 512, 4, 4]           1,024
             ReLU-27            [-1, 512, 4, 4]               0
        MaxPool2d-28            [-1, 512, 1, 1]               0
           Linear-29                   [-1, 10]           5,130
================================================================
Total params: 6,575,370
Trainable params: 6,575,370
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.44
Params size (MB): 25.08
Estimated Total Size (MB): 31.54
----------------------------------------------------------------


**Last Epoch Results:**
**EPOCH: 23**
**Loss=0.042819224298000336 LR =-1.5486702470463194e-06 Batch_id=48 Accuracy=98.64: 100% 49/49 [00:09<00:00,  5.15it/s]**

**Test set: Average loss: 0.0002, Accuracy: 9239/10000 (92.39%)**

Following are the plot of train and test losses and accuracies:
<img src="https://github.com/saurabhmangal/era1_s10/blob/main/images/Accuracy%20%26%20Loss.jpg" alt="alt text" width="600px">

Some of the sample misclassified images are as follows:
<img src="https://github.com/saurabhmangal/era1_s10/blob/main/images/miss_classified_image.jpg" alt="alt text" width="600px">

Plot for One Cycle LR policy:
<img src="https://github.com/saurabhmangal/era1_s10/blob/main/images/oneLRcurve.png" alt="alt text" width="600px">

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
