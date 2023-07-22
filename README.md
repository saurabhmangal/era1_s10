<!-- #region -->
**This is the submission for assigment number 10 of ERA V1 course.**<br> 

**Problem Statement**<br> 
The Task given was to use CIFAR 10 data and get the custom resnet network with accuracy of minimum 90% in 24 EPOCHS. <br> 

The architecture has to be followed as provided in the question. This architecture is same as one used by David C Page. The same is as follows:<br> 
-**PrepLayer** <br> 
  -Conv 3x3 s1, p1) >> BN >> RELU [64k]<br> 

-**Layer1**<br> 
  -X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]<br> 
  -R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] <br> 
  -Add(X, R1)<br> 

-**Layer 2**<br> 
  -Conv 3x3 [256k]<br> 
  -MaxPooling2D<br> 
  -BN<br> 
  -ReLU<br> 

-**Layer 3**<br> 
  -X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]<br> 
  -R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]<br> 
  -Add(X, R2)<br> 
  -MaxPooling with Kernel Size 4<br> 

-FC Layer <br> 
-SoftMax <br> 

The image transformations are also specified which is as follows:<br> 
  -RandomCrop 32, 32 (after padding of 4)<br> 
  -FlipLR<br> 
  -Followed by CutOut(8, 8)<br> 

For the learning rate One Cycle LR is to be used with the following parameters:<br> 
Uses One Cycle Policy such that:<br> 
  -Total Epochs = 24<br> 
  -Max at Epoch = 5<br> 
  -LRMIN = FIND<br> 
  -LRMAX = FIND<br> 
  -NO Annihilation<br> 

**File Structure**<br> 
-custom_resnet.py - has the customer resnet model created by me<br>
-era_s10_cifar.ipynb - the main file<br> 
-images:<br> 
  -Accuracy & Loss.jpg   -- Plot of train and test accuracy and loss with respect to epochs<br> 
  -miss_classified_image.jpg  -- sample mis classified images. <br> 
  -test_dataset.jpg           -- sample test dataset<br> 
  -train_dataset.jpg          -- sample train dataset after tranformation<br> 
-modular:<br> 
  -create_data_loader.py      -- import CIFAR dataset<br> 
  -dataloader.py              -- to create train and test loader<br> 
  -plots.py                   -- function to plot images<br> 
  -train.py                   -- function to train model by calulating loss<br> 
  -tranforms.py               -- function to transform image<br> 

The tranformation performed as as follows:<br> 

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
        
Following are the sample images of train dataset:<br> 
<img src="https://github.com/saurabhmangal/era1_s10/blob/main/images/test_dataset.jpg" alt="alt text" width="600px">

Following are the sample imagese of the test dataset:<br> 
<img src="https://github.com/saurabhmangal/era1_s10/blob/main/images/test_dataset.jpg" alt="alt text" width="600px">


**Custom Resnet ARCHITECTURE**<br> 
<img src="https://github.com/saurabhmangal/era1_s10/blob/main/images/model10.JPG" alt="alt text" width="600px">


**Last Epoch Results:**<br>
EPOCH: 23<br>
Loss=0.042819224298000336 LR =-1.5486702470463194e-06 Batch_id=48 Accuracy=98.64: 100% 49/49 [00:09<00:00,  5.15it/s]<br>
Test set: Average loss: 0.0002, Accuracy: 9239/10000 (92.39%)<br>

Following are the plot of train and test losses and accuracies:<br> 
<img src="https://github.com/saurabhmangal/era1_s10/blob/main/images/Accuracy%20%26%20Loss.jpg" alt="alt text" width="600px"><br> 

Some of the sample misclassified images are as follows:<br> 
<img src="https://github.com/saurabhmangal/era1_s10/blob/main/images/mis_classified_image.jpg" alt="alt text" width="600px"><br> 

Plot for One Cycle LR policy:<br> 
<img src="https://github.com/saurabhmangal/era1_s10/blob/main/images/oneLRcurve.png" alt="alt text" width="600px"><br> 

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
