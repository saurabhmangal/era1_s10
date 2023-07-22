<!-- #region -->
**This is the submission for assigment number 8 of ERA V1 course.**

**Problem Statement**
The Task given was to use CIFAR 10 data and get the convolutional network with atleast 70% accuracy. 

The number of parameters had to be less than 50,000 parameters. 

It was also asked to use Batch Normalization, Group Normalization and Layer Normalization and observe the results. 

**File Structure**
model.py - has different classes for batch normalization, group normalization and layer normalization. Also as asked, I have also provided the networks used for Assignment 6 and 7. The names are as follows:
     Net_batch_normalization
     Net_group_norm
     Net_layer_normalization
     Net_s6
     Net_s7

The description of the data is as follows:

Dataset CIFAR10
    Number of datapoints: 50000
    Root location: ./data
    Split: Train
    StandardTransform
Transform: Compose(
               RandomAutocontrast(p=0.1)
               RandomRotation(degrees=[-7.0, 7.0], interpolation=nearest, expand=False, fill=1)
               ToTensor()
               Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
           )
Dataset CIFAR10
    Number of datapoints: 10000
    Root location: ./data
    Split: Test
    StandardTransform
Transform: Compose(
               ToTensor()
               Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
           )

Following are the sample images of train dataset:
<img src="https://github.com/saurabhmangal/era_s8/blob/master/train_dataset.jpg" alt="alt text" width="600px">

Following are the sample imagese of the test dataset:
<img src="https://github.com/saurabhmangal/era_s8/blob/master/test_dataset.jpg" alt="alt text" width="600px">




**PARAMETERS FOR BATCH NORMALIZARTION ARCHITECTURE**

----------------------------------------------------------------
        Layer (type)               Output Shape         Param # <\b>
        
================================================================
            Conv2d-1           [-1, 48, 32, 32]           1,344 <\b>
              ReLU-2           [-1, 48, 32, 32]               0 <\b>
       BatchNorm2d-3           [-1, 48, 32, 32]              96 <\b>
         Dropout2d-4           [-1, 48, 32, 32]               0 <\b>
            Conv2d-5           [-1, 48, 32, 32]          20,784 <\b>
              ReLU-6           [-1, 48, 32, 32]               0 <\b>
       BatchNorm2d-7           [-1, 48, 32, 32]              96 <\b>
         Dropout2d-8           [-1, 48, 32, 32]               0 <\b>
            Conv2d-9           [-1, 32, 32, 32]           1,568 <\b>
        MaxPool2d-10           [-1, 32, 16, 16]               0 <\b>
           Conv2d-11           [-1, 32, 16, 16]           9,248 <\b>
             ReLU-12           [-1, 32, 16, 16]               0 <\b>
      BatchNorm2d-13           [-1, 32, 16, 16]              64 <\b>
        Dropout2d-14           [-1, 32, 16, 16]               0 <\b>
           Conv2d-15           [-1, 32, 16, 16]           9,248 <\b>
             ReLU-16           [-1, 32, 16, 16]               0 <\b>
      BatchNorm2d-17           [-1, 32, 16, 16]              64 <\b>
        Dropout2d-18           [-1, 32, 16, 16]               0 <\b>
           Conv2d-19           [-1, 32, 16, 16]           1,056 <\b>
        MaxPool2d-20             [-1, 32, 8, 8]               0 <\b>
           Conv2d-21             [-1, 16, 8, 8]           4,624 <\b>
             ReLU-22             [-1, 16, 8, 8]               0 <\b>
      BatchNorm2d-23             [-1, 16, 8, 8]              32 <\b>
        Dropout2d-24             [-1, 16, 8, 8]               0 <\b>
           Conv2d-25              [-1, 8, 8, 8]           1,160 <\b>
             ReLU-26              [-1, 8, 8, 8]               0 <\b>
      BatchNorm2d-27              [-1, 8, 8, 8]              16 <\b>
        Dropout2d-28              [-1, 8, 8, 8]               0 <\b>
           Conv2d-29             [-1, 10, 8, 8]             730 <\b>
             ReLU-30             [-1, 10, 8, 8]               0 <\b>
      BatchNorm2d-31             [-1, 10, 8, 8]              20 <\b>
        Dropout2d-32             [-1, 10, 8, 8]               0 <\b>
AdaptiveAvgPool2d-33             [-1, 10, 1, 1]               0 <\b>
           Conv2d-34             [-1, 10, 1, 1]             110 <\b>


================================================================
Total params: 50,260
Trainable params: 50,260
Non-trainable params: 0

----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.96
Params size (MB): 0.19
Estimated Total Size (MB): 4.16

----------------------------------------------------------------


** TRAIN ACCURACY:  72.792 TRAIN LOSS:  0.8170133829116821 **
** TEST ACCURACY:  74.02 TEST LOSS:  0.7378784431934357  **

Following are the sample images of the test dataset:
<img src="https://github.com/saurabhmangal/era_s8/blob/master/batch_norm.jpg" alt="alt text" width="600px">

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------

**PARAMETERS FOR BATCH Group Normalization ARCHITECTURE**

----------------------------------------------------------------
        Layer (type)               Output Shape         Param # <\b>
        
================================================================
            Conv2d-1           [-1, 48, 32, 32]           1,344 <\b>
              ReLU-2           [-1, 48, 32, 32]               0 <\b>
         GroupNorm-3           [-1, 48, 32, 32]              96 <\b>
         Dropout2d-4           [-1, 48, 32, 32]               0 <\b>
            Conv2d-5           [-1, 48, 32, 32]          20,784 <\b>
              ReLU-6           [-1, 48, 32, 32]               0 <\b>
         GroupNorm-7           [-1, 48, 32, 32]              96 <\b>
         Dropout2d-8           [-1, 48, 32, 32]               0 <\b>
            Conv2d-9           [-1, 32, 32, 32]           1,568 <\b>
        MaxPool2d-10           [-1, 32, 16, 16]               0 <\b>
           Conv2d-11           [-1, 32, 16, 16]           9,248 <\b>
             ReLU-12           [-1, 32, 16, 16]               0 <\b>
        GroupNorm-13           [-1, 32, 16, 16]              64 <\b>
        Dropout2d-14           [-1, 32, 16, 16]               0 <\b>
           Conv2d-15           [-1, 32, 16, 16]           9,248 <\b>
             ReLU-16           [-1, 32, 16, 16]               0 <\b>
        GroupNorm-17           [-1, 32, 16, 16]              64 <\b>
        Dropout2d-18           [-1, 32, 16, 16]               0 <\b>
           Conv2d-19           [-1, 32, 16, 16]           1,056 <\b>
        MaxPool2d-20             [-1, 32, 8, 8]               0 <\b>
           Conv2d-21             [-1, 16, 8, 8]           4,624 <\b>
             ReLU-22             [-1, 16, 8, 8]               0 <\b>
        GroupNorm-23             [-1, 16, 8, 8]              32 <\b>
        Dropout2d-24             [-1, 16, 8, 8]               0 <\b>
           Conv2d-25              [-1, 8, 8, 8]           1,160 <\b>
             ReLU-26              [-1, 8, 8, 8]               0 <\b>
        GroupNorm-27              [-1, 8, 8, 8]              16 <\b>
        Dropout2d-28              [-1, 8, 8, 8]               0 <\b>
           Conv2d-29             [-1, 10, 8, 8]             730 <\b>
             ReLU-30             [-1, 10, 8, 8]               0 <\b>
        GroupNorm-31             [-1, 10, 8, 8]              20 <\b>
        Dropout2d-32             [-1, 10, 8, 8]               0 <\b>
AdaptiveAvgPool2d-33             [-1, 10, 1, 1]               0 <\b>
           Conv2d-34             [-1, 10, 1, 1]             110 <\b>
           
================================================================
Total params: 50,260
Trainable params: 50,260
Non-trainable params: 0

----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.96
Params size (MB): 0.19
Estimated Total Size (MB): 4.16

----------------------------------------------------------------

** TRAIN ACCURACY:  71.612 TRAIN LOSS:  0.4181869626045227 **
** TEST ACCURACY:  71.23 TEST LOSS:  0.8142993453979492 **


Following are the sample imagese of the test dataset:
<img src="https://github.com/saurabhmangal/era_s8/blob/master/group_norm.jpg" alt="alt text" width="600px">

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------




**PARAMETERS FOR LAYER NORMALIZARTION ARCHITECTURE**

----------------------------------------------------------------
        Layer (type)               Output Shape         Param # <\b>
        
================================================================
            Conv2d-1           [-1, 10, 32, 32]             280 <\b>
              ReLU-2           [-1, 10, 32, 32]               0 <\b>
         LayerNorm-3           [-1, 10, 32, 32]          20,480 <\b>
         Dropout2d-4           [-1, 10, 32, 32]               0 <\b>
            Conv2d-5            [-1, 8, 32, 32]             728 <\b>
              ReLU-6            [-1, 8, 32, 32]               0 <\b>
         LayerNorm-7            [-1, 8, 32, 32]          16,384 <\b>
         Dropout2d-8            [-1, 8, 32, 32]               0 <\b>
            Conv2d-9           [-1, 16, 32, 32]             144 <\b>
        MaxPool2d-10           [-1, 16, 16, 16]               0 <\b>
           Conv2d-11            [-1, 8, 16, 16]           1,160 <\b>
             ReLU-12            [-1, 8, 16, 16]               0 <\b>
        LayerNorm-13            [-1, 8, 16, 16]           4,096 <\b>
        Dropout2d-14            [-1, 8, 16, 16]               0 <\b>
           Conv2d-15           [-1, 16, 16, 16]             144 <\b>
        MaxPool2d-16             [-1, 16, 8, 8]               0 <\b>
           Conv2d-17              [-1, 8, 8, 8]           1,160 <\b>
             ReLU-18              [-1, 8, 8, 8]               0 <\b>
        LayerNorm-19              [-1, 8, 8, 8]           1,024 <\b>
        Dropout2d-20              [-1, 8, 8, 8]               0 <\b>
           Conv2d-21              [-1, 4, 8, 8]             292 <\b>
             ReLU-22              [-1, 4, 8, 8]               0 <\b>
        LayerNorm-23              [-1, 4, 8, 8]             512 <\b>
        Dropout2d-24              [-1, 4, 8, 8]               0 <\b>
           Conv2d-25             [-1, 10, 8, 8]             370 <\b>
             ReLU-26             [-1, 10, 8, 8]               0 <\b>
        LayerNorm-27             [-1, 10, 8, 8]           1,280 <\b>
        Dropout2d-28             [-1, 10, 8, 8]               0 <\b>
AdaptiveAvgPool2d-29             [-1, 10, 1, 1]               0 <\b>
           Conv2d-30             [-1, 10, 1, 1]             110 <\b>
           
================================================================
Total params: 48,164
Trainable params: 48,164
Non-trainable params: 0

----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.86
Params size (MB): 0.18
Estimated Total Size (MB): 1.06

----------------------------------------------------------------


** TRAIN ACCURACY:  53.694 TRAIN LOSS:  1.3298343420028687 **
** TEST ACCURACY:  55.56 TEST LOSS:  1.2183823497772217   **

Following are the sample imagese of the test dataset:
<img src="https://github.com/saurabhmangal/era_s8/blob/master/group_norm.jpg" alt="alt text" width="600px">

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------

<!-- #endregion -->

**PARAMETERS FOR BATCH NORMALIZARTION ARCHITECTURE**

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
        
================================================================
            Conv2d-1           [-1, 48, 32, 32]           1,344
              ReLU-2           [-1, 48, 32, 32]               0
       BatchNorm2d-3           [-1, 48, 32, 32]              96
         Dropout2d-4           [-1, 48, 32, 32]               0
            Conv2d-5           [-1, 48, 32, 32]          20,784
              ReLU-6           [-1, 48, 32, 32]               0
       BatchNorm2d-7           [-1, 48, 32, 32]              96
         Dropout2d-8           [-1, 48, 32, 32]               0
            Conv2d-9           [-1, 32, 32, 32]           1,568
        MaxPool2d-10           [-1, 32, 16, 16]               0
           Conv2d-11           [-1, 32, 16, 16]           9,248
             ReLU-12           [-1, 32, 16, 16]               0
      BatchNorm2d-13           [-1, 32, 16, 16]              64
        Dropout2d-14           [-1, 32, 16, 16]               0
           Conv2d-15           [-1, 32, 16, 16]           9,248
             ReLU-16           [-1, 32, 16, 16]               0
      BatchNorm2d-17           [-1, 32, 16, 16]              64
        Dropout2d-18           [-1, 32, 16, 16]               0
           Conv2d-19           [-1, 32, 16, 16]           1,056
        MaxPool2d-20             [-1, 32, 8, 8]               0
           Conv2d-21             [-1, 16, 8, 8]           4,624
             ReLU-22             [-1, 16, 8, 8]               0
      BatchNorm2d-23             [-1, 16, 8, 8]              32
        Dropout2d-24             [-1, 16, 8, 8]               0
           Conv2d-25              [-1, 8, 8, 8]           1,160
             ReLU-26              [-1, 8, 8, 8]               0
      BatchNorm2d-27              [-1, 8, 8, 8]              16
        Dropout2d-28              [-1, 8, 8, 8]               0
           Conv2d-29             [-1, 10, 8, 8]             730
             ReLU-30             [-1, 10, 8, 8]               0
      BatchNorm2d-31             [-1, 10, 8, 8]              20
        Dropout2d-32             [-1, 10, 8, 8]               0
AdaptiveAvgPool2d-33             [-1, 10, 1, 1]               0
           Conv2d-34             [-1, 10, 1, 1]             110


================================================================
Total params: 50,260
Trainable params: 50,260
Non-trainable params: 0

----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.96
Params size (MB): 0.19
Estimated Total Size (MB): 4.16

----------------------------------------------------------------


** TRAIN ACCURACY:  72.792 TRAIN LOSS:  0.8170133829116821 **
** TEST ACCURACY:  74.02 TEST LOSS:  0.7378784431934357  **

Following are the sample imagese of the test dataset:
<img src="https://github.com/saurabhmangal/era1_s6/blob/master/E_total_vs_Learning_rate.png" alt="alt text" width="600px">

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------

**PARAMETERS FOR BATCH Group Normalization ARCHITECTURE**

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
        
================================================================
            Conv2d-1           [-1, 48, 32, 32]           1,344
              ReLU-2           [-1, 48, 32, 32]               0
         GroupNorm-3           [-1, 48, 32, 32]              96
         Dropout2d-4           [-1, 48, 32, 32]               0
            Conv2d-5           [-1, 48, 32, 32]          20,784
              ReLU-6           [-1, 48, 32, 32]               0
         GroupNorm-7           [-1, 48, 32, 32]              96
         Dropout2d-8           [-1, 48, 32, 32]               0
            Conv2d-9           [-1, 32, 32, 32]           1,568
        MaxPool2d-10           [-1, 32, 16, 16]               0
           Conv2d-11           [-1, 32, 16, 16]           9,248
             ReLU-12           [-1, 32, 16, 16]               0
        GroupNorm-13           [-1, 32, 16, 16]              64
        Dropout2d-14           [-1, 32, 16, 16]               0
           Conv2d-15           [-1, 32, 16, 16]           9,248
             ReLU-16           [-1, 32, 16, 16]               0
        GroupNorm-17           [-1, 32, 16, 16]              64
        Dropout2d-18           [-1, 32, 16, 16]               0
           Conv2d-19           [-1, 32, 16, 16]           1,056
        MaxPool2d-20             [-1, 32, 8, 8]               0
           Conv2d-21             [-1, 16, 8, 8]           4,624
             ReLU-22             [-1, 16, 8, 8]               0
        GroupNorm-23             [-1, 16, 8, 8]              32
        Dropout2d-24             [-1, 16, 8, 8]               0
           Conv2d-25              [-1, 8, 8, 8]           1,160
             ReLU-26              [-1, 8, 8, 8]               0
        GroupNorm-27              [-1, 8, 8, 8]              16
        Dropout2d-28              [-1, 8, 8, 8]               0
           Conv2d-29             [-1, 10, 8, 8]             730
             ReLU-30             [-1, 10, 8, 8]               0
        GroupNorm-31             [-1, 10, 8, 8]              20
        Dropout2d-32             [-1, 10, 8, 8]               0
AdaptiveAvgPool2d-33             [-1, 10, 1, 1]               0
           Conv2d-34             [-1, 10, 1, 1]             110
           
================================================================
Total params: 50,260
Trainable params: 50,260
Non-trainable params: 0

----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.96
Params size (MB): 0.19
Estimated Total Size (MB): 4.16

----------------------------------------------------------------

** TRAIN ACCURACY:  71.612 TRAIN LOSS:  0.4181869626045227 **
** TEST ACCURACY:  71.23 TEST LOSS:  0.8142993453979492 **


Following are the sample imagese of the test dataset:
<img src="https://github.com/saurabhmangal/era1_s6/blob/master/E_total_vs_Learning_rate.png" alt="alt text" width="600px">

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------




**PARAMETERS FOR LAYER NORMALIZARTION ARCHITECTURE**

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
        
================================================================
            Conv2d-1           [-1, 10, 32, 32]             280
              ReLU-2           [-1, 10, 32, 32]               0
         LayerNorm-3           [-1, 10, 32, 32]          20,480
         Dropout2d-4           [-1, 10, 32, 32]               0
            Conv2d-5            [-1, 8, 32, 32]             728
              ReLU-6            [-1, 8, 32, 32]               0
         LayerNorm-7            [-1, 8, 32, 32]          16,384
         Dropout2d-8            [-1, 8, 32, 32]               0
            Conv2d-9           [-1, 16, 32, 32]             144
        MaxPool2d-10           [-1, 16, 16, 16]               0
           Conv2d-11            [-1, 8, 16, 16]           1,160
             ReLU-12            [-1, 8, 16, 16]               0
        LayerNorm-13            [-1, 8, 16, 16]           4,096
        Dropout2d-14            [-1, 8, 16, 16]               0
           Conv2d-15           [-1, 16, 16, 16]             144
        MaxPool2d-16             [-1, 16, 8, 8]               0
           Conv2d-17              [-1, 8, 8, 8]           1,160
             ReLU-18              [-1, 8, 8, 8]               0
        LayerNorm-19              [-1, 8, 8, 8]           1,024
        Dropout2d-20              [-1, 8, 8, 8]               0
           Conv2d-21              [-1, 4, 8, 8]             292
             ReLU-22              [-1, 4, 8, 8]               0
        LayerNorm-23              [-1, 4, 8, 8]             512
        Dropout2d-24              [-1, 4, 8, 8]               0
           Conv2d-25             [-1, 10, 8, 8]             370
             ReLU-26             [-1, 10, 8, 8]               0
        LayerNorm-27             [-1, 10, 8, 8]           1,280
        Dropout2d-28             [-1, 10, 8, 8]               0
AdaptiveAvgPool2d-29             [-1, 10, 1, 1]               0
           Conv2d-30             [-1, 10, 1, 1]             110
           
================================================================
Total params: 48,164
Trainable params: 48,164
Non-trainable params: 0

----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.86
Params size (MB): 0.18
Estimated Total Size (MB): 1.06

----------------------------------------------------------------


** TRAIN ACCURACY:  53.694 TRAIN LOSS:  1.3298343420028687 **
** TEST ACCURACY:  55.56 TEST LOSS:  1.2183823497772217   **

Following are the sample imagese of the test dataset:
<img src="https://github.com/saurabhmangal/era1_s6/blob/master/E_total_vs_Learning_rate.png" alt="alt text" width="600px">

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
