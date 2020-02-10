# SK-Project Brain Hemorrhage Segmentation 


## Information
### Measures
- Accuracy
- DICE Coefficient
- Jaccard Coefficient (IoU)
- Slice-Level-Accuracy (For Segmentation 2D) \
  If model found Hemorrhage Pixel(at least 1 pixels), then get accuracy scores

### Data Description 
- SK Dataset
- Kaggle Dataset (RSNA Datasets) \ 
  Kaggle RSNA Intracranial Hemorrhage Detection (https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) 

### Logger
- Train Logger       : epoch, loss, IoU, Dice, Slice-Level-Accuracy
- Test Logger        : epoch, loss, IoU, Dice, Slice-Level-Accuracy
- Classifications : epoch, loss, accuracy

## Getting Started
### Requirements
- Python3 (3.6.8)
- PyTorch (1.2)
- torchvision (0.4)
- NumPy
- pandas
- matplotlib
- pytorch-efficientnet (Only Use Classificaitions)
- tensorboard (Use Segmentation-2D)

### Classifications - Train Examples
* python3 main.py  --loss-function bce --exp MTL/Classification/Unet-Encoder-Classification --optim-function radam --momentum 0.9 --initial-lr 0.0001 --lr-schedule 75 100 --weight-decay 0.0001 --batch-size 24 --tenosrboardwriter Classifications/Unet-Encoder-Classification --arch unet --aug False --smooth False

```
python main.py \
--epochs 200 \
--batch_size 4 * n (n is Number of GPUs) \
--scheduler [100 150 200] \
--lr 0.1 \
--weight_decay 0.0001 \
--momentum 0.9 \
--model Unet \
--data sk-datasets \
--exp ./MTL/Classification/Unet-Encoder-Classification/
```
| Args 	| Options 	| Description 	|
|---------|--------|----------------------------------------------------|
| trn-root 	|  [str] 	| dataset locations. 	|
| tst-root | [str] | dataset locations. |
| model 	| efficientnet, resnet	| model architecture : efficientnet (defalut : b1 model), Resnet (defalut : 50), default : efficient net	|
| batch_size 	| [int] 	| Number of samples per batch. default : 8|
| epochs 	| [int] 	| Number of epochs for training. default : 200|
| scheduler 	| 1, 2	| 1.[150, 200] epoch decay 0.1, / 2.consine_lr 	defalut : 1|
| learning_rate 	| [float] 	| Learning rate. defalut : 0.1	|
| exp 	| [str] 	| ./test/	|
| number | [int] | A number of Efficient net. default : b1 |
| momentum | [int] | Momentum of Optimizers. default : 0.9 |
| tenosrboardwriter | [str] | save path of tensor board |

### Segmentation - 2D Train Examples
* python3 main.py  --loss-function bce --exp Segmentation/Unet-Encoder --optim-function radam --momentum 0.9 --initial-lr 0.0001 --lr-schedule 75 100 --weight-decay 0.0001 --batch-size 24 --tenosrboardwriter Segmentation/Unet-Encoder --arch unet --aug False --smooth False --coordconv [none] 

```
python main.py \
--epochs 200 \
--batch_size 4 * n (n is Number of GPUs) \
--scheduler [100 150 200] \
--lr 0.1 \
--weight_decay 0.0001 \
--momentum 0.9 \
--model Unet \
--data sk-datasets \
--exp ./MTL/Classification/Unet-Encoder-Classification/
```
| Args 	| Options 	| Description 	|
|---------|--------|----------------------------------------------------|
| trn-root 	|  [str] 	| dataset locations. 	|
| tst-root | [str] | dataset locations. |
| model 	| unet, unet coordconv, unet scse, unet multiinput	| model architecture : unet base models, default : unet	|
| batch_size 	| [int] 	| Number of samples per batch. default : 8|
| epochs 	| [int] 	| Number of epochs for training. default : 200|
| scheduler 	| [int]	| 100 170 200 epoch decay 0.1 	defalut : 100 170 200|
| learning_rate 	| [float] 	| Learning rate. defalut : 0.1	|
| exp 	| [str] 	| ./test/	|
| momentum | [int] | Momentum of Optimizers. default : 0.9 |
| tenosrboardwriter | [str] | save path of tensor board |
| coorconv | [list] | the Number of coordconv layers. default : [9] |

### Segmentation - 3D Train Examples
* python main.py --trn-root /data1/JM/sk_project/data2th_trainvalid_3d_patches_48_48_48_st_16_bg_0.1_nonzero_0.1 --val-root /data1/JM/sk_project/data2th_test_3d_patches_48_48_48_st_16_bg_1_nonzero_0.1 --input-size 48 48 48 --lr-schedule 20 30 --lr 0.01 --weight-decay 0.0005 --gamma 0.1 --f-maps 32 64 128 256 --stride-test 16 --exp jm_test1 --batch-size 128 --loss-function dice

```
python main.py \
--trn-root /data1/JM/sk_project/data2th_trainvalid_3d_patches_48_48_48_st_16_bg_0.1_nonzero_0.1 \
--val-root /data1/JM/sk_project/data2th_test_3d_patches_48_48_48_st_16_bg_1_nonzero_0.1 \
--input-size 48 48 48 \
--lr-schedule 20 30 \
--lr 0.01 \
--weight-decay 0.0005 \
--gamma 0.1 \
--f-maps 32 64 128 256 \
--stride-test 16 \
--exp exp_sk-project-3d \
--batch-size 128 \
--loss-function dice
```

| Args 	| Options 	| Description 	|
|:---------|:--------|:----------------------------------------------------|
| trn-root 	|  [str] 	| dataset locations. 	|
| tst-root | [str] | dataset locations. |
| model 	| [str] | model architecture.  default : unet |
| f-maps 	| [int] |Feature map size of encoder layer.  default : [32, 64, 128, 256]	|
| conv-layer-order 	| [str] | Order of layers. cbr -> conv + BN + Relu. default : cbr	|
| batch-size 	| [int] 	| Number of samples per batch. default : 8  |
| input-size 	| [int] 	| Size of input patch. default : [48,48,48].   |
| epochs 	| [int] 	| Number of epochs for training. default : 200  |
| lr-schedule 	| [int]	| epoch decay 0.1. 	defalut : [20,30,35]  |
| weight-decay 	| [float]	| weight-decay. 	defalut : 0.0005|
| loss-function 	| [str]	| bce ,dice, weight_bce.  defalut : bce  |
| learning_rate 	| [float] 	| Learning rate. defalut : 0.1	|
| exp 	| [str] 	| save folder name.  |


### Reference
[1] U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberrger et al.) \
[2] 3D U-Net: learning dense volumetric segmentation from sparse annotation (Abdulkadir et al.) \
[3] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (Tan et al.) \
[4] A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation (Abraham and Khan) \
[5] Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Network (Roy et al.) \
[6] An intriguring falling of convolutional neural networks and the coordconv solution (Liu et al.) \






