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
* python3 main.py  --loss-function bce --exp Classification/Classification --optim-function radam --momentum 0.9 --initial-lr 0.0001 --lr-schedule 75 100 --weight-decay 0.0001 --batch-size 24 --tenosrboardwriter Classifications/Unet-Encoder-Classification --model unet --aug False --smooth False

```
python main.py \
--batch_size 4 * n (n is Number of GPUs) \
--scheduler 75 100 \
--initial-lr 0.0001 \
--weight_decay 0.0001 \
--batch-size 24 \
--tenosrboardwriter Classifications/Unet-Encoder-Classification
--momentum 0.9 \
--model Unet \
--data sk-datasets \
--exp Classification/Classification \
--smooth False \
--aug False  
```
| Args 	| Options 	| Description 	|
|---------|--------|----------------------------------------------------|
| trn-root 	|  [str] 	| dataset locations. 	|
| tst-root | [str] | dataset locations. |
| model 	| efficientnet, resnet	| model architecture : efficientnet (defalut : b1 model), Resnet (defalut : 50), default : efficient net	|
| batch_size 	| [int] 	| number of samples per batch. default : 8|
| epochs 	| [int] 	| number of epochs for training. default : 50|
| learning_rate 	| [float] 	| learning rate. defalut : 0.1	|
| exp 	| [str] 	| ./test/	|
| number | [int] | a number of efficient net. default : b1 |
| momentum | [int] | momentum of optimizers. default : 0.9 |
| tenosrboardwriter | [str] | save path of tensor board |
| kaggle | [str] | Kaggle Datasets. If True use Kaggle Datasets, default : False |

### Segmentation - 2D Train Examples
* python3 main.py  --loss-function bce --exp Segmentation/Unet --optim-function radam --momentum 0.9 --initial-lr 0.0001 --lr-schedule 75 100 --weight-decay 0.0001 --batch-size 24 --tenosrboardwriter Segmentation/Unet-Encoder --model unet --aug False --smooth False --coordconv [none] 

```
python main.py \
--batch_size 4 * n (n is Number of GPUs) \
--scheduler 75 100 \
--lr 0.0001 \
--optim-function radam \
--weight_decay 0.0001 \
--momentum 0.9 \
--model Unet \
--data sk-datasets \
--exp ./MTL/Segmentation/Unet/ \
--tenosrboardwriter Segmentation/Unet-Encoder \
--aug False \
--smooth False \
--coordconv [none] 
```
| Args 	| Options 	| Description 	|
|---------|--------|----------------------------------------------------|
| trn-root 	|  [str] 	| dataset locations. 	|
| tst-root | [str] | dataset locations. |
| model 	| unet, unet coordconv, unet scse, unet multiinput	| model architecture : unet base models, default : unet	|
| batch_size 	| [int] 	| number of samples per batch. default : 8|
| epochs 	| [int] 	| number of epochs for training. default : 200|
| scheduler 	| [int]	| 100 170 200 epoch decay 0.1 	defalut : 100 170 200|
| learning_rate 	| [float] 	| learning rate. defalut : 0.1	|
| exp 	| [str] 	| ./test/	|
| momentum | [int] | momentum of optimizers. default : 0.9 |
| tenosrboardwriter | [str] | save path of tensor board |
| coorconv | [list] | the number of coordconv layers. default : [9] |

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
| f-maps 	| [int] |feature map size of encoder layer.  default : [32, 64, 128, 256]	 |
| conv-layer-order 	| [str] | order of layers.  'cr' -> conv + ReLU ,'crg' -> conv + ReLU + groupnorm , 'cbr' -> conv + BN + Relu.   default : cbr	|
| batch-size 	| [int] 	| number of samples per batch. default : 8  |
| input-size 	| [int] 	| size of input patch. default : [48,48,48].   |
| epochs 	| [int] 	| number of epochs for training. default : 200  |
| lr-schedule 	| [int]	| epoch decay 0.1. 	defalut : [20,30,35]  |
| weight-decay 	| [float]	| weight-decay. 	defalut : 0.0005|
| loss-function 	| [str]	| bce ,dice, weight_bce.  defalut : bce  |
| learning_rate 	| [float] 	| learning rate. defalut : 0.1	|
| exp 	| [str] 	| save folder name.  |


### Reference
[1] U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberrger et al.) \
[2] 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation (Abdulkadir et al.) \
[3] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (Tan et al.) \
[4] A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation (Abraham and Khan) \
[5] Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Network (Roy et al.) \
[6] An intriguring falling of convolutional neural networks and the coordconv solution (Liu et al.) \






