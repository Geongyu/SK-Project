# SK-Project Brain Hemorrhage Segmentation 


## Information
### Measures
- Accuracy
- DICE Coefficient
- Jaccard Coefficient (IoU)
- Slice-Level-Accuracy (For Segmentation 2D)
  If model found Hemorrhage Pixel(at least 1 pixels), then get more accuracy

### Data Description 
- SK Dataset
- Kaggle Dataset (RSNA Datasets)

### Logger
- Train Logger       : epoch, loss, IoU, Dice, Slice-Level-Accuracy
- Test Logger        : epoch, loss, IoU, Dice, Slice-Level-Accuracy

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
* python main.py --save_path ./resnet_cifar10/entropy/ --model res110 --data cifar10 --rank_target entropy --epochs 300 --scheduler 1 --gpu_id 0
* python main.py --save_path ./densenet_BC_cifar10/entropy/ --model densenet_BC --data cifar10 --rank_target entropy --epochs 200 --scheduler 2 --gpu_id 0
```
python main.py \
--epochs 200 \
--batch_size 4 * n (n is Number of GPUs) \
--scheduler [100 150 200] \
--lr 0.1 \
--weight_decay 0.0001 \
--momentum 0.9 \
--nesterov False \
--gpu_id 0,1,2,3 \
--model Unet \
--data sk-datasets \
--save_path ./res110_cifar10_softmax/
```
| Args 	| Options 	| Description 	|
|---------|--------|----------------------------------------------------|
| dataset 	| cifar10, cifar100, <br>svhn 	| dataset. 	|
| model 	| res110, densenet_BC, mobil, vgg16	| model architecture : res110(Pre_Act), densenet_BC(d=100,k=12), mobil(V2), vgg16(bn) 	|
| rank_target 	| softmax, <br>entropy, <br>margin 	| rank target. 	|
| rank_weight 	| [float] 	| rank_weight. defalut : 1.0	ensemble : 0.5|
| batch_size 	| [int] 	| Number of samples per batch. defalut : 128|
| epochs 	| [int] 	| Number of epochs for training. defalut : 300|
| scheduler 	| 1, 2	| 1.[150, 250] epoch decay 0.1, / 2.consine_lr 	defalut : 1|
| learning_rate 	| [float] 	| Learning rate. defalut : 0.1	|
| gpu_id 	| [str] 	| Learning rate. defalut : 0	|
| save_path 	| [str] 	| ./res110_cifar10_softmax/	|

### Segmentation - 2D Train Examples
* python main.py --save_path ./resnet_cifar10/entropy/ --model res110 --data cifar10 --rank_target entropy --epochs 300 --scheduler 1 --gpu_id 0
* python main.py --save_path ./densenet_BC_cifar10/entropy/ --model densenet_BC --data cifar10 --rank_target entropy --epochs 200 --scheduler 2 --gpu_id 0
```
python main.py \
--epochs 300 \
--batch_size 128 \
--scheduler 1 \
--lr 0.1 \
--weight_decay 0.0001 \
--momentum 0.9 \
--nesterov False \
--gpu_id 0 \
--model res110 \
--data cifar10 \
--rank_target softmax \
--rank_weight 1.0 \
--save_path ./res110_cifar10_softmax/
```
| Args 	| Options 	| Description 	|
|---------|--------|----------------------------------------------------|
| dataset 	| cifar10, cifar100, <br>svhn 	| dataset. 	|
| model 	| res110, densenet_BC, mobil, vgg16	| model architecture : res110(Pre_Act), densenet_BC(d=100,k=12), mobil(V2), vgg16(bn) 	|
| rank_target 	| softmax, <br>entropy, <br>margin 	| rank target. 	|
| rank_weight 	| [float] 	| rank_weight. defalut : 1.0	ensemble : 0.5|
| batch_size 	| [int] 	| Number of samples per batch. defalut : 128|
| epochs 	| [int] 	| Number of epochs for training. defalut : 300|
| scheduler 	| 1, 2	| 1.[150, 250] epoch decay 0.1, / 2.consine_lr 	defalut : 1|
| learning_rate 	| [float] 	| Learning rate. defalut : 0.1	|
| gpu_id 	| [str] 	| Learning rate. defalut : 0	|
| save_path 	| [str] 	| ./res110_cifar10_softmax/	|

### Segmentation - 3D Train Examples
* python main.py --save_path ./resnet_cifar10/entropy/ --model res110 --data cifar10 --rank_target entropy --epochs 300 --scheduler 1 --gpu_id 0
* python main.py --save_path ./densenet_BC_cifar10/entropy/ --model densenet_BC --data cifar10 --rank_target entropy --epochs 200 --scheduler 2 --gpu_id 0
```
python main.py \
--epochs 300 \
--batch_size 128 \
--scheduler 1 \
--lr 0.1 \
--weight_decay 0.0001 \
--momentum 0.9 \
--nesterov False \
--gpu_id 0 \
--model res110 \
--data cifar10 \
--rank_target softmax \
--rank_weight 1.0 \
--save_path ./res110_cifar10_softmax/
```
| Args 	| Options 	| Description 	|
|---------|--------|----------------------------------------------------|
| dataset 	| cifar10, cifar100, <br>svhn 	| dataset. 	|
| model 	| res110, densenet_BC, mobil, vgg16	| model architecture : res110(Pre_Act), densenet_BC(d=100,k=12), mobil(V2), vgg16(bn) 	|
| rank_target 	| softmax, <br>entropy, <br>margin 	| rank target. 	|
| rank_weight 	| [float] 	| rank_weight. defalut : 1.0	ensemble : 0.5|
| batch_size 	| [int] 	| Number of samples per batch. defalut : 128|
| epochs 	| [int] 	| Number of epochs for training. defalut : 300|
| scheduler 	| 1, 2	| 1.[150, 250] epoch decay 0.1, / 2.consine_lr 	defalut : 1|
| learning_rate 	| [float] 	| Learning rate. defalut : 0.1	|
| gpu_id 	| [str] 	| Learning rate. defalut : 0	|
| save_path 	| [str] 	| ./res110_cifar10_softmax/	|


### Reference
[1] U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberrger et al.) \
[2] 3D U-Net: learning dense volumetric segmentation from sparse annotation (Abdulkadir et al.) \
[3] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (Tan et al.) \
[4] A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation (Abraham and Khan) \
[5] Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Network (Roy et al.) \
[6] An intriguring falling of convolutional neural networks and the coordconv solution (Liu et al.) \






