# SK-Project Brain Hemorrhage Segmentation 

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="92" height="20"><linearGradient id="b" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient><clipPath id="a"><rect width="92" height="20" rx="3" fill="#fff"/></clipPath><g clip-path="url(#a)"><path fill="#555" d="M0 0h69v20H0z"/><path fill="#007ec6" d="M69 0h23v20H69z"/><path fill="url(#b)" d="M0 0h92v20H0z"/></g><g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="110"> <text x="355" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="590">SK-Project</text><text x="355" y="140" transform="scale(.1)" textLength="590">SK-Project</text><text x="795" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="130">v1</text><text x="795" y="140" transform="scale(.1)" textLength="130">v1</text></g> </svg>

## Information
### Measures
- Accuracy
- DICE Coefficient
- Jaccard Coefficient (IoU)
- Slice-Level-Accuracy (For Segmentation 2D,)

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



