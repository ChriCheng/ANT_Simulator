python train_alexnet.py --batch_size=128 --ptq --dataset=cifar10 --model=alexnet --mode=ant-int-pot-flint --wbit=4 --abit=4 
# python train_alexnet.py --batch_size=128 --ptq --dataset=cifar10 --model=alexnet --mode=ant-int-pot-flint --wbit=6 --abit=6 
# python main.py --batch_size=128 --ptq --dataset=cifar10 --model=alexnet --mode=ant-int-pot-flint --wbit=6 --abit=6 
# distributed
# python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46666 main.py --batch_size=128 --ptq --dataset=cifar10 --model=alexnet --mode=ant-int-pot-flint --wbit=6 --abit=6 -wl=100 -al=100
# python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46666 main.py --batch_size=128 --ptq --dataset=imagenet --model=vgg16_bn --mode=ant-int-pot-flint --wbit=6 --abit=6 -wl=100 -al=100
# python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46666 main.py --batch_size=128 --ptq --dataset=imagenet --model=resnet50 --mode=ant-int-pot-flint --wbit=6 --abit=6 -wl=100 -al=100
# python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 46666 main.py --batch_size=128 --ptq --dataset=imagenet --model=resnet152 --mode=ant-int-pot-flint --wbit=6 --abit=6 -wl=100 -al=100