cd ../
python tools/train.py \
	--cfg experiments/coco/resnet/res34_256x192_d256x3_adam_lr1e-3.yaml

python tools/train.py \
	--cfg experiments/coco/resnet/res34_384x288_d256x3_adam_lr1e-3.yaml
