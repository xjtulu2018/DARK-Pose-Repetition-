
#！/bin/bash

date
echo 'HRNet with endoding'
python tools/train.py --cfg experiments/coco/hrnet/1/w32_128x96_adam_lr1e-3-DARK.yaml
date
