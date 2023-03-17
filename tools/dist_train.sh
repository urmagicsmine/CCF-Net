##!/usr/bin/env bash

#PYTHON=${PYTHON:-"python"}

#CONFIG=$1
#GPUS=$2

#$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    #--master_port 2253 \
    #$(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2
$PYTHON -m torch.distributed.launch --master_port=$((RANDOM + 10000))  --nproc_per_node=$GPUS \
	$(dirname "$0")/train.py $CONFIG --launcher pytorch --gpus=$GPUS ${@:3}

