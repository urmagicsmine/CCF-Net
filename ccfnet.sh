#!/bin/bash
if [ "$1" == "train" ]
then
	## res2d 3 slice
	CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/tianchi_configs/faster_rcnn_r50_fpn_ms576_as16_slice3_lesion_1x.py 4 --autoscale-lr --validate
	## res2d msbnet 3 slice
	#CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/tianchi_configs/faster_rcnn_r50_fpn_ms576_as16_slice3_msb_lesion_1x.py 4 --autoscale-lr --validate
	### res3d 7 slice
	#CUDA_VISIBLE_DEVICES=4,5,6,7 bash tools/dist_train.sh configs/tianchi_configs/faster_rcnn_r3d50_fpn_ms576_as16_slice7_lesion_scratch_1x.py 4 --autoscale-lr --validate
	### 3DCE
	#CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/tianchi_configs/faster_rcnn_r50_fpn_ms576_as16_slice9_lesion_1x.py 4 --autoscale-lr --validate
	#CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/tianchi_configs/faster_rcnn_r50_fpn_slice27_lesion_1x.py 4 --autoscale-lr --validate
	### MVP-Net
	#CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/tianchi_configs/MVPNet_r50_fpn_ms576_as16_slice3_lesion_1x.py 4 --autoscale-lr --validate
	#CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/tianchi_configs/MVPNet_r50_fpn_ms576_as16_slice9_lesion_1x.py 4 --autoscale-lr --validate
	### CCF-Net
	#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist_train.sh configs/tianchi_configs/two_stream_ms576_as16_nonlocal_scratch_1x.py 8 --autoscale-lr --validate 
elif [ "$1" == "test" ]
then
	#CUDA_VISIBLE_DEVICES=4,5,6,7 bash tools/dist_test.sh configs/tianchi_configs/faster_rcnn_r50_fpn_ms576_as16_slice3_lesion_1x.py work_dirs/tianchi/faster_rcnn_r50_fpn_ms640_slice3_lesion_1x/latest.pth 4 --eval bbox --out wo.pkl --eval_froc
	CUDA_VISIBLE_DEVICES=4,5,6,7 bash tools/dist_test.sh configs/tianchi_configs/MVPNet_r50_fpn_ms576_as16_slice9_lesion_1x.py work_dirs/tianchi/MVPNet_r50_fpn_ms576_as16_slice9_lesion_1x/latest.pth 4 --eval bbox --out wo.pkl --eval_froc
else
	echo "choose from [train,test]"
fi
