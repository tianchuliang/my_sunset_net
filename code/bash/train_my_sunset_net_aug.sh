#!/usr/bin/env sh

/usr/local/caffe/build/tools/caffe train -solver /Users/tianchuliang/Documents/GT_Acad/7616Spring16/HWSoln/hw4/nets/my_sunset_net/my_sunset_net_solver_aug_cmdline.prototxt -weights /Users/tianchuliang/documents/gt_acad/7616Spring16/HWSoln/hw4/nets/caffenet/caffenet.caffemodel 2>&1 | tee  /Users/tianchuliang/documents/gt_acad/7616Spring16/HWSoln/hw4/nets/my_sunset_net/my_sunset_net_aug_cmdline.log

