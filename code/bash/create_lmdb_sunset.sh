
#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

OUTPUT_TRAIN=/Users/tianchuliang/Documents/GT_Acad/7616Spring16/HWSoln/hw4/data/Sunset/Train
OUTPUT_TEST=/Users/tianchuliang/Documents/GT_Acad/7616Spring16/HWSoln/hw4/data/Sunset/Test

TXTFILES=/Users/tianchuliang/Documents/GT_Acad/7616Spring16/HWSoln/hw4/data/Sunset

TOOLS=/usr/local/caffe/build/tools

DATA_ROOT=/Users/tianchuliang/Documents/GT_Acad/7616Spring16/HWSoln/hw4/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the data is stored."
  exit 1
fi


echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT \
    $TXTFILES/Train/Train.txt \
    $OUTPUT_TRAIN/sunset_train_lmdb

echo "Creating test lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT \
    $TXTFILES/Test/Test.txt \
    $OUTPUT_TEST/sunset_test_lmdb

echo "Done."