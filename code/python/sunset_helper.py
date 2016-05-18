# Author: Tianchu Liang
'''
This script contains code that generates proper prototxts for sunset
training and testing.
'''
# Define paths and import modules
caffe_root = '/usr/local/caffe/'
code_root='/Users/tianchuliang/documents/gt_acad/7616spring16/hwsoln/hw4/code'
data_root='/Users/tianchuliang/documents/gt_acad/7616spring16/hwsoln/hw4/data'
nets_root='/Users/tianchuliang/documents/gt_acad/7616spring16/hwsoln/hw4/nets'

import sys 
import caffe 
import os 
import numpy as np 
from pylab import *
import tempfile
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
caffe.set_mode_cpu()
sys.path.insert(0,caffe_root+'python')
assert os.path.exists(code_root)
assert os.path.exists(data_root)
assert os.path.exists(nets_root)

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2
sunset_lmdb=data_root+"/Sunset"

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

'''
-----------------------------------------------------------
-----------------------------------------------------------
'''
def caffenet(data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
    n = caffe.NetSpec()
    n.data = data
    param = learned_param if learn_all else frozen_param
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)

    # To make sure Command Line interface training and testing can 
    # go through normally, without producing shape mismatch error, 
    # I had to rename fc6 layer, and make this layer's weights to be
    # learned as well, in addition to the final fc8 layer. 
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=learned_param)
    n.__setattr__("fc6_sunset",n.fc6)

    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    # Similarly, since n.fc6(renamed as fc6_sunset) learns new weights,
    # n.fc7 has to learn new weights as well:
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=learned_param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    # write the net to a temporary file and return its filename
#     with tempfile.NamedTemporaryFile(delete=False) as f:
#         f.write(str(n.to_proto()))
#         return f.name
    if train==True:
        with open('/Users/tianchuliang/documents/gt_acad/7616spring16/hwsoln/hw4/nets/my_sunset_net/my_sunset_net_train_cmdline.prototxt', 'w') as f:
            f.write(str(n.to_proto()))
            return f.name
    else:
        with open('/Users/tianchuliang/documents/gt_acad/7616spring16/hwsoln/hw4/nets/my_sunset_net/my_sunset_net_test_cmdline.prototxt', 'w') as f:
            f.write(str(n.to_proto()))
            return f.name

        
def sunset_net(lmdb, batch_size, train=True, learn_all=False,subset=None,mirror=False,crop_size=256):
    lmdb_mode=dict({'train':'Train','test':'Test'})
    
    if subset is None:
        subset = 'train' if train else 'test'
    
    lmdb=lmdb+"/"+lmdb_mode[subset]+"/"+"sunset_%s_lmdb" % subset
    
    # When mirror is set to true, then random mirroring is turned on during training, this is part of 
    # image augmentation;
    # When crop_size is less than 256, i.e the standard sizes of all training images, then cropping is applied;
    # this cropping is squared cropping. 
    transform_param = dict(mirror = mirror, crop_size=crop_size,
                           mean_file=data_root + '/Sunset/Train/sunset_train_mean.binaryproto')
    
    sunset_data, sunset_label = L.Data(transform_param=transform_param, 
                                       backend=P.Data.LMDB, 
                                       source=lmdb,
                                       batch_size=batch_size,
                                       ntop=2)
    return caffenet(data=sunset_data, label=sunset_label, train=train,
                   num_classes=2,classifier_name='fc8_sunset',learn_all=learn_all)

def solver(train_net_path, test_net_path=None, base_lr = 0.001,displayFreq=20,snapshotFreq=100
    ,test_interval=10,test_iter=50,stepsize=10000):
    s = caffe_pb2.SolverParameter()
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = test_interval
        s.test_iter.append(test_iter)
        
    s.iter_size=1
    s.max_iter=100000
    s.type = 'SGD'
    s.base_lr = base_lr 
    
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize=stepsize
    
    s.momentum = 0.9
    s.weight_decay=5e-4
    s.display=displayFreq
    
    s.snapshot=snapshotFreq
    s.snapshot_prefix=nets_root+"/my_sunset_net/my_sunset_net_cmdline"
    s.solver_mode = caffe_pb2.SolverParameter.CPU
    
#     with tempfile.NamedTemporaryFile(delete=False) as f:
#         f.write(str(s))
#         return f.name
    with open('/Users/tianchuliang/documents/gt_acad/7616spring16/hwsoln/hw4/nets/my_sunset_net/my_sunset_net_solver_cmdline.prototxt', 'w') as f:
        f.write(str(s))
        return f.name
