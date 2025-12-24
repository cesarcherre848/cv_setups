#!/bin/bash
################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

# This script downloads NVIDIA TAO Toolkit pre-trained models available on NGC
# and converts them to TRT engine files for adding to Triton Inference Server
# model repository.
# If required the script downloads the tao-converter tool and installs it in
# $HOME/bin/tao-converter directory.
# While generating the models, batch size of 16 is used for dGPU and 1 for Jetson
# platforms.

set -e

TRITON_REPO_DIR="triton_tao_model_repo"
DOWNLOAD_DIR="/tmp/tao_models"
export PATH=$PATH:/usr/src/tensorrt/bin

if [[ $(uname -m) == "aarch64" ]]; then
  BATCH_SIZE=1
else
  BATCH_SIZE=16
fi


echo "Using tao-converter utility from this location: $TAO_CONVERTER_BIN"
echo "Downloading PeopleNet Transformer model from NGC TAO repository"

MODEL_REPO_DIR=$TRITON_REPO_DIR/peoplenet_transformer
MODEL_DOWNLOAD_DIR=$DOWNLOAD_DIR/peoplenet_transformer

mkdir -p $MODEL_DOWNLOAD_DIR

wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet_transformer/deployable_v1.1/files?redirect=true&path=resnet50_peoplenet_transformer_op17.onnx' \
 -O $MODEL_DOWNLOAD_DIR/resnet50_peoplenet_transformer_op17.onnx
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet_transformer/versions/deployable_v1.0/files/labels.txt -O $MODEL_REPO_DIR/labels.txt

echo "Creating TensorRT engine file for PeopleNet Transformer"
echo "Setting batch size to $BATCH_SIZE"
CONFIG_FILE=$MODEL_REPO_DIR/config.pbtxt
sed -i -e "s|max_batch_size.*|max_batch_size: $BATCH_SIZE|" $CONFIG_FILE

mkdir -p $MODEL_REPO_DIR/1

trtexec --onnx=$MODEL_DOWNLOAD_DIR/resnet50_peoplenet_transformer_op17.onnx --fp16 \
 --saveEngine=$MODEL_REPO_DIR/1/model.plan \
 --minShapes="inputs":1x3x544x960 --optShapes="inputs":${BATCH_SIZE}x3x544x960 --maxShapes="inputs":${BATCH_SIZE}x3x544x960
echo "Downloading PeopleSemSegNet Shuffle model from NGC TAO repository"

MODEL_REPO_DIR=$TRITON_REPO_DIR/peoplesemsegnet_shuffle
MODEL_DOWNLOAD_DIR=$DOWNLOAD_DIR/peoplesemsegnet_shuffle

mkdir -p $MODEL_DOWNLOAD_DIR

wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplesemsegnet/deployable_shuffleseg_unet_onnx_v1.0.1/files?redirect=true&path=labels.txt' -O $MODEL_REPO_DIR/labels.txt && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplesemsegnet/deployable_shuffleseg_unet_onnx_v1.0.1/files?redirect=true&path=peoplesemsegnet_shuffleseg.onnx' -O $MODEL_DOWNLOAD_DIR/peoplesemsegnet_shuffleseg.onnx && \
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplesemsegnet/deployable_shuffleseg_unet_onnx_v1.0.1/files?redirect=true&path=peoplesemsegnet_shuffleseg_int8.txt' -O $MODEL_DOWNLOAD_DIR/peoplesemsegnet_shuffleseg_cache.txt

echo "Setting batch size to $BATCH_SIZE"
CONFIG_FILE=$MODEL_REPO_DIR/config.pbtxt
sed -i -e "s|max_batch_size.*|max_batch_size: $BATCH_SIZE|" $CONFIG_FILE

mkdir -p $MODEL_REPO_DIR/1

trtexec --onnx=$MODEL_DOWNLOAD_DIR/peoplesemsegnet_shuffleseg.onnx --int8 \
 --calib=$MODEL_DOWNLOAD_DIR/peoplesemsegnet_shuffleseg_cache.txt --saveEngine=$MODEL_REPO_DIR/1/model.plan \
 --minShapes="input_2:0":1x3x544x960 --optShapes="input_2:0":${BATCH_SIZE}x3x544x960 --maxShapes="input_2:0":${BATCH_SIZE}x3x544x960

echo "Deleting the downloaded model files in $DOWNLOAD_DIR."
rm -rf $DOWNLOAD_DIR

echo "Model repository prepared successfully."
