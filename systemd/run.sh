#!/bin/bash
export BIBO_PATH=/home/ubuntu/workspace/bibo

export PATH=/usr/local/cuda/bin:/usr/local/bin:/opt/aws/bin:/home/ubuntu/src/cntk/bin:/usr/local/mpi/bin:$PATH
export LD_LIBRARY_PATH=/home/ubuntu/src/cntk/bindings/python/cntk/libs:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/mpi/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export PYTHONPATH=/home/ubuntu/src/cntk/bindings/python
export PYTHONPATH=/home/ubuntu/workspace/models:$PYTHONPATH
export PYTHONPATH=/home/ubuntu/workspace/models/slim:$PYTHONPATH
export PYTHONPATH=/home/ubuntu/workspace/models/object_detection/protos:$PYTHONPATH

. "$BIBO_PATH/systemd/database.sh"
