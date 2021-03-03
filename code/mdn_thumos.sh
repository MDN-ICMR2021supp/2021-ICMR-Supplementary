#!/usr/bin/env bash
set -ex

echo "$(date "+%Y.%m.%d-%H.%M.%S")"
python mdn_train.py  
python mdn_inference.py 
python mdn_postprocess.py 
echo "$(date "+%Y.%m.%d-%H.%M.%S")"
