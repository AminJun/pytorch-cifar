#!/bin/bash
CUDA_VISIBLE_DEVICES="${1}" python main.py --fast 1 > fast.out & 
CUDA_VISIBLE_DEVICES="${2}" python main.py --fast 0 > boring.out &
