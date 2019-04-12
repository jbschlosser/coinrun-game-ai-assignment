#!/bin/bash
MODEL=$1
EASY=1
MEDIUM=20
MONSTER=10
python main.py --render --eval --load ${MODEL} --seed ${MONSTER} --model_path .
