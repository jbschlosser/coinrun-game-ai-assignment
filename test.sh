#!/bin/bash
EASY=1
MEDIUM=20
MONSTER=10
python main.py --render --eval --load best_easy.pth --seed ${MEDIUM} --model_path .
