#!/bin/bash
EASY=1
MEDIUM=20
MONSTER=10
python main.py --save model.pth --model_path logs_medium --load best_easy.pth --seed ${MEDIUM}
