# -lam 0.01 -mu 0.1 for 4-layer CNN and 3-layer MLP
# -lam 0.0001 -mu 0.0 for ResNet-18 and fastText
# -lam 0.01 -mu 1.0 for HAR-CNN
nohup python -u main.py -t 1 -jr 1 -nc 20 -nb 10 -data mnist-0.1-npz -m cnn -algo GPFL -did 6 -lam 0.01 -mu 0.1 > result-mnist-0.1-npz.out 2>&1 & 