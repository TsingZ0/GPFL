# Due to the file size limitation of the supplementary material, we only upload the mnist dataset for example. 

nohup python -u main.py -t 1 -jr 1 -nc 20 -nb 10 -data mnist-0.1-npz -m cnn -algo GPFL -did 6 -lam 0.01 -lamr 0.1 > result-mnist-0.1-npz.out 2>&1 &