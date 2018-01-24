import os, sys

learning_rates = [1e-5]
epochs = 5
base_cmd = 'python train.py -d datasets/test/blurred_4 -t datasets/test/originals -b 2'
cmd = base_cmd + ' -e ' + str(epochs)

for lr in range(len(learning_rates)):
    cmd = cmd + ' -l ' + str(learning_rates[lr])
    cmd = cmd + ' --log ' + 'lr-' + str(learning_rates[lr])
    print(cmd)
    os.system(cmd)
