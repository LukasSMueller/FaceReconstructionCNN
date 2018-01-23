import os, sys

learning_rates = [1e-5]
epochs = 200
base_cmd = 'python3.5 train.py -d datasets/test/blurred_4 -t datasets/test/originals -b 2'
cmd = base_cmd + ' -e ' + str(epochs)

for lr in range(len(learning_rates)):
    cmd = cmd + ' -l ' + str(learning_rates[lr])
    cmd = cdm + '--log ' + 'lr-' + str(learning_rates[lr])
    print(cmd)
    os.system(cmd)
