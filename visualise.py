from numpy import genfromtxt
import matplotlib.pyplot as plt
import argparse

# Command line argument parser
parser = argparse.ArgumentParser(description='CSV Visualisation')
parser.add_argument('--datafile', '-d', required=True, type=str,
                    help='Path to the file containing the data to plot.')
args = parser.parse_args()

filepath = args.datafile

# Plot the given data
data=genfromtxt(filepath, delimiter=',', skip_header=1)
#pylab.plot(x, y1, '-b', label='sine')
plt.plot(data[:,0], label='training loss')
plt.plot(data[:,1], label='validation loss')
#plt.plot(per_data)
plt.xlabel ('Epoch')
plt.ylabel ('Loss')
plt.title('Loss')
plt.legend(loc='upper right')
plt.show()
