from numpy import loadtxt
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import os


# Use grab.sh to first get the results of certify.py out as a table

def draw(file, label):
    table = loadtxt(file, dtype=np.float32, delimiter='\t')
    x = table[:, 0].astype(np.float32)
    y = table[:, 1].astype(np.float32)
    plt.plot(x, y, label=label)
    plt.legend()


def main():
    parse = ArgumentParser()
    parse.add_argument('-f', '--files', nargs='+')
    parse.add_argument('-l', '--labels', nargs='+')
    parse.add_argument('-o', '--output', type=str, default='output')
    parse.add_argument('-d', '--dir', type=str, default='.')

    args = parse.parse_args()

    plt.ylabel('Certificate Accuracy %')
    plt.xlabel('Radius')
    for file, label in zip(args.files, args.labels):
        file = os.path.join(args.dir, file)
        draw(file, label)

    plt.savefig('{}.pdf'.format(args.output))


if __name__ == '__main__':
    main()
