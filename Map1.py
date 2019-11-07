#!/usr/bin/python2.7
# Map1.py

import sys
import itertools
import numpy as np

def main(numberOfArms, stepSize):
	rangeMin = 0.0
	rangeMax = 1.0

	generateBandits(rangeMin, rangeMax, stepSize, numberOfArms)


def generateBandits(rangeMin, rangeMax, stepSize, numberOfArms):
	probRange = np.arange(rangeMin, rangeMax + stepSize, stepSize)
	for p in itertools.product(probRange, repeat=numberOfArms):
		print(p)

if __name__ == '__main__':
	numberOfArms = int(sys.argv[1])
	stepSize = float(sys.argv[2])
	main(numberOfArms, stepSize)