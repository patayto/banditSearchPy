#!/usr/bin/python2.7
# Map1.py

import sys
import itertools

def main(numberOfArms, stepSize):
	rangeMin = 0.0
	rangeMax = 1.0

	generateBandits(rangeMin, rangeMax, stepSize, numberOfArms)


def arange(rangeMin, rangeMax, stepSize):
	probRange = []
	currProb = rangeMin
	while (currProb <= rangeMax):
		probRange.append(currProb)
		currProb = round(currProb + stepSize, 3)
	return probRange


def generateBandits(rangeMin, rangeMax, stepSize, numberOfArms):
	probRange = arange(rangeMin, rangeMax, stepSize)
	for p in itertools.product(probRange, repeat=numberOfArms):
		print(p)


if __name__ == '__main__':
	numberOfArms = int(sys.argv[1])
	stepSize = float(sys.argv[2])
	main(numberOfArms, stepSize)