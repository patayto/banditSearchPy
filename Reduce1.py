#!/usr/bin/python2.7
# Reduce1.py
import sys
import re
import random
from Models import *

modelNames = {
    0: "Random",
    1: "Optimal",
    2: "Win-Stay Lose-Shift",
    3: "Success Ratio",
    4: "Pi-First",
    5: "Epsilon-Greedy",
    6: "Epsilon-Decreasing",
    7: "Extended Win-Stay Lose-Shift",
    8: "Tau-Switch",
    9: "Latent State"
}

def rankModels(modelScores):
	gameScore = list(enumerate(modelScores)) # [(model, score)]

	# shift indices due to exclusion of optimal model
	gameScore = [(ind + 1 if ind >=1 else ind, score) for (ind, score) in gameScore]

	maxScore = max(modelScores)

	rankedModels = sorted([(modelNames[ind], 1.0 if maxScore == 0.0 else score/maxScore, score) for (ind, score) in gameScore], key=lambda a: a[1], reverse=True)

	return rankedModels


def parseFloat(word):
	return float(re.sub(r'[\( \)]', '', word))


def getModelScores(numArms, hor, probs, numSamples, numModels):

	avgScores = []

	for i in range(numModels):
		totalScore = 0
		if (i != 1):
			for n in range(numSamples):
				seed = (hor, n).__hash__() % ((2**32 - 1))
				random.seed(seed)

				model = initialiseModel(i, numArms, hor, 1, seed)

				model.init()

				model.run(probs)

				# get the number of successes for this game
				totalScore += sum([s for (s,f) in model.realHistory.events])

			avgScores.append(totalScore/(numSamples*1.0))

	return avgScores


def initialiseModel(mode, numArms, hor, numGames, seed):
	model = Guessing(numArms, hor, numGames, seed)
	if mode == 0:
		# random
		model = Guessing(numArms, hor, numGames, seed)
		model.init()
	elif mode == 1:
		# optimal, don't have to worry about it
		return -1
	elif mode == 2:
		# WSLS
		lambd = 1.0
		model = WSLS(numArms, lambd, hor, numGames, seed)
	elif mode == 3:
		# SR
		delta = 1.0
		model = SuccessRatio(numArms, delta, hor, numGames, seed)
	elif mode == 4:
		# pi-first
		delta = 1.0
		pi = 4
		model = PiFirst(numArms, pi, delta, hor, numGames, seed)
	elif mode == 5:
		# epsilon greedy
		epsilon = 1.0
		model = EpsilonGreedy(numArms, epsilon, hor, numGames, seed)
	elif mode == 6:
		# epsilon decreasing
		epsilon = 1.0
		model = EpsilonDecreasing(numArms, epsilon, hor, numGames, seed)
	elif mode == 7:
		# eWSLS
		lambdaW = 1.0
		lambdaL = 1.0
		model = EWSLS(numArms, lambdaW, lambdaL, hor, numGames, seed)
	elif mode == 8:
		# tau switch
		delta = 1.0
		tau = 4
		model = TauSwitch(numArms, tau, delta, hor, numGames, seed)
	elif mode == 9:
		# latent state
		delta = 1.0
		model = LatentState(numArms, delta, hor, numGames, seed)

	return model


def processBandits(probs, hor, numArms, numModels, numSamples):
	random.seed(hor)

	avgBanditScores = getModelScores(numArms, hor, probs, numSamples, numModels)
	rankedModels = rankModels(avgBanditScores)

	print("{} | {} | {} | {}".format(hor, numArms, probs, rankedModels))

def main(numArms, numSamples):
	numModels = 10

	floatRegex = r"[0-9]+.[0-9]+"

	horizons = [5, 10, 15, 20, 25, 30, 35, 40]

	for line in sys.stdin:
		probs = list(map(parseFloat, line.split(",")))
		for hor in horizons:
			processBandits(probs, hor, numArms, numModels, numSamples)


if __name__ == '__main__':
	numArms = int(sys.argv[1])
	numSamples = int(sys.argv[2])
	main(numArms, numSamples)
