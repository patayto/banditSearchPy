#!/usr/bin/python2.7
# Reduce1.py
import sys
import re
import random

class History:
	def __init__(self, events):
		"""

		:type events: [(successes for an arm, failures for an arm)]
		"""
		self.events = events

	def addEvent(self, arm, success):
		ov = self.events[arm]
		nv = (ov[0]+1, ov[1]) if success else (ov[0], ov[1]+1)
		newEvents = self.events[:]
		newEvents[arm] = nv
		return History(newEvents)

	def __str__(self):
		return ", ".join(["({}, {})".format(s,f) for (s,f) in self.events])


class Model(object):

	def __init__(self, numArms, horizon, numGames, seed):
		self.realHistory = History([(0, 0) for i in range(numArms)])
		self.decisions = []
		self.rewardHistory = []
		self.entireHistory = []

		self.numArms = numArms
		self.horizon = horizon
		self.numGames = numGames
		self.seed = seed

	def run(self):
		pass

	def resetRealHistory(self):
		events = [(0, 0) for i in range(self.numArms)]
		self.realHistory = History(events)

	def setRealHistory(self, history):
		self.realHistory = history

	def resetDecisions(self):
		self.decisions = []

	def updateDecisions(self, decision):
		self.decisions.append(decision+1)

	def resetRewardHistory(self):
		self.rewardHistory = []

	def updateRewardHistory(self, success):
		self.rewardHistory.append(success)

	def updateEntireHistory(self, history):
		self.entireHistory.append(history)

	def init(self):
		random.seed(self.seed)
		self.resetRealHistory()
		self.resetDecisions()
		self.resetRewardHistory()


class Guessing(Model):

	def __init__(self, numArms, horizon, numGames, seed):
		super(Guessing, self).__init__(numArms, horizon, numGames, seed)

	def choose(self):
		return random.randint(0, self.numArms - 1)

	def run(self, probs):
		random.seed(self.seed)

		for i in range(self.horizon):
			armToPull = self.choose()
			self.updateDecisions(armToPull)
			success = 1 if random.random() < probs[armToPull] else 0 # bernoulli pull
			self.updateRewardHistory(success)
			newHist = self.realHistory.addEvent(armToPull, success)
			self.setRealHistory(newHist)
			self.updateEntireHistory(self.realHistory)



class WSLS(Model):

	def __init__(self, numArms, lambd, hor, numGames, seed):
		super(WSLS, self).__init__(numArms, hor, numGames, seed)
		self.lambd = lambd

	def choose(self, decision, previous, success):
		if decision == 0:
			return random.randint(0, self.numArms - 1)
		else:
			prob = random.random()
			if (success):
				if prob < self.lambd:
					return previous
				else:
					next = random.randint(0, self.numArms - 1)
					while (next == previous):
						next = random.randint(0, self.numArms - 1)
					return next
			else:
				if prob < self.lambd:
					next = random.randint(0, self.numArms - 1)
					while (next == previous):
						next = random.randint(0, self.numArms - 1)
					return next
				else:
					return previous

	def run(self, probs):
		random.seed(self.seed)

		previous = 0
		success = 0

		for i in range(self.horizon):
			armToPull = self.choose(i, previous, success)
			self.updateDecisions(armToPull)
			previous = armToPull
			success = 1 if random.random() < probs[armToPull] else 0  # bernoulli pull
			self.updateRewardHistory(success)
			newHist = self.realHistory.addEvent(armToPull, success)
			self.setRealHistory(newHist)
			self.updateEntireHistory(self.realHistory)


class SuccessRatio(Model):

	def __init__(self, numArms, delta, hor, numGames, seed):
		super(SuccessRatio, self).__init__(numArms, hor, numGames, seed)
		self.delta = delta

	def choose(self):
		sr = list(map(lambda event: (event[0]+1.0)/(event[0]+event[1]+2.0), self.realHistory.events))
		srWithInd = [(ind, s) for (ind, s) in enumerate(sr) if s == max(sr)]
		next = srWithInd[random.randint(0, len(srWithInd))-1][0]
		prob = random.random()
		if (prob < self.delta):
			return next
		else:
			return random.randint(0, self.numArms-1)

	def run(self, probs):
		random.seed(self.seed)

		for i in range(self.horizon):
			armToPull = self.choose()
			self.updateDecisions(armToPull)
			success = 1 if random.random() < probs[armToPull] else 0  # bernoulli pull
			self.updateRewardHistory(success)
			newHist = self.realHistory.addEvent(armToPull, success)
			self.setRealHistory(newHist)
			self.updateEntireHistory(self.realHistory)

class PiFirst(Model):

	def __init__(self, numArms, pi, delta, hor, numGames, seed):
		super(PiFirst, self).__init__(numArms, hor, numGames, seed)
		self.pi = pi
		self.delta = delta

	def choose(self, trial):
		if (trial <= self.pi):
			return random.randint(0, self.numArms-1)
		else:
			sr = list(map(lambda event: (event[0]+1.0)/(event[0]+event[1]+2.0), self.realHistory.events))
			srWithInd = [(ind, s) for (ind, s) in enumerate(sr) if s == max(sr)]
			next = srWithInd[random.randint(0, len(srWithInd))-1][0]
			prob = random.random()
			if (prob < self.delta):
				return next
			else:
				return random.randint(0, self.numArms-1)

	def run(self, probs):
		random.seed(self.seed)

		for i in range(self.horizon):
			armToPull = self.choose(i)
			self.updateDecisions(armToPull)
			success = 1 if random.random() < probs[armToPull] else 0  # bernoulli pull
			self.updateRewardHistory(success)
			newHist = self.realHistory.addEvent(armToPull, success)
			self.setRealHistory(newHist)
			self.updateEntireHistory(self.realHistory)


class EpsilonGreedy(Model):

	def __init__(self, numArms, epsilon, hor, numGames, seed):
		super(EpsilonGreedy, self).__init__(numArms, hor, numGames, seed)
		self.epsilon = epsilon

	def choose(self):
		sr = list(map(lambda event: (event[0]+1.0)/(event[0]+event[1]+2.0), self.realHistory.events))
		srWithInd = [(ind, s) for (ind, s) in enumerate(sr) if s == max(sr)]
		next = srWithInd[random.randint(0, len(srWithInd))-1][0]
		prob = random.random()
		if (prob < self.epsilon):
			return next
		else:
			return random.randint(0, self.numArms-1)

	def run(self, probs):
		random.seed(self.seed)

		for i in range(self.horizon):
			armToPull = self.choose()
			self.updateDecisions(armToPull)
			success = 1 if random.random() < probs[armToPull] else 0  # bernoulli pull
			self.updateRewardHistory(success)
			newHist = self.realHistory.addEvent(armToPull, success)
			self.setRealHistory(newHist)
			self.updateEntireHistory(self.realHistory)



class EpsilonDecreasing(Model):

	def __init__(self, numArms, epsilon, hor, numGames, seed):
		super(EpsilonDecreasing, self).__init__(numArms, hor, numGames, seed)
		self.epsilon = epsilon

	def choose(self, trial):
		sr = list(map(lambda event: (event[0]+1.0)/(event[0]+event[1]+2.0), self.realHistory.events))
		srWithInd = [(ind, s) for (ind, s) in enumerate(sr) if s == max(sr)]
		next = srWithInd[random.randint(0, len(srWithInd))-1][0]
		prob = random.random()
		if (prob < (self.epsilon/(trial+1)*1.0)):
			return next
		else:
			return random.randint(0, self.numArms-1)

	def run(self, probs):
		random.seed(self.seed)

		for i in range(self.horizon):
			armToPull = self.choose(i)
			self.updateDecisions(armToPull)
			success = 1 if random.random() < probs[armToPull] else 0  # bernoulli pull
			self.updateRewardHistory(success)
			newHist = self.realHistory.addEvent(armToPull, success)
			self.setRealHistory(newHist)
			self.updateEntireHistory(self.realHistory)


class EWSLS(Model):

	def __init__(self, numArms, lambdaW, lambdaL, hor, numGames, seed):
		super(EWSLS, self).__init__(numArms, hor, numGames, seed)
		self.lambdaW = lambdaW
		self.lambdaL = lambdaL

	def choose(self, decision, previous, success):
		if decision == 0:
			return random.randint(0, self.numArms - 1)
		else:
			prob = random.random()
			if (success):
				if prob < self.lambdaW:
					return previous
				else:
					next = random.randint(0, self.numArms - 1)
					while (next == previous):
						next = random.randint(0, self.numArms - 1)
					return next
			else:
				if prob < self.lambdaL:
					next = random.randint(0, self.numArms - 1)
					while (next == previous):
						next = random.randint(0, self.numArms - 1)
					return next
				else:
					return previous

	def run(self, probs):
		random.seed(self.seed)

		previous = 0
		success = 0

		for i in range(self.horizon):
			armToPull = self.choose(i, previous, success)
			self.updateDecisions(armToPull)
			previous = armToPull
			success = 1 if random.random() < probs[armToPull] else 0  # bernoulli pull
			self.updateRewardHistory(success)
			newHist = self.realHistory.addEvent(armToPull, success)
			self.setRealHistory(newHist)
			self.updateEntireHistory(self.realHistory)


class TauSwitch(Model):

	def __init__(self, numArms, tau, delta, hor, numGames, seed):
		super(TauSwitch, self).__init__(numArms, hor, numGames, seed)
		self.tau = tau
		self.delta = delta

	def choose(self, trial):
		# get successes of each arm
		succ = [s for (s,f) in self.realHistory.events]

		# get failures of each arm
		fail = [f for (s, f) in self.realHistory.events]

		# find all of the arms whose number of successes is equal to the max number of successes
		succWithInd = [(ind, s) for (ind, s) in enumerate(succ) if s == max(succ)]
		succInd = set([ind for (ind, s) in succWithInd])

		# find all of the arms whose number of failures is equal to the min number of failures
		failWithInd = [(ind, f) for (ind, f) in enumerate(fail) if f == min(fail)]
		failInd = set([ind for (ind, f) in failWithInd])

		sitArms = list(succInd.intersection(failInd))

		next = random.randint(0, self.numArms - 1)
		prob = random.random()

		if len(sitArms) > 1:
			# same situation, choose an arm from the intersection at random
			next = sitArms[random.randint(0, len(sitArms))-1]

		elif len(sitArms) == 1:
			# better worse-situation
			if (prob < self.delta):
				next = succWithInd[0][0]
			else:
				while next == succWithInd[0][0]:
					next = random.randint(0, self.numArms - 1)
		else:
			# explore-exploit
			if trial <= self.tau:
				# explore
				if prob < self.delta:
					# explore with prob delta
					next = failWithInd[random.randint(0, len(failWithInd))-1][0]
				else:
					# exploit with prob 1 - delta
					next = succWithInd[random.randint(0, len(succWithInd))-1][0]
			else:
				# exploit
				if prob < self.delta:
					# expoit with prob delta
					next = succWithInd[random.randint(0, len(succWithInd)) - 1][0]
				else:
					# explore with prob 1 - delta
					next = failWithInd[random.randint(0, len(failWithInd))-1][0]

		return next

	def run(self, probs):
		random.seed(self.seed)

		for i in range(self.horizon):
			armToPull = self.choose(i)
			self.updateDecisions(armToPull)
			previous = armToPull
			success = 1 if random.random() < probs[armToPull] else 0  # bernoulli pull
			self.updateRewardHistory(success)
			newHist = self.realHistory.addEvent(armToPull, success)
			self.setRealHistory(newHist)
			self.updateEntireHistory(self.realHistory)


class LatentState(Model):

	def __init__(self, numArms, delta, hor, numGames, seed):
		super(LatentState, self).__init__(numArms, hor, numGames, seed)
		self.delta = delta

	def choose(self, trial):
		# get successes of each arm
		succ = [s for (s,f) in self.realHistory.events]

		# get failures of each arm
		fail = [f for (s, f) in self.realHistory.events]

		# find all of the arms whose number of successes is equal to the max number of successes
		succWithInd = [(ind, s) for (ind, s) in enumerate(succ) if s == max(succ)]
		succInd = set([ind for (ind, s) in succWithInd])

		# find all of the arms whose number of failures is equal to the min number of failures
		failWithInd = [(ind, f) for (ind, f) in enumerate(fail) if f == min(fail)]
		failInd = set([ind for (ind, f) in failWithInd])

		sitArms = list(succInd.intersection(failInd))

		next = random.randint(0, self.numArms - 1)
		prob = random.random()

		if len(sitArms) > 1:
			# same situation, choose an arm from the intersection at random
			next = sitArms[random.randint(0, len(sitArms))-1]

		elif len(sitArms) == 1:
			# better worse-situation
			if (prob < self.delta):
				next = succWithInd[0][0]
			else:
				while next == succWithInd[0][0]:
					next = random.randint(0, self.numArms - 1)
		else:
			# explore-exploit
			latentState = random.randint(0,1)
			if not latentState:
				# explore
				if prob < self.delta:
					# explore with prob delta
					next = failWithInd[random.randint(0, len(failWithInd))-1][0]
				else:
					# exploit with prob 1 - delta
					next = succWithInd[random.randint(0, len(succWithInd))-1][0]
			else:
				# exploit
				if prob < self.delta:
					# expoit with prob delta
					next = succWithInd[random.randint(0, len(succWithInd)) - 1][0]
				else:
					# explore with prob 1 - delta
					next = failWithInd[random.randint(0, len(failWithInd))-1][0]

		return next

	def run(self, probs):
		random.seed(self.seed)

		for i in range(self.horizon):
			armToPull = self.choose(i)
			self.updateDecisions(armToPull)
			previous = armToPull
			success = 1 if random.random() < probs[armToPull] else 0  # bernoulli pull
			self.updateRewardHistory(success)
			newHist = self.realHistory.addEvent(armToPull, success)
			self.setRealHistory(newHist)
			self.updateEntireHistory(self.realHistory)

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
