import random
import numpy as np
from abc import ABC, abstractmethod

from History import History

class Model(ABC):

	def __init__(self, numArms, horizon, numGames, seed):
		self.realHistory = History([(0, 0) for i in range(numArms)])
		self.decisions = []
		self.rewardHistory = []
		self.entireHistory = []

		self.numArms = numArms
		self.horizon = horizon
		self.numGames = numGames
		self.seed = seed

	@abstractmethod
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
		np.random.seed(self.seed)

		for i in range(self.horizon):
			armToPull = self.choose()
			self.updateDecisions(armToPull)
			success = np.random.binomial(1, probs[armToPull])
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
		np.random.seed(self.seed)

		previous = 0
		success = 0

		for i in range(self.horizon):
			armToPull = self.choose(i, previous, success)
			self.updateDecisions(armToPull)
			previous = armToPull
			success = np.random.binomial(1, probs[armToPull])
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
		np.random.seed(self.seed)

		for i in range(self.horizon):
			armToPull = self.choose()
			self.updateDecisions(armToPull)
			success = np.random.binomial(1, probs[armToPull])
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
		np.random.seed(self.seed)

		for i in range(self.horizon):
			armToPull = self.choose(i)
			self.updateDecisions(armToPull)
			success = np.random.binomial(1, probs[armToPull])
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
		np.random.seed(self.seed)

		for i in range(self.horizon):
			armToPull = self.choose()
			self.updateDecisions(armToPull)
			success = np.random.binomial(1, probs[armToPull])
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
		np.random.seed(self.seed)

		for i in range(self.horizon):
			armToPull = self.choose(i)
			self.updateDecisions(armToPull)
			success = np.random.binomial(1, probs[armToPull])
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
		np.random.seed(self.seed)

		previous = 0
		success = 0

		for i in range(self.horizon):
			armToPull = self.choose(i, previous, success)
			self.updateDecisions(armToPull)
			previous = armToPull
			success = np.random.binomial(1, probs[armToPull])
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
		np.random.seed(self.seed)

		for i in range(self.horizon):
			armToPull = self.choose(i)
			self.updateDecisions(armToPull)
			previous = armToPull
			success = np.random.binomial(1, probs[armToPull])
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
		np.random.seed(self.seed)

		for i in range(self.horizon):
			armToPull = self.choose(i)
			self.updateDecisions(armToPull)
			previous = armToPull
			success = np.random.binomial(1, probs[armToPull])
			self.updateRewardHistory(success)
			newHist = self.realHistory.addEvent(armToPull, success)
			self.setRealHistory(newHist)
			self.updateEntireHistory(self.realHistory)