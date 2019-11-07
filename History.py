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
