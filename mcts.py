# from agent_network import Agent as agent
from gym import go  # go environment 
# import numpy as np


class MCTS:

	def __init__(self):
		self.env = go.reset()
		self.nodes = np.array([])  # saved nodes

	def select(self, state):
		action = agent.act(state)
		policy = agent.get_policy(state)
		next_state, reward, done = go.step(action)
		
		node = Node(next_state)
		if node.state not in [n.state for n in self.nodes]:
			node.policy = policy
			node.n += 1
			self.nodes.append(node)
		else:


		return next_state

	def simulate(self):
		# expand and evaluate
		pass
	
	def backup(self):
		pass		


class Node():

	def __init__(self, state):
		self.state = state
		self.policy = None
		self.value = None
		self.n = 0
		self.q = 0
		self.w = 0











