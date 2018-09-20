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
		node.policy = policy
		node.n += 1

		if node.state not in [n.state for n in self.nodes]:
			self.nodes.append(node)

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
		
# 위에거는 그냥 나둘게

import numpy as np
import math


class Mcts:
    def __init__(self, model, history, c_puct, temperature):
        self.model = model
        self.history = history
        self.c_puct = c_puct
        self.temperature = temperature

    def U(num):
        pass

    def Q(num):
        pass


    def search(s, env):

        s = tuple(s)

        if env.ended():
            return env.reward()

        if s not in history:
            p, v = self.model(s)
            action_lst = env.action_lst() # including pass
            node_lst = np.array([node(s, action, 0, 0, 0, p) for action in action_lst])
            self.history[s] = node_lst
            return v

        a, max_U = -1, -float('inf')
        node_lst = history[s]
        sigma_n = sum(n for n in node_lst)
        node_update = 0

        for node in node_lst:
            U = node.q + self.c_puct * node.p * math.sqrt(sigma_n) / node.n
            if U > max_U:
                node_update = node
                a = node.a
                max_U = U
                
                
        env.move(a) # a = -1 --> pass
        next_s = env.state()

        v = search(next_s, env)
        node_update.n += 1
        node_update.w += v
        node_update.Q += node_update.w/node.update.n

        return v
    
    def simulation(self, state):
        #multi processing
        return self.next_move(state)
        

    def next_move(self, state):
        pass










