#!/usr/bin/env python
import keras, tensorflow as tf, numpy as npy, gym, sys, copy, argparse, random
from collections import deque

INIT_EPSILON = 0.5
FINAL_EPSILON = 0.05
DECAY_ITERATION = 100000
MAX_ITERATION_PER_EPISODE = 10000

class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_name, model = None):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.
		env = gym.make(environment_name)
		self.state_dim = env.observation_space.shape[0]
		self.action_dim = env.action_space.n
		self.learning_rate = 0.0001

		self.session = tf.InteractiveSession()

		if model != None:
			self.load_model(model)
		else:
			self.CreateMLP()
			self.CreateOptimizer()


	def CreateMLP(self):
		self.hidden_units = 20

		self.w1 = tf.Variable(tf.random_normal([self.state_dim, self.hidden_units]))
		self.b1 = tf.Variable(tf.random_normal([self.hidden_units]))
		self.w2 = tf.Variable(tf.random_normal([self.hidden_units, self.action_dim]))
		self.b2 = tf.Variable(tf.random_normal([self.action_dim]))

		self.state_input = tf.placeholder(tf.float32, [None, self.state_dim], name = "state_input")

		h_layer = tf.nn.relu(tf.matmul(self.state_input,self.w1) + self.b1)

		self.q_values = tf.add(tf.matmul(h_layer, self.w2), self.b2, name = "q_values")

	def CreateLinearNetwork(self):
		w = tf.Variable(tf.zeros([self.state_dim, self.action_dim]))
		b = tf.Variable(tf.zeros([self.action_dim]))

		self.state_input = tf.placeholder(tf.float32, [None, self.state_dim], name = "state_input")

		self.q_values = tf.add(tf.matmul(self.state_input, w), b, name = "q_values")

	def CreateOptimizer(self):
		self.action_input = tf.placeholder(tf.float32, [None, self.action_dim], name = "action_input")
		self.target_q_value = tf.placeholder(tf.float32, [None], name = "target_q_value")
		q_value_output = tf.reduce_sum(tf.multiply(self.q_values, self.action_input), 1)
		cost = tf.reduce_mean(tf.square(tf.subtract(self.target_q_value, q_value_output)))
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost, name = "optimizer")

		self.session.run(tf.global_variables_initializer())

	def get_q_values(self, state):
		return self.q_values.eval(feed_dict = {self.state_input : state})

	def save_model_weights(self, suffix, step):
		# Helper function to save your model / weights.
		saver = tf.train.Saver()
		tf.add_to_collection("optimizer", self.optimizer)
		saver.save(self.session, suffix, global_step = step)
		pass

	def load_model(self, model_file):
		# Helper function to load an existing model.
		saver = tf.train.import_meta_graph(model_file + '.meta')
		saver.restore(self.session, model_file)

		graph = tf.get_default_graph()
		self.q_values = graph.get_tensor_by_name("q_values:0")
		self.state_input = graph.get_tensor_by_name("state_input:0")
		self.action_input = graph.get_tensor_by_name("action_input:0")
		self.target_q_value = graph.get_tensor_by_name("target_q_value:0")
		self.optimizer = tf.get_collection("optimizer")[0]
		pass

	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights. 
		file = open("./test", "r")
		print file.readline()
		pass

class Replay_Memory():

	# burn_in = 10000
	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
		self.buffer = deque()
		self.memory_size = memory_size
		self.burn_in = burn_in
		pass

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		return random.sample(self.buffer, batch_size)

	def append(self, transition):
		# Appends transition to the memory.
		self.buffer.append(transition)
		if len(self.buffer) > self.memory_size:
			self.buffer.popleft()

class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, render=False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.
		self.q_network = QNetwork(environment_name)
		self.replay_memory = Replay_Memory()
		self.env = gym.make(environment_name)
		# self.env = gym.wrappers.Monitor(self.env, '.', force = True)
		self.action_dim = self.env.action_space.n
		self.epsilon = INIT_EPSILON # this value should be decayed
		self.gamma = 0.9
		self.episode = 1000000

	def epsilon_greedy_policy(self, q_values, epsilon):
		# Creating epsilon greedy probabilities to sample from.     
		if random.random() <= epsilon:
			return random.randint(0, self.env.action_space.n - 1)
		else:
			return self.greedy_policy(q_values)

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time.
		return npy.argmax(q_values)

	def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
		for i_episode in range(self.episode):
			state = self.env.reset()
			q_values = self.q_network.get_q_values([state])[0] # this q values could be multiple states
			reward_sum = 0
			for t in range(MAX_ITERATION_PER_EPISODE):
				# self.env.render()
				action = self.epsilon_greedy_policy(q_values, self.epsilon)
				if (self.epsilon > FINAL_EPSILON):
					self.epsilon -= (INIT_EPSILON - FINAL_EPSILON) / DECAY_ITERATION

				next_state, reward, done, info = self.env.step(action)

				reward_sum += reward

				action_input = npy.zeros(self.action_dim)
				action_input[action] = 1

				next_state_q_values = self.q_network.get_q_values([next_state])[0]
				target = reward
				if not done:
					target += self.gamma * next_state_q_values[self.greedy_policy(next_state_q_values)]

				self.replay_memory.append([state, action_input, target])

				if len(self.replay_memory.buffer) >= self.replay_memory.burn_in:
					# train
					batch = self.replay_memory.sample_batch()
					state_batch = []
					action_batch = []
					target_batch = []
					for data in batch:
						state_batch.append(data[0])
						action_batch.append(data[1])
						target_batch.append(data[2])

						self.q_network.optimizer.run(feed_dict = {self.q_network.state_input : state_batch, 
							self.q_network.action_input : action_batch, self.q_network.target_q_value : target_batch})
				
				state = next_state
				q_values = next_state_q_values

				if done:
					break

			# if reward_sum > -200:
			# 	print("episode", i_episode, " reward: ", reward_sum)

			if i_episode % 200 == 0:
				print("episode: ", i_episode)
				self.test()
				# self.q_network.save_model_weights("./checkpoints/CartPole-v0", i_episode)


	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		episode_num = 10
		total_reward = 0
		for i in range(episode_num):
			state = self.env.reset()
			for t in range(MAX_ITERATION_PER_EPISODE):
				q_values = self.q_network.get_q_values([state])[0] # this q values could be multiple states
				self.env.render()
				action = self.epsilon_greedy_policy(q_values, FINAL_EPSILON)

				state, reward, done, info = self.env.step(action)

				total_reward += reward

				if done:
					break
		ave_reward = total_reward / episode_num
		print 'Evaluation Average Reward:',ave_reward

	def burn_in_memory():
		# Initialize your replay memory with a burn_in number of episodes / transitions. 

		pass

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
	# keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 

	environment_name = 'CartPole-v0'
	agent = DQN_Agent(environment_name)
	agent.train()

if __name__ == '__main__':
	main(sys.argv)

