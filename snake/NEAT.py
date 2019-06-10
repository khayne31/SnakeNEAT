import numpy as np
import random
import networkx as nx



def random_exception(length, avoid = None):	
	if avoid == None:
		return random.randint(0, length)
	else:
		return_num = random.randint(0, length)
		while return_num == avoid :
			return_num = random.randint(0, length)
		return return_num


class connection:
	def __init__(self, input_node, output_node, weight, enabled:bool = True):
		self.input_node = input_node
		self.output_node = output_node
		self.weight = weight
		self.enabled = enabled

	def in_value(self):
		return self.input_node.value

	def diable(self):
		self.enabled = False;

	def enable(self):
		self.enabled = True

class node:
	def __init__(self, label, value = 0, input_node = False, output_node = False):
		self.value = value
		self.label = label
		self.connected_to_in = []
		self.connected_to_out = []
		self.is_input = input_node
		self.is_output = output_node
		self.evaluated = False
		self.contributes_to = []	

	def evaluate_contributions(self, start_node):
		# this makes it so that each node knoews that nodes that it contributes to. Each node inherits the contributions of all the nodes that
		# it directly connects to. Meaning if the output of node A goes into node B and the output from node B goes into both node C and Node D
		# then node A contributes to node A, node B, and node C where node B contributes only to node C and node D. This should work for the 
		# purpose of adding new connections to the network. We dont want there to be a connection formed from one node(X) to another node (Y)
		# s.t. Y contributes to X. This will lead to infinite loops in the evaluation stage. The only problem with this way is that when a 
		# connection is diabled we need to adjust the connected_to list accordingly.
		for connection in start_node.connected_to_in:
			input_node = connection.input_node
			if start_node not in input_node.contributes_to:
				input_node.contributes_to.append(start_node)
			for node in start_node.contributes_to:
				if node not in input_node.contributes_to:
					input_node.contributes_to.append(node)
			self.evaluate_contributions(input_node)
			



	def get_value(self):
		if self.is_input or self.evaluated:
			return self.value
		else:
			for connection in self.connected_to_in:
				if connection.enabled:
					self.value += connection.weight * connection.input_node.get_value()
			self.evaluated = True 
		return self.value


	def add_input_node(self, node, weight):
		if not self.is_input:
			connect = connection(node, self, weight)
			self.connected_to_in.append(connect)
			node.connected_to_out.append(connect)
			return connect

	def add_output_node(self, node, weight):
		if not self.is_output: 	
			con = connection(self, node, weight)
			self.connected_to_out.append(con)
			node.connected_to_in.append(con)
			return con
	def is_connected_to(self, node):
		for connection in connected_to_in:
			if connection.input_node == node:
				return True

		for connection in connected_to_out:
			if connection.output_node == node:
				return True
		return False

	def does_contributes_to(self, node):
		return node in self.contributes_to

	#gonna have to clear before each evaluation of the output
	def clear_contributions(self):
		self.contributes_to = []

class network:
	def __init__(self):
		self.nodes = []
		self.input_nodes = []
		self.output_nodes = []
		self.weights = []
		self.graph = nx.DiGraph()

	def initalize_network(self, num_input_nodes: int, num_output_nodes: int, value_list: list = []):
		#The idea is to create a basic network with no hidden layers where the inputs connect directly to the outputs. This will be the basic network
		#The NEAT network will evolve from
		#the value_list parameter will be the values which go into the input nodes, in order. if no list is given the values default to 0
		values = value_list if value_list != [] else [0] * num_input_nodes

		#initializes all of the input nodes
		for i in range(num_input_nodes):
			new_node = node("input " + str(i), values[i], input_node = True)
			self.nodes.append(new_node)
			self.input_nodes.append(new_node)
		#initializes all of the output nodes
		for i in range(num_output_nodes):
			new_node = node("output " + str(i+num_input_nodes), output_node = True)
			self.nodes.append(new_node)
			self.output_nodes.append(new_node)
		#records all of the connections in the network
		for input_node in self.input_nodes:
			for output in self.output_nodes:
				self.weights.append(input_node.add_output_node(output, 1))

	
	# TODO: Develop a way to convert a network into a list of genes, probally should make each connection have an innovation number which gets incremented
	# with each structural mutation. Then create a way to convert from a list of genes into a network
 	
	def mutation(self):
		# fifty percent chance to add a new node and fifty percent change to add a connection.
		coin_flip = random.random()

		# 70% chance to change the weight of a random connection through mutation, 10 percent chance to enable or disable a connection,
		# 10% to add a random node, 10% to add a new connection witha  random weight value
		if coin_flip > .3:
			#weight mutatiions
			coin_flip_2 = random.random()	
			enabled_weights = [weight for weight in self.weights if weight.enabled]
			rand_num = random.randint(0, len(enabled_weights) - 1)
			random_connection = enabled_weights[rand_num]

			# we can a) completly change it with a random number b) change the weight by some percentage (multiply by some number between 0 and 2) 
			# c) add or subtract a random number between 0 and 1 to/from the weight d) change the sign of the weight e) some combination of these 
			# techniqiues

			#40% chance to change by some percentage, 40% to add a number from [-1, 1) 10% to flip sign, and 10% to chnage to a completly random number
			if coin_flip_2 > .6		
				#multiply by a random_number from  0 to 2
				random_connection.weight *= random.random() * 2
				
			elif coin_flip_2 > .2:
				#add a random number from -1 to 1
				random_connection.weight += random.random() * 2 - 1
			elif coin_flip_2 > .1:
				#change the sign
				random_connection.weight *= -1
			else:
				#adjust later curently [-100, 100)
				random_connection.weight = random.random() * 200 - 100

		elif coin_flip > .2:
			#enable or disable a weight
			rand_num = random.randint(0, len(self.weights) - 1)
			rand_connection = self.weights[rand_num]
			connection.enabled = not connection.enabled

		elif coin_flip > .1:
			#adds a node, c , into the network by splitting a random edge a->b into two new edges a->c and c->b. Where weight(a->c) = 1 and 
			#weight(c->b) = weight(a->b)
			enabled_weights = [weight for weight in self.weights if weight.enabled]
			rand_num = random.randint(0, len(enabled_weights) - 1)
			random_connection = enabled_weights[rand_num]
			in_node = random_connection.input_node
			out_node = random_connection.output_node
			new_node = node("hidden"+str(len(self.nodes)))
			self.weights.append(in_node.add_output_node(new_node, 1))
			self.weights.append(new_node.add_output_node(out_node, random_connection.weight))
			self.nodes.append(new_node)
			random_connection.enabled = False
		else:
			#mutate the structutre of the network
			for output in self.output_nodes:
				output.evaluate_contributions(output)

			non_output_nodes = [node for node in self.nodes if node not in self.output_nodes]
			rand_num_1 = random_exception(len(non_output_nodes) -  1)
			rand_num_2 = random_exception(len(non_output_nodes) -  1, avoid = rand_num_1)
			random_node_1 = non_output_nodes[rand_num_1]
			random_node_2 = non_output_nodes[rand_num_2]

			# this prevents a connection being two nodes where the input node in this connecion X has its value affected by the output node Y
			# in other words Y's value contributes to the value of X. Tbis would create and infinite loop
			while random_node_2.does_contributes_to(random_node_1) or random_node_2.is_input:
				rand_num_1 = random_exception(len(non_output_nodes) -  1)
				rand_num_2 = random_exception(len(non_output_nodes) -  1, avoid = rand_num_1)
				random_node_1 = non_output_nodes[rand_num_1]
				random_node_2 = non_output_nodes[rand_num_2]

			#adjust exact number range later. not just in b ut floats toos
			self.weights.append(random_node_1.add_output_node(random_node_2, random.randint(-1000, 1000)))

			



	def generate_graph(self):
		self.graph.clear() 	
		self.graph.add_nodes_from(self.nodes)


		
		for node in self.nodes:
		    for connection in node.connected_to_out:
		    	if connection.enabled:
		        	self.graph.add_edge(connection.input_node, connection.output_node, weight = connection.weight)
		        #G.add_edge(connection.output_node, connection.input_node, weight = connection.weight)
		        
		labels = []
		nodes = self.nodes
		for node in nodes:
		    labels.append(node.label)
		mapping = dict(zip(nodes, labels))
		self.graph = nx.relabel_nodes(self.graph, mapping)

	def draw_graph(self):
		nx.draw_shell(self.graph, with_labels = True)





class gene:
	def __init__(self, input_node, output_node, weight, enabled, inovation_number):
		self.input = input_node
		self.output = output_node
		self.weight = weight
		self.enabled = enabled
		self.inovation = inovation_number

input1 = node("1", 1, input_node = True)
input2 = node("1", 2, input_node = True)
input3 = node("1", 3, input_node = True)
input4 = node("4", 4, input_node = True)
output1 = node("4", output_node = True)
output2 = node("10", output_node = True)
output3 = node("11", output_node = True)
output4 = node("12", output_node = True)
hidden1 = node("5")
hidden2 = node("6")
hidden3 = node("7")
hidden4 = node("8")
hidden5 = node("9")

x = [output1, output2, output3, output4]

input1.add_output_node(output1, 1)
input1.add_output_node(hidden2, 1)
input2.add_output_node(hidden1, 1)
input3.add_output_node(hidden1, 1)
input3.add_output_node(hidden2, 1)
input3.add_output_node(hidden3, 1)
input4.add_output_node(hidden2, 1)
input4.add_output_node(hidden3, 1)
hidden1.add_output_node(hidden4, 1)
hidden1.add_output_node(hidden5, 1)
hidden2.add_output_node(hidden4, 1)
hidden2.add_output_node(output3, 1)
hidden3.add_output_node(hidden5, 1)
hidden4.add_output_node(output1, 1)
hidden4.add_output_node(output3, 1)
hidden5.add_output_node(output2, 1)
hidden5.add_output_node(output4, 1)
print(output1.get_value())
print(output2.get_value())
print(output3.get_value())
print(output4.get_value())

expected =np.matrix([2,4,6,8])
values = [result.get_value() for result in x]
actual = np.matrix(values)
cost = .5 * np.linalg.norm(expected - actual)
print(cost)

