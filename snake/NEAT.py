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
	def __init__(self, input_node, output_node, weight, innovation_number, enabled: bool = True):
		self.input_node = input_node
		self.output_node = output_node
		self.weight = weight
		self.enabled = enabled
		self.innovation_number = innovation_number

	def in_value(self):
		return self.input_node.value

	def diable(self):
		self.enabled = False;

	def enable(self):
		self.enabled = True

class node:
	def __init__(self, label, value = 0, input_node = False, output_node = False):
		self.value = value #filled by parameter
		self.label = label #filled by parameter, unique to a given node (mayb a hash value)
		self.is_input = input_node #filled by parameter
		self.is_output = output_node #filled by parameter
		self.connected_to_in = [] #modified by add_input_node in this node and add_output_node in another node
		self.connected_to_out = [] #modified by add_output_node in this node and add_input_node in another node
		self.evaluated = False #modified by get_value
		self.contributes_to = [] #modified by evaluate_contributions

		#CHECKLIST FOR CREATING A NEW NODE
		# 1) Make sure all the parameters are correct. This will take care of the 
		# value, label, is_input, and is_output attributes
		# 2) Depending on whether or not this node is in input or output node it will
		# either call be called by add_input_node or it will call add_input_node. 
		# This will either modify the connected_to_in or connected_to_out lists


	
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


	def add_input_node(self, node, weight, inno = None):
		curr_iter = inno if inno != None else 0
		if not self.is_input:
			connect = connection(node, self, weight, curr_iter)
			self.connected_to_in.append(connect)
			node.connected_to_out.append(connect)
			return connect

	def add_output_node(self, node, weight, inno = None):
		curr_iter = inno if inno != None else 0
		if not self.is_output: 	
			con = connection(self, node, weight, curr_iter)
			self.connected_to_out.append(con)
			node.connected_to_in.append(con)
			return con

	#checks if a given node is an input to the current node
	def is_connected_to_in(self, node):
		for connection in self.connected_to_in:
			if connection.input_node == node:
				return True

		
		return False
	#checks if the current node outputs to the specified node
	def is_connected_to_out(self, node):
		for connection in self.connected_to_out:
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
		self.nodes = [] #contains all the nodes in the network
		self.input_nodes = [] #contains all the input nodes of the network
		self.output_nodes = [] #contains all the putput nodes of the network
		self.weights = [] #contains all the connections of the network
		self.graph = nx.DiGraph() # a graph representation of the network
		self.current_innovation = 0 #the current innovation nnumber, representing the number of structural mutations
		self.genes = [] #contains the genes which reppresent this network


	def reset_network(self):
		for node in self.nodes: 
			if not node.is_input:
				node.value = 0


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
			if coin_flip_2 > .6:		
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
			rand_connection.enabled = not rand_connection.enabled

		elif coin_flip > .1 or self.current_innovation == 0: # in the event that the network is just beginning
			# innovation number gets updated for each of the structural mutations
			self.current_innovation += 1

			#adds a node, c , into the network by splitting a random edge a->b into two new edges a->c and c->b. Where weight(a->c) = 1 and 
			#weight(c->b) = weight(a->b)
			enabled_weights = [weight for weight in self.weights if weight.enabled]
			rand_num = random.randint(0, len(enabled_weights) - 1)
			random_connection = enabled_weights[rand_num]
			in_node = random_connection.input_node
			out_node = random_connection.output_node
			new_node = node("hidden"+str(len(self.nodes)))
			self.weights.append(in_node.add_output_node(new_node, 1, self.current_innovation))
			self.weights.append(new_node.add_output_node(out_node, random_connection.weight, self.current_innovation))
			self.nodes.append(new_node)
			random_connection.enabled = False
			

		else:

			# innovation number gets updated for each of the structural mutations
			self.current_innovation += 1

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
			while (random_node_2.does_contributes_to(random_node_1) and random_node_1.is_connected_to_out(random_node_2)) or random_node_2.is_input:
				rand_num_1 = random_exception(len(non_output_nodes) -  1)
				rand_num_2 = random_exception(len(non_output_nodes) -  1, avoid = rand_num_1)
				random_node_1 = non_output_nodes[rand_num_1]
				random_node_2 = non_output_nodes[rand_num_2]

			#adjust exact number range later. not just in b ut floats toos


			self.weights.append(random_node_1.add_output_node(random_node_2, random.randint(-1000, 1000), self.current_innovation))
			

			



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

	def convert_to_genes(self):
		self.genes = []
		sorted_weights = sorted(self.weights, key = lambda x: x.innovation_number)
		for weight in sorted_weights:
			self.genes.append(gene(weight.input_node, weight.output_node, weight.weight, weight.enabled, weight.innovation_number))

	def add_new_genes(self):
		sorted_weights = sorted(self.weights, key = lambda x: x.innovation_number)
		for weight in sorted_weights:
			if weight.inovation_number > self.current_innovation:
				self.genes.append(gene(weight.input_node, weight.output_node, weight.weight, weight.enabled, weight.innovation_number))


	def evaluate_network(self):
		return_values = []
		for node in self.output_nodes:
			return_values.append((node.get_value(), node.label))
		return return_values

class gene:
	def __init__(self, input_node, output_node, weight, enabled, innovation_number):
		self.input = input_node
		self.output = output_node
		self.weight = weight
		self.enabled = enabled
		self.innovation = innovation_number

	def print_gene(self):
		print("input: ", self.input.label)
		print("output: ", self.output.label)
		print("innotvaion number: ", self.innovation)
		print("DISABLED \n\n\n" if not self.enabled else "\n\n\n")


def create_new_network_from_genes(genes):
	return_network = network()
	for gene in genes:
		in_node = gene.input
		out_node = gene.output
		new_in_node = None
		new_out_node = None

		found_in = False
		found_out = False
		# iIn the event that the input or output node is already in the network
		for item in return_network.nodes:
			# Compares the labels since they are unique. Cannot compare the actual objects
			if item.label == in_node.label:
				new_in_node = item
				found_in = True
			if item.label == out_node.label:
				new_out_node = item
				found_out = True

		if not found_in:
			new_in_node = node(in_node.label, in_node.value, in_node.is_input, in_node.is_output)
		if not found_out:
			new_out_node = node(out_node.label, out_node.value, out_node.is_input, out_node.is_output)

		# This covers the nodes attribtute of the network
		if new_in_node not in return_network.nodes:
			return_network.nodes.append(new_in_node)
		if new_out_node not in return_network.nodes:
			return_network.nodes.append(new_out_node)

		# This covers the input_nodes and output_nodes attributes of the network
		if new_in_node not in return_network.input_nodes and  new_in_node.is_input:
			return_network.input_nodes.append(new_in_node)
		if new_out_node not in return_network.output_nodes and new_out_node.is_output:
			return_network.output_nodes.append(new_out_node)


		# This covers the weights attribute of the network
		new_connection = new_in_node.add_output_node(new_out_node, gene.weight, gene.innovation)
		new_connection.enabled = gene.enabled
		return_network.weights.append(new_connection)

		#this will update the innovation number
		return_network.current_innovation = gene.innovation

	return_network.genes = genes
	return return_network

def is_networks_equal(X, Y):
	X.generate_graph()
	Y.generate_graph()

	graph1 = X.graph
	graph2 = Y.graph

	if not nx.is_isomorphic(graph1, graph2):
		return False

	for node in X.nodes:
	    for another_node in Y.nodes:
	        if node.label == another_node.label:
	            if node.value != another_node.value:
	            	return False
	return True





