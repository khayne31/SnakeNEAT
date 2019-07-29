import numpy as np
import random
import networkx as nx
import time



def random_exception(length, avoid = None):	
	if avoid == None:
		return random.randint(0, length)
	else:
		if len(avoid) >= length:
			return -1
		return_num = random.randint(0, length)
		while return_num in avoid:
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

	def to_string(self):
		return self.input_node.label + " -> " + self.output_node.label + " -> " + ": "+ str(self.weight) + (" - ENABLED" if self.enabled else " - DISABLED")

	def terminate(self, net):
		net.weights.remove(self)
		self.input_node.connected_to_out.remove(self)
		self.output_node.connected_to_in.remove(self)					
		self = None

class node:
	def __init__(self, label, value = 1, input_node = False, output_node = False):
		self.value = value #filled by parameter
		self.label = label #filled by parameter, unique to a given node (mayb a hash value)
		self.is_input = input_node #filled by parameter
		self.is_output = output_node #filled by parameter
		self.connected_to_in = [] #modified by add_input_node in this node and add_output_node in another node
		self.connected_to_out = [] #modified by add_output_node in this node and add_input_node in another node
		self.evaluated = False #modified by get_value
		self.parents = [] #holds the ancestors of the node
		self.log = "" #the log for the node
		self.times_called = 0
		self.mutation_log = ""

		#CHECKLIST FOR CREATING A NEW NODE
		# 1) Make sure all the parameters are correct. This will take care of the 
		# value, label, is_input, and is_output attributes
		# 2) Depending on whether or not this node is in input or output node it will
		# either call be called by add_input_node or it will call add_input_node. 
		# This will either modify the connected_to_in or connected_to_out lists




	def get_value(self, check = "", loop = ""):
		self.times_called += 1
		if self.is_input or self.evaluated:
			return self.value
		else:
			#print(self.label)
			self.evaluated = True
			summation = 0
			for connection in self.connected_to_in:
				if connection.enabled:
					i_node = connection.input_node
					old_sum = summation
					diff = connection.weight * i_node.get_value(self.label, "get_value()")
					summation += diff
			self.value += summation

			total = 0
			for con in self.connected_to_in:
				if con.enabled:
					total += con.weight * con.input_node.value

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
		if not self.is_output and not self.is_connected_to_out(node): 		
			con = connection(self, node, weight, curr_iter)
			self.connected_to_out.append(con)
			node.connected_to_in.append(con)
			return con
		return None

	#checks if a given node is an input to the current node
	def is_connected_to_in(self, node):
		for connection in self.connected_to_in:
			if connection.input_node == node:
				return True

		
		return False
	#checks if the current node outputs to the specified node
	def is_connected_to_out(self, node):
		for connection in self.connected_to_out:
			if connection.output_node == node or connection.output_node.label == node.label:
				return True
		return False

	#determines if the node calling this function contributes to the node in the parameter

	def is_ancestor(self, node):
		return node in self.parents

	#TODO: check to make sure your onlt taking values from nodes connected to the inputs
	def test_value(self):
		total = 1
		child_values = []
		for connection in self.connected_to_in:
			if connection.enabled:
				val = connection.input_node.value 
				child_values.append(val)
				total += val * connection.weight
		return (total == self.value, total, self.value, child_values)

	def test_children(self):
		for connection in self.connected_to_in:
			child = connection.input_node
			if child.is_ancestor(self):
				return (False, child.label, child.value)
		return (True, "")


class network:
	def __init__(self):
		self.nodes = [] #contains all the nodes in the network
		self.input_nodes = [] #contains all the input nodes of the network
		self.output_nodes = [] #contains all the putput nodes of the network
		self.weights = [] #contains all the connections of the network
		self.graph = nx.DiGraph() # a graph representation of the network
		self.current_innovation = 0 #the current innovation nnumber, representing the number of structural mutations
		self.genes = [] #contains the genes which reppresent this network
		self.fitness = 0


	# TODO: 1) write a test which check if only the nodes whih are ancestors of the output nodes
	# have corret values 2) whenever we call get_values() on a already evaluated node get data 
	# about that node and its children (what the value shold be, are its children also evaluated)
	# 3) potentially test when all weights are enabled

	def test_network(self):
		#Gets all the nodes that are connected to output nodes
		copy_net = self
		list_of_nodes = list(copy_net.graph.nodes)
		connected_to_outputs = []
		connected_to_inputs = []
		input_labels = [i.label for i in copy_net.input_nodes]
		output_labels = [o.label for o in copy_net.output_nodes]
		for n in list_of_nodes:
			input_ancestors = nx.ancestors(copy_net.graph, n)

			for ancestor in input_ancestors:
				if ancestor in input_labels:
					current_node = copy_net.get_node_from_label(n)
					if current_node not in connected_to_inputs:
						connected_to_inputs.append(current_node)


			if n in output_labels:
				output_ancestors = nx.ancestors(copy_net.graph, n)

				for ancestor in output_ancestors:
					current_node = copy_net.get_node_from_label(ancestor)
					if current_node not in connected_to_outputs:
						connected_to_outputs.append(current_node)

		copy_net.evaluate_network()
		#FOR TESTING N
		for node in connected_to_outputs:
			result = node.test_value()
			if not result[0] and not node.is_input:
				in_values = [con.input_node.value if con.enabled else 0 for con in node.connected_to_in]
				print("-------------------------------")
				print("failed: ", node.label)
				print("actual value:", node.value)
				print("target value: ", result[1])
				print("diffrence: ", str(abs(node.value - result[1])))
				print(in_values)
				print("sum: ", sum(in_values))
				print("length inout connections: ", len(node.connected_to_in))
				print("child test ", node.test_children())
				print("-------------------------------")
		for node in self.nodes:
			if node in connected_to_outputs:
				print(node.test_value(), node.label)
		print("test done")



	# ANOTHER WAY TO DO GET_VALUE:
	# A node cannot be evaluated unless all of its ancestors are also evaluated (except inputs)
	# in the event that a ancestor node is not evaluated, run get_value() on that ancestor recursivly
	# this might solver a problem or might add a new one 
	def evaluate_network(self):
		self.reset_network()
		return_values = []
		# for output in self.output_nodes:
		# 	return_values.append(output.get_value("output"))

		# for node in self.nodes:
		# 	total = 0
		# 	for connection in node.connected_to_in:
		# 		if connection.enabled:
		# 			total += connection.input_node.value * connection.weight
		# 	total += 1
		# 	if total != node.value:
		# 		node.evaluated = False
		# 		node.value = 1
		# 		node.get_value("testing values")

		return [op.get_value("return", "evaluate_network") for op in self.output_nodes]
	def determine_ancestory(self):
		#TODO: UPDATE THE GPAPH EVERYTIME U CALL THIS. RIGHT NOW WHEN WE ADD A NEW CONNECTION
		# OR ENABLE A CONNECTION THE GRAPH IS NEVER UPDATED
		self.generate_graph()
		for n in self.nodes:
			n.parents = []
		list_of_nodes = list(self.graph.nodes)
		for n in list_of_nodes:
			ancestors = nx.ancestors(self.graph, n)
			corr_node = self.get_node_from_label(n)


			for ancestor in ancestors:
				corr_node.parents.append(self.get_node_from_label(ancestor))

	def get_node_from_label(self, label):
		for n in self.nodes:
			if n.label == label:
				return n 
		return None


	def reset_network(self):
		for node in self.nodes: 
			if not node.is_input:
				node.log += "RESET: \t Old Value: " + str(node.value) + "\n"
				node.value = 1
				node.evaluated = False
				node.times_called = 0




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
				new_con = input_node.add_output_node(output, 1)
				if new_con != None:
					self.weights.append(new_con)

	
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
				random_connection.weight = random.random() * 100 - 50
			return False
		#change back to .2
		elif coin_flip > .2:
			self.determine_ancestory()
			#enable or disable a weight
			rand_num = random.randint(0, len(self.weights) - 1)
			rand_connection = self.weights[rand_num]


			
			#check if the output node for the connection is an ancestor of the input node (the output contributes to the input, making a loop)
			if not rand_connection.enabled:
				while rand_connection.output_node.is_ancestor(rand_connection.input_node):
					#rand_connection.terminate(self)
					rand_num = random.randint(0, len(self.weights) - 1)
					rand_connection = self.weights[rand_num]

			
			inode = rand_connection.input_node
			onode = rand_connection.output_node
			#inode.mutation_log += "ENABLE/DISABLE CONNECTION: \t "+ rand_connection.to_string() + "\n Ancestor Check: " + str(inode.is_ancestor(onode)) + " Innovation: " + str(self.current_innovation) + "\n"
			rand_connection.enabled = not rand_connection.enabled
			return False
		#turn back to .1
		elif coin_flip > .1 or self.current_innovation == 0: # in the event that the network is just beginning
			# innovation number gets updated for each of the structural mutations
			#self.current_innovation += 1
			#adds a node, c , into the network by splitting a random edge a->b into two new edges a->c and c->b. Where weight(a->c) = 1 and 
			#weight(c->b) = weight(a->b)
			enabled_weights = [weight for weight in self.weights if weight.enabled]
			rand_num = random.randint(0, len(enabled_weights) - 1)
			random_connection = enabled_weights[rand_num]
			in_node = random_connection.input_node
			out_node = random_connection.output_node
			new_node = node("hidden"+str(len(self.nodes)))
			connection_1 = in_node.add_output_node(new_node, 1, self.current_innovation)
			connection_2 = new_node.add_output_node(out_node, random_connection.weight, self.current_innovation + 1)
			if connection_1 != None:
				self.weights.append(connection_1)
			if connection_2 != None:
				self.weights.append(connection_2)
			self.nodes.append(new_node)


			#self.determine_ancestory()
			random_connection.enabled = False
			return True

		elif False	:
			# innovation number gets updated for each of the structural mutations
			#self.current_innovation += 1

			self.determine_ancestory()
			#mutate the structutre of the network
			#for output in self.output_nodes:
			non_input_nodes = [node for node in self.nodes is not node.is_input]
			non_output_nodes = [node for node in self.nodes if not node.is_output]
			rand_num_1 = random_exception(len(non_output_nodes) - 1)
			random_input_node = non_output_nodes[rand_num_1]
			legal_nodes = [node for node in self.nodes if not random_input_node.is_ancestor(node) 
			and not random_input_node.is_connected_to_out(node) and not node.is_input]
			ban_list = []
			while len(legal_nodes) < 1:
				ban_list.append(rand_num_1)
				rand_num_1 = random_exception(len(non_output_nodes - 1), ban_list)
				random_input_node = non_output_nodes[rand_num_1]
				legal_nodes = [node for node in self.nodes if not random_input_node.is_ancestor(node) 
				and not random_input_node.is_connected_to_out(node) and not node.is_input]

			rand_num_2 = random_exception(len(legal_nodes) - 1)
			random_output_node = legal_nodes[rand_num_2]

			new_connection = random_input_node.add_output_node(random_node_2, 1, self.current_innovation)
			if new_connection != None:
				self.weights.append(new_connection)
			return False	

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

	def convert_to_genes(self, innovation):
		self.genes = []
		pointer = 0
		print(innovation)
		sorted_weights = sorted([weight for weight in self.weights if weight.innovation_number != 0 ], key = lambda x: x.innovation_number)
		initial_weights = [weight for weight in self.weights if weight.innovation_number == 0]
		for w in initial_weights:
			self.genes.append(gene(w.input_node, w.output_node, w.weight, w.enabled, w.innovation_number))

		for i in range(innovation + 1):
			if pointer < len(sorted_weights):	
				weight = sorted_weights[pointer]
				if weight.innovation_number != i:
					#print(weight.innovation_number, i, pointer)
					self.genes.append(None)
				else:
					self.genes.append(gene(weight.input_node, weight.output_node, weight.weight, weight.enabled, weight.innovation_number))
					pointer += 1
			else:
				self.genes.append(None)
	

	def add_new_genes(self):
		sorted_weights = sorted(self.weights, key = lambda x: x.innovation_number)
		for weight in sorted_weights:
			if weight.inovation_number > self.current_innovation:
				self.genes.append(gene(weight.input_node, weight.output_node, weight.weight, weight.enabled, weight.innovation_number))

	#def determine_fitness(self):

	


class gene:
	def __init__(self, input_node, output_node, weight, enabled, innovation_number):
		self.input = input_node
		self.output = output_node
		self.weight = weight
		self.enabled = enabled
		self.innovation = innovation_number

	def print_gene(self):
		#print("input: ", self.input.label)
		#print("output: ", self.output.label)
		#print("innotvaion number: ", self.innovation)
		#print("DISABLED \n\n\n" if not self.enabled else "\n\n\n")
		print(self.input.label, "->", self.output.label, "DISABLED" if not self.enabled else "", "Inovation_number: ", self.innovation , "\n")


def create_new_network_from_genes(genes):
	return_network = network()
	for gene in genes:
		if gene != None:
			in_node = gene.input
			out_node = gene.output
			new_in_node = None
			new_out_node = None

			found_in = False
			found_out = False
			# In the event that the input or output node is already in the network
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
			if new_in_node not in return_network.input_nodes and new_in_node.is_input:
				return_network.input_nodes.append(new_in_node)
			if new_out_node not in return_network.output_nodes and new_out_node.is_output:
				return_network.output_nodes.append(new_out_node)


			# This covers the weights attribute of the network
			new_connection = new_in_node.add_output_node(new_out_node, gene.weight, gene.innovation)
			if new_connection != None:
				new_connection.enabled = gene.enabled
				return_network.weights.append(new_connection)
			else:
				print("CREATE NETWORK FROM GENE: new_connection is None")

			#this will update the innovation number
			return_network.current_innovation = gene.innovation

	return_network.genes = genes
	return return_network

def are_networks_equal(X, Y):
	X.generate_graph()
	Y.generate_graph()

	graph1 = X.graph
	graph2 = Y.graph

	if not nx.is_isomorphic(graph1, graph2):
		print("not iso")
		return False

	for node in X.nodes:
	    for another_node in Y.nodes:
	        if node.label == another_node.label:
	            if node.value != another_node.value:
	            	return False
	return True

class Population:

	def __init__(self, size):
		self.size = size

	def reproduce(self, network1, network2):
		if network1.fitness > network2.fitness:
			parent1 = network1
			parent2 = network2
		else: 
			parent1 = network2
			parent2 = network1

		parent1.convert_to_genes()
		parent2.convert_to_genes()

		parent1_genes = parent1.genes
		parent2_genes = parent2.genes
		child_genes = []

		for gene1, gene2 in zip(parent1_genes, parent2_genes):
			#when both parents have genes on the same innovation number take one at random
			#to pass on to the child
			if gene1 != None and gene2 != None:
				if random.random() < .5:
					child_genes.append(gene1)
				else: 
					child_genes.append(gene2)
			#in the event that there is a mismatch only take the gene from the most fit parent
			elif gene2 == None and gene1 != None:
				child_genes.append(gene1)
			else:
				child_genes.append(None)
		return child_genes



