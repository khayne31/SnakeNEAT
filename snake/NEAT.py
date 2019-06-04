import numpy as np



class layer:
	def __init__(self):
		self.nodes = []
		self.prev_layer = None
		self.next_layer = None

	def add_node(self, node):
		self.nodes.append(node)

	def remove_node(self, node):
		self.nodes.remove(node)



class connection:
	def __init__(self, input_node, output_node, weight, enabled: bool):
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


class network:
	def __init__(self):
		self.head_layer_pointer = None #head pointer of a linked list of layers
		self.tail_layer_pointer = None
		self.nodes = []
		self.input_nodes = []
		self.output_nodes = []
		self.weights = []

	def initalize_network(self):
		self.head_layer_pointer = layer()
		self.tail_layer_pointer = layer()
		self.head_layer_pointer.next_layer = self.tail_layer_pointer
		self.tail_layer_pointer.prev_layer = self.head_layer_pointer

	def create_network(self, num_input_nodes: int, num_output_nodes: int, value_list: list = None):
		#The idea is to create a basic network with no hidden layers where the inputs connect directly to the outputs. This will be the basic network
		#The NEAT network will evolve from
		#the value_list parameter will be the values which go into the input nodes, in order. if no list is given the values default to 0
		values = value_list if value_list != None or len(value_list) != num_input_nodes else [0] * num_input_nodes
		for i in range(num_input_nodes):
			new_node = node(str(i), values[i], input_node = True)
			self.nodes.append(new_node)
			self.input_nodes.append(new_node)
		for i in range(num_output_nodes):
			new_node = node(str(i+num_input_nodes), output_node = True)
			self.nodes.append(new_node)
			self.output_node()






class gene:
	def __init__(self, input_node, output_node, weight, enabled, inovation_number):
		self.input = input_node
		self.output = output_node
		self.weight = weight
		self.enabled = enabled
		self.inovation = inovation_number

class node:
	def __init__(self, label, value = 0, input_node = False, output_node = False):
		self.value = value
		self.label = label
		self.connected_to_in = []
		self.connected_to_out = []
		self.is_input = input_node
		self.is_output = output_node
		self.evaluated = False

	def get_value(self):
		if self.is_input or self.evaluated:
			return self.value
		else:
			for connection in self.connected_to_in:
				if connection.enabled:
					self.value += connection.weight * connection.input_node.get_value()
			self.evaluated =True 

		return self.value
	def add_input_node(self, node, weight):
		connect = connection(node, self, weight, True)
		self.connected_to_in.append(connect)
		node.connected_to_out.append(connect)

	def add_output_node(self, node, weight):
		con = connection(self, node, weight, True)
		self.connected_to_out.append(con)
		node.connected_to_in.append(con)

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

