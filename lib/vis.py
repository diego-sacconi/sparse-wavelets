import networkx
import math
import scipy.optimize
import numpy
import sys
from scipy import linalg
import matplotlib.pyplot as plt
from IPython.display import Image
import pywt
import scipy.fftpack
import random
import operator
import copy
from collections import deque
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
import os

def set_f(G,F,ids=None):
	if ids is None:
		i = 0
		for v in G.nodes():
			G.node[v]["value"] = F[i]
			i = i + 1
	else:
		i = 0
		for v in range(len(ids)):
			G.node[ids[v]]["value"] = F[i]
			i = i + 1

def get_f(G):
	F = []
	for v in G.nodes():
		F.append(G.node[v]["value"])
	
	return numpy.array(F)

def rgb_to_hex(r,g,b):
	return '#%02x%02x%02x' % (r,g,b)

def rgb(minimum, maximum, value):
	mi, ma = float(minimum), float(maximum)
	ratio = 2 * (value-mi) / (ma - mi)
	b = int(max(0, 255*(1 - ratio)))
	r = int(max(0, 255*(ratio - 1)))
	g = 255 - b - r
	
	return rgb_to_hex(r, g, b)

def draw_graph_with_values(G, dot_output_file_name, maximum=None, minimum=None):
	output_file = open(dot_output_file_name, 'w')
	output_file.write("graph G{\n")
	output_file.write("rankdir=\"LR\";\n")
	output_file.write("size=\"10,2\";\n")

	if maximum is None:
		maximum = -sys.float_info.max
		minimum = sys.float_info.max

		for v in G.nodes():
			if G.node[v]["value"] > maximum:
				maximum = G.node[v]["value"]
	     
			if G.node[v]["value"] < minimum:
		 		minimum = G.node[v]["value"]

	for v in G.nodes():
		color = rgb(minimum, maximum, G.node[v]["value"])
		if G.node[v]["value"] != 0.0:
			output_file.write("\""+str(v)+"\" [shape=\"circle\",label=\"\",style=filled,fillcolor=\""+str(color)+"\",penwidth=\"2\",fixedsize=true,width=\"1\",height=\"1\"];\n")
		else:
			output_file.write("\""+str(v)+"\" [shape=\"circle\",label=\"\",style=filled,fillcolor=\""+str(color)+"\",penwidth=\"0\",fixedsize=true,width=\"1\",height=\"1\"];\n")

	for edge in G.edges():
		output_file.write("\""+str(edge[0])+"\" -- \""+str(edge[1])+"\"[dir=\"none\",color=\"black\",penwidth=\"1\"];\n")

	
	output_file.write("}")

	output_file.close()

def draw_graph_dynamic_values(G, FT, fig_output_file_name):
	maximum = -sys.float_info.max
	minimum = sys.float_info.max

	for i in range(FT.shape[0]):
		for j in range(FT.shape[1]):
			if FT[i][j] > maximum:
				maximum = FT[i][j]
			
			if FT[i][j] < minimum:
				minimum = FT[i][j]

	svg_names = ""

	for i in range(FT.shape[0]):
		set_f(G, FT[i])
		#draw_graph_with_values(G, "dyn_graph-"+str(i)+".dot", maximum, minimum)
		draw_graph_with_values(G, "dyn_graph-"+str(i)+".dot")
		os.system("sfdp -Goverlap=prism -Tsvg dyn_graph-"+str(i)+".dot > dyn_graph-"+str(i)+".svg")
		os.system("rm dyn_graph-"+str(i)+".dot")
		svg_names = svg_names + " dyn_graph-"+str(i)+".svg"
		
	os.system("python lib/svg_stack-master/svg_stack.py --direction=v --margin=0 "+svg_names+" > "+fig_output_file_name)

	for i in range(FT.shape[0]):
		os.system("rm dyn_graph-"+str(i)+".svg")
	
def draw_partitions_with_values(G, partitions, dot_output_file_name, maximum=None, minimum=None):
	output_file = open(dot_output_file_name, 'w')
	output_file.write("graph G{\n")
	output_file.write("rankdir=\"LR\";\n")
	output_file.write("size=\"10,2\";\n")

	if maximum is None:
		maximum = -sys.float_info.max
		minimum = sys.float_info.max

		for v in G.nodes():
			if G.node[v]["value"] > maximum:
				maximum = G.node[v]["value"]
	     
			if G.node[v]["value"] < minimum:
		 		minimum = G.node[v]["value"]
	
	part_map = {}

	for p in range(len(partitions)):
		for i in range(len(partitions[p])):
			part_map[partitions[p][i]] = p

	for v in G.nodes():
		color = rgb(minimum, maximum, G.node[v]["value"])
		if G.node[v]["value"] != 0.0:
			output_file.write("\""+str(v)+"\" [shape=\"circle\",label=\"\",style=filled,fillcolor=\""+str(color)+"\",penwidth=\"2\",fixedsize=true,width=\"0.5\",height=\"0.5\"];\n")
		else:
			output_file.write("\""+str(v)+"\" [shape=\"circle\",label=\"\",style=filled,fillcolor=\""+str(color)+"\",penwidth=\"0\",fixedsize=true,width=\"0.5\",height=\"0.5\"];\n")

	for edge in G.edges():
		if part_map[edge[0]] == part_map[edge[1]]:
			output_file.write("\""+str(edge[0])+"\" -- \""+str(edge[1])+"\"[dir=\"none\",color=\"black\",penwidth=\"4\"];\n")
		else:
			output_file.write("\""+str(edge[0])+"\" -- \""+str(edge[1])+"\"[dir=\"none\",color=\"black\",penwidth=\"1\"];\n")
	
	output_file.write("}")

	output_file.close()

def draw_graph(G, dot_output_file_name):
	output_file = open(dot_output_file_name, 'w')
	output_file.write("graph G{\n")
	output_file.write("rankdir=\"LR\";\n")
	output_file.write("size=\"10,2\";\n")

	for v in G.nodes():
		color = rgb_to_hex(0,255,0)
		output_file.write("\""+str(v)+"\" [shape=\"circle\",label=\"\",style=filled,fillcolor=\""+str(color)+"\",penwidth=\"2\",fixedsize=true,width=\"1\",height=\"1\"];\n")

	for edge in G.edges():
		output_file.write("\""+str(edge[0])+"\" -- \""+str(edge[1])+"\"[dir=\"none\",color=\"black\",penwidth=\"1\"];\n")

	
	output_file.write("}")

	output_file.close()

def draw_time_graph(G, fig_output_file_name):
	svg_names = ""

	for i in range(G.num_snaps()):
		draw_graph(G.snap(i), "graph-"+str(i)+".dot")
		os.system("sfdp -Goverlap=prism -Tsvg graph-"+str(i)+".dot > graph-"+str(i)+".svg")
		os.system("rm graph-"+str(i)+".dot")
		svg_names = svg_names + " graph-"+str(i)+".svg"
		
	os.system("python lib/svg_stack-master/svg_stack.py --direction=v --margin=0 "+svg_names+" > "+fig_output_file_name)

	for i in range(G.num_snaps()):
		os.system("rm graph-"+str(i)+".svg")

def draw_time_graph_eig(G, eig, fig_output_file_name):
	svg_names = ""
	G.set_values(eig)
	maximum = numpy.max(eig)
	minimum = numpy.min(eig)

	for i in range(G.num_snaps()):
		draw_graph_with_values(G.snap(i),"graph-"+str(i)+".dot", minimum, maximum)
		os.system("sfdp -Goverlap=prism -Tsvg graph-"+str(i)+".dot > graph-"+str(i)+".svg")
		os.system("rm graph-"+str(i)+".dot")
		svg_names = svg_names + " graph-"+str(i)+".svg"
		
	os.system("python lib/svg_stack-master/svg_stack.py --direction=v --margin=0 "+svg_names+" > "+fig_output_file_name)

	for i in range(G.num_snaps()):
		os.system("rm graph-"+str(i)+".svg")

