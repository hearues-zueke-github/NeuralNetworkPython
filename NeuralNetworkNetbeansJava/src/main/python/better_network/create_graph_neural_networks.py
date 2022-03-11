#! /usr/bin/python2.7

import numpy as np

nl = [6, 7, 7, 3]

graph = {i: [] for i in xrange(-1, len(nl)+1)}

colors = ["blue4", "red2", "seagreen2"]

spaces = "    "
def get_subgraph_cluster_input_output(layer, num_neur, is_input_layer=True):
    string = ""
    layer_str = str(layer).replace("-", "_")
    string += spaces*1     + "subgraph cluster_0_{}".format(layer_str)+" {\n"
    string += spaces*2 + "color=white;\n"
    string += spaces*2 + "node [style=solid, color=\"#00000000\", shape=circle, width=0.3, fontsize=12, fixedsize=true];\n"
    for i in xrange(0, num_neur):
        graph[layer].append("x_{}_{}".format(layer_str, i))
    string += spaces*2 + " ".join(graph[layer]) + ";\n"
    string += spaces*2 + 'label = ""\n'
    string += spaces*1     + "}\n"

    return string

def get_subgraph_cluster(layer, num_neur, color, start_spaces=0, label="", picture_path="", is_output_layer=False):
    string = ""
    layer_str = str(layer).replace("-", "_")
    string += spaces*start_spaces     + "subgraph cluster_"+str(layer)+" {\n"
    string += spaces*(start_spaces+1) + "color=white;\n"
    string += spaces*(start_spaces+1) + "node [style=solid, color={}, shape=circle, width=0.3, fixedsize=true];\n".format(color)
    for i in xrange(0, num_neur):
        graph[layer].append("x{}_{}".format(layer, i))
    if picture_path != "":
        for x in graph[layer]:
            string += spaces*(start_spaces+1)+x+"  [scale=15, label=<<TABLE><tr><td color=\"#12345600\" fixedsize=\"true\" width=\"28\" height=\"28\"><img src=\"{}\" /></td></tr></TABLE>>];\n".format(picture_path)
    else:
        string += spaces*(start_spaces+1) + " ".join(graph[layer]) + ";\n"
    string += spaces*(start_spaces+1) + 'label = "{}"\n'.format(label)
    if not is_output_layer:
        string += spaces*(start_spaces+1)+"b_{} [color={}, label=\"b_{}\", fontsize=10];\n".format(layer_str, colors[0] if layer == 0 else colors[1], layer)
    string += spaces*start_spaces+"}\n"

    return string

def get_connection_inner_layers(nl):
    string = ""
    idxs = np.arange(0, len(nl))
    for l1, l2 in zip(idxs[:-1], idxs[1:]):
        n1 = graph[l1]
        n2 = graph[l2]
        string += "\n"
        for x1 in n1:
            for x2 in n2:
                string += spaces+"{} -> {};\n".format(x1, x2)

    return string

def get_connection_for_input_output(nl):
    string = ""
    gi, go = graph[-1], graph[0]
    for xi, xj in zip(gi, go):
        string += spaces+"{} -> {} [constraint=true];\n".format(xi, xj)
        # string += spaces+"{} -> {} [dir=\"arrowhead\"];\n".format(xi, xj)
    string += "\n"
    gi, go = graph[len(nl)-1], graph[len(nl)]
    for xi, xj in zip(gi, go):
        string += spaces+"{} -> {} [constraint=true];\n".format(xi, xj)
        # string += spaces+"{} -> {} [dir=\"arrowhead\"];\n".format(xi, xj)

    return string

def get_connection_biases(nl):
    string = "\n"
    for i, n in enumerate(nl[1:]):
        for j in xrange(0, n):
            string += spaces+"b_{} -> x{}_{} [constraint=false];\n".format(i, i+1, j)

    return string

with open("new_graph_2.dot", "w") as fout:
    fout.write("digraph {\n")
    # fout.write("graph {\n")

    fout.write(spaces*1+"rankdir=LR;\n")
    fout.write(spaces*1+"splines=line;\n")
    fout.write(spaces*1+"nodesep=.05;\n")
    fout.write(spaces*1+'node [label=""];\n')

    fout.write("\n"+get_subgraph_cluster_input_output(-1, nl[0], is_input_layer=True))
    fout.write("\n"+get_subgraph_cluster(0, nl[0], colors[0], 1, label="Input\\nLayer"))
    for i in xrange(1, len(nl)-1):
        fout.write("\n"+get_subgraph_cluster(i, nl[i], "\"#00000000\"", 1, label="Hidden\\nLayer {}".format(i), picture_path="node_activation_red_{}.png".format(i)))
    fout.write("\n"+get_subgraph_cluster(len(nl)-1, nl[-1], "\"#00000000\"", 1, label="Output\\nLayer", picture_path="node_activation_green.png", is_output_layer=True))
    fout.write("\n"+get_subgraph_cluster_input_output(len(nl), nl[-1], is_input_layer=False))
    
    g = graph[-1]
    fout.write("\n")
    for i, x in enumerate(g):
        fout.write(spaces+x+" [label=\"x_{}\"];\n".format(i+1))

    g = graph[len(nl)]
    fout.write("\n")
    for i, y in enumerate(g):
        fout.write(spaces+y+" [label=\"y_{}\"];\n".format(i+1))

    fout.write("\n"+spaces+"edge [dir=none]\n")
    fout.write(get_connection_inner_layers(nl))
    fout.write(get_connection_biases(nl))
    fout.write("\n"+spaces+"edge [dir=\"arrowhead\"]\n")
    fout.write(get_connection_for_input_output(nl))

    fout.write("}\n")
