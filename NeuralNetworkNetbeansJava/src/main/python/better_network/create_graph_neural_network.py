#! /usr/bin/python3.5

nl = [3, 4, 5]

with open("new_graph.dot", "w") as f:
    f.write("graph {\n")

    f.write("splines=line;\n")
    for i, n in enumerate(nl):
        f.write("  "*1+"subgraph cluster_{} {\n".format(i))
        for j in range(0, n):
            f.write("  "*2+"n_{}_{} [label=\"\"];\n".format())
        f.write("  "*1+"}\n")

"""graph { 
        splines=line; 
        subgraph cluster_0 { 
            label="Subgraph A"; 
            a; b; c
        } 
 
        subgraph cluster_1 { 
            label="Subgraph B"; 
            d; e;
        }

        a -- e; 
        a -- d; 
        b -- d; 
        b -- e; 
        c -- d; 
        c -- e; 
    }"""

    f.write("}\n")
