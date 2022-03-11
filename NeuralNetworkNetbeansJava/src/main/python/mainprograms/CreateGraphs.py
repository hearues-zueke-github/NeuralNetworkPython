#! /usr/bin/python2.7

from __init__ import *
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# import Utils as utils

# matplotlib.use("Agg")

def get_output_targets_binary_adder_plot_points(bits, iterations=0, show_not_save_file=True):
    # if iterations == 0:
    #     iter_str = ""
    # else:
    iter_str = "_"+str(iterations)+"_iterations"
    neuron_list = [bits*2, bits*3, bits+1]
    name_part = "".join(map(lambda x: "_"+str(x), neuron_list))

    nn = utils.load_pkl_file("lnn_binadder"+name_part+".pkl.gz")
    learning_rate = nn.get_learning_rate()

    inputs, targets = utils.load_pkl_file("inputs_targets_binary_"+str(bits)+"_bits.pkl.gz")
    fig = nn.show_outputs_targets_plot_points(inputs, targets, should_show=False)
    plt.figure(fig.number)
    plt.title("Output/Target Graph for a "+str(bits)+" bits binary adder\nNetwork Layers: "+str(neuron_list)+"; Learning rate: "+str(learning_rate)+"; "+str(iterations)+" iterations")
    
    pic_dir = "pictures"
    out_targ_dir = "output_targets_"+str(bits)+"_bits"

    directory = pic_dir
    utils.check_create_dir(directory)

    directory += "/" + out_targ_dir
    utils.check_create_dir(directory)

    if show_not_save_file:
        plt.show(block=False)
    else:
        plt.savefig(pic_dir+"/"+out_targ_dir+"/output_targest_"+str(bits)+"_bits"+name_part+iter_str+".eps", format="eps")
