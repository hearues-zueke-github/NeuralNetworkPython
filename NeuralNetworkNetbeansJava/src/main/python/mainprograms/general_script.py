#! /usr/bin/python2.7

import subprocess
import sys

program_iterations = 1
iterations = str(6)
name_suffix = sys.argv[1]
is_cont = "0"
if len(sys.argv) > 2:
    is_cont = sys.argv[2]
offset = 0

func = "sigmoid"
type = "sgd"
is_adapt = 1
is_no_random = 0

name_suffixes = ["1", "2", "3", "4", "5", "6", "7", "8"]
params = {
    name_suffixes[0]: {"nl": "[28*28, 28*1, 10]", "max_e": "10.", "min_e": "0.000005", "adpt_p": "1.02", "adpt_n": "0.8", "f": "sigmoid", "type": "sgd", "adpt": "1", "norand": "0"},
    name_suffixes[1]: {"nl": "[28*28, 28*2, 10]", "max_e": "10.", "min_e": "0.00005", "adpt_p": "1.03", "adpt_n": "0.8", "f": "sigmoid", "type": "sgd", "adpt": "1", "norand": "0"},
    name_suffixes[2]: {"nl": "[28*28, 28*3, 10]", "max_e": "0.2", "min_e": "0.0005", "adpt_p": "1.02", "adpt_n": "0.75", "f": "sigmoid", "type": "sgd", "adpt": "1", "norand": "0"},
    name_suffixes[3]: {"nl": "[28*28, 28*4, 10]", "max_e": "0.2", "min_e": "0.0005", "adpt_p": "1.02", "adpt_n": "0.75", "f": "sigmoid", "type": "sgd", "adpt": "1", "norand": "0"},
    name_suffixes[4]: {"nl": "[28*28, 28*5, 10]", "max_e": "0.2", "min_e": "0.0005", "adpt_p": "1.02", "adpt_n": "0.75", "f": "sigmoid", "type": "sgd", "adpt": "1", "norand": "0"},
    name_suffixes[5]: {"nl": "[28*28, 28*3, 28*2, 10]", "max_e": "0.2", "min_e": "0.0000005", "adpt_p": "1.02", "adpt_n": "0.75", "f": "sigmoid", "type": "sgd", "adpt": "1", "norand": "0"},
    name_suffixes[6]: {"nl": "[28*28, 28*4, 28*2, 10]", "max_e": "0.2", "min_e": "0.0000005", "adpt_p": "1.02", "adpt_n": "0.75", "f": "sigmoid", "type": "sgd", "adpt": "1", "norand": "0"},
    name_suffixes[7]: {"nl": "[28*28, 28*4, 28*3, 10]", "max_e": "0.2", "min_e": "0.0000005", "adpt_p": "1.02", "adpt_n": "0.75", "f": "sigmoid", "type": "sgd", "adpt": "1", "norand": "0"}
}

if name_suffix in name_suffixes:
    one_params = params[name_suffix]
    nl = one_params["nl"]
    max_e = one_params["max_e"]
    min_e = one_params["min_e"]
    adpt_p = one_params["adpt_p"]
    adpt_n = one_params["adpt_n"]
    f = one_params["f"]
    type = one_params["type"]
    adpt = one_params["adpt"]
    norand = one_params["norand"]
    for i in xrange(program_iterations):
        subprocess.call(["./create_network_folder.py",
                         "digitmany",
                         "nl_"+"_".join(map(str, eval(nl)))+
                         "_max_e_"+max_e+
                         "_min_e_"+min_e+
                         "_adpt_p_"+adpt_p+
                         "_adpt_n_"+adpt_n+
                         "_f_"+f+
                         "_type_"+type+
                         "_adpt_"+adpt+
                         "_norand_"+norand+"_"+str(i+offset), iterations, is_cont, nl, max_e, min_e, adpt_p, adpt_n, f, type, adpt, norand])
