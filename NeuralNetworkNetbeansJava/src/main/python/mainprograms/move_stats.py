#! /usr/bin/python2.7

import shutil
import subprocess
import Utils as utils

for i in xrange(5):
    bits = 5
    name1 = "144_"+str(i)
    name2 = "145_"+str(i)
    name3 = "146_"+str(i)
    name4 = "147_"+str(i)

    network_name1 = "binary_adder_{}_bits_{}".format(bits, name1)
    network_name2 = "binary_adder_{}_bits_{}".format(bits, name2)
    network_name3 = "binary_adder_{}_bits_{}".format(bits, name3)
    network_name4 = "binary_adder_{}_bits_{}".format(bits, name4)

    full_name1 = "networks/" + network_name1
    full_name2 = "networks/" + network_name2
    full_name3 = "networks/" + network_name3
    full_name4 = "networks/" + network_name4

    utils.check_create_dir("networks/statistics")
    shutil.copy(full_name1+"/statistics_map.pkl.gz", "networks/statistics/statistics_map_"+name1+".pkl.gz")
    shutil.copy(full_name2+"/statistics_map.pkl.gz", "networks/statistics/statistics_map_"+name2+".pkl.gz")
    shutil.copy(full_name3+"/statistics_map.pkl.gz", "networks/statistics/statistics_map_"+name3+".pkl.gz")
    shutil.copy(full_name4+"/statistics_map.pkl.gz", "networks/statistics/statistics_map_"+name4+".pkl.gz")

    utils.check_create_dir("networks/sqrt_mean_squared_error")
    shutil.copy(full_name1+"/sqrt_mean_squared_error.png", "networks/sqrt_mean_squared_error/sqrt_mean_squared_error_"+name1+".png")
    shutil.copy(full_name2+"/sqrt_mean_squared_error.png", "networks/sqrt_mean_squared_error/sqrt_mean_squared_error_"+name2+".png")
    shutil.copy(full_name3+"/sqrt_mean_squared_error.png", "networks/sqrt_mean_squared_error/sqrt_mean_squared_error_"+name3+".png")
    shutil.copy(full_name4+"/sqrt_mean_squared_error.png", "networks/sqrt_mean_squared_error/sqrt_mean_squared_error_"+name4+".png")

    subprocess.call(["rm", "-rf", full_name1])
    subprocess.call(["rm", "-rf", full_name2])
    subprocess.call(["rm", "-rf", full_name3])
    subprocess.call(["rm", "-rf", full_name4])
