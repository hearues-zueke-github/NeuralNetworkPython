#! /bin/bash

path1="/home/haris/Dropbox/uni/bakk/programs/"
path2="ziko@figipc180.tu-graz.ac.at:/calc/students/ziko/fusessh"

pass="5wsdvTDBwtx"

file="InitShell2.py"
sshpass -p $pass scp $path1$file $path2

file="get_file_size.sh"
sshpass -p $pass scp $path1$file $path2

file="DigitRecognition.py"
sshpass -p $pass scp $path1$file $path2

file="InitShell5Set_lnn_14x14_network.py"
sshpass -p $pass scp $path1$file $path2

file="NeuralNetworkDecimalMultiprocess.py"
sshpass -p $pass scp $path1$file $path2

file="TrainedNetwork.py"
sshpass -p $pass scp $path1$file $path2
