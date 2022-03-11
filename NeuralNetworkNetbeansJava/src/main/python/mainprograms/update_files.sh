#! /bin/bash

path_temp="/home/haris/temp_folder"
path_computer="/home/haris/Dropbox/uni/bakk/programs"
path_server="ziko@figipc180.tu-graz.ac.at:/calc/students/ziko/fusessh"

function check_files_equals {
    file=$1

    file="testtext.txt"
    sshpass -p $pass scp $path_server"/"$file $path_temp #computer"/temp_folder"
    a=($(md5sum $path_computer"/"$file))
    b=($(md5sum $path_temp"/"$file))

    rm $path_temp"/"$file

    if [ "$a" != "$b" ]; then
        echo "0"
        return
    fi

    echo "1"
}

all_files=( #"testtext.txt"
           "NeuralNetworkDecimalMultiprocess.py"
           "InitShell2.py"
           "InitShell5Set_lnn_14x14_network.py"
           "DigitRecognition"
           "lnn_14x14_statistics.txt"
           "lnn_14x14_autoencoder_statistics_backup.txt"
           "TrainedNetwork.py"
           "Utils.py"
           "create_autoencoder_inputs_targets.py"
           "lnn_14x14.pkl.gz"
           "lnn_14x14_autoencoder.pkl.gz")

pass="5wsdvTDBwtx"

val="$1"

if [ ! -d $path_temp ]; then
    mkdir $path_temp
fi

if [ "$val" == "toserver" ]; then
    echo "toserver"
    echo "server = "${path_server}
    
    for file in "${all_files[@]}"; do
        value=$(check_files_equals $file)
        echo $value
        if [ $value = "0" ]; then
            sshpass -p $pass scp $path_computer"/"$file $path_server
            echo "copying file form server to computer"
        else
            echo "Files are equals!"
        fi
    done
elif [ "$val" == "tocomputer" ]; then
    echo "tocomputer"
    echo "computer = "${path_computer}

    for file in "${all_files[@]}"; do
        value=$(check_files_equals $file)
        echo $value
        if [ $value = "0" ]; then
            sshpass -p $pass scp $path_server"/"$file $path_computer
            echo "copying file form server to computer"
        else
            echo "Files are equals!"
        fi
    done
else
    echo "NOthing!!!"
fi

rmdir $path_temp
