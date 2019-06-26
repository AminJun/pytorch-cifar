#Uskerim
#gpu 1
source `which virtualenvwrapper.sh`
workon certifiable
cd /home/alishafahi/AJ/FastTraining

mkdir -p "outs"
mkdir -p "logs"
fast=FAST
epochs=EPOC
name="${fast}_${epochs}"
python main.py --fast "${fast}" --epochs "${epochs}" > "outs/${name}.out" & 
