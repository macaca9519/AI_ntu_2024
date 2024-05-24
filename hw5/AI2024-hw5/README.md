# Homework 5

## Install Necessary Packages
conda create -n hw5 python=3.11 -y
conda activate hw5
pip install -r requirements.txt

## set up environment:
```
cd AI2024-hw5-v2
python3 -m venv ./vir
source ./vir/bin/activate
pip3 install -r requirement.txt
```
## execute the training
```
python pacman.py --save_root "./submissions"
```

## evaluation of my code
```
python pacman.py --eval --eval_model_path "./submissions/pacman_dqn.pt"
```