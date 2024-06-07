reproducibility
===

cuda==V12.2.140

run on NVIDIA GeForce RTX 2080

Driver Version: 535.104.05 
```bash
conda create -y -n ai_hw6 python=3.10
conda activate ai_hw6
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install --no-deps trl peft accelerate bitsandbytes
pip install tqdm packaging wandb

#sloth
conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

pip install -U accelerate

```
training command:
```sh
bash run.sh DPO unsloth/tinyllama-bnb-4bit a6eef608269715d8f6f6037168e426e457226e5f

bash run.sh ORPO unsloth/tinyllama-bnb-4bit a6eef608269715d8f6f6037168e426e457226e5f
```