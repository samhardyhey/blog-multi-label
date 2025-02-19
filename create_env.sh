conda config --set auto_activate_base false

echo "Creating p38 conda env"
conda create -n p38 python=3.8 -y
eval "$(conda shell.bash hook)"
conda activate p38

echo "Installing project requirements"
pip install -r ./requirements.txt

echo "Installing torch/conda binaries"
conda remove pytorch torchvision torchaudio -y # silly runpod env
conda install pytorch torchvision torchaudio -c pytorch -y

echo "Testing torch installation"
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'

echo "Git LFS"
apt-get update
apt-get install git-lfs

echo "Add git config"
git config --global user.name "Sam Hardy"
git config --global user.email "samhardyhey@gmail.com"