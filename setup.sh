#! /bin/bash

if [-f .env]; then
    echo "Loading .env file"
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found! Exiting."
    exit 1
fi

echo "--------- Starting Environment Setup ---------"
pip3 install --upgrade pip
pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install matplotlib \
             Pillow \
             tqdm \
             einops \
             PyYAML \
             tensorboardX \
             opencv-python \
             scipy \
             "numpy<2.0" \
             cityscapesscripts

echo "--------- Library Installation Complete ---------"

echo "--------- Install Cityscapes dataset ---------"

mkdir -P /workspace/data/cityscapes
cd /workspace/data/cityscapes

csDownload gtFine_trainvaltest.zip
csDownload leftImg8bit_trainvaltest.zip

unzip gtFine_trainvaltest.zip && rm gtFine_trainvaltest.zip
unzip leftImg8bit_trainvaltest.zip && rm leftImg8bit_trainvaltest.zip

echo "--------- Complete Cityscapes Installation ---------"


echo "--------- Setup Project Directory ---------"

mkdir -P /workspace/projects
cd /workspace/projects
git clone https://github.com/Semantic-Segmentation-Research/semi-supervised-semantic-segmentation.git

echo "--------- Complete Project Setup ---------"

