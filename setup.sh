mkdir -p /home/dev/data

git clone https://github.com/Semantic-Segmentation-Research/ssl-semantic-segmentation.git
cd /home/dev/ssl-semantic-segmentation

git config --global init.defaultBranch main
git config --local user.name MKHan91
git config --local user.email audrbz@naver.com

pip install gdown
gdown --id 10ibJq8sEUK-KKO4jqNTcdl2p5ohp1Kso -O data.zip
unzip data.zip -d /home/dev/data