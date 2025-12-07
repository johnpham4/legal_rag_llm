```shell
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
sudo apt update

sudo apt install -y google-chrome-stable

poetry env use /home/minhpham/miniconda3/envs/llm/bin/python
poetry install --without aws

mkdir -p ./data/mongo_data
sudo chown -R 999:999 ./data/mongo_data
```