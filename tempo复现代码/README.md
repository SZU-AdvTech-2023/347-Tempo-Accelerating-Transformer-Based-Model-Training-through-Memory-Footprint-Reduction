1. 构建docker
  sudo docker build -t tempo -f Dockerfile .
2. 运行容器
  sudo docker run --gpus all -it --rm --ipc=host --shm-size=1g --ulimit memlock=-1 --name="tempo" -v $(pwd):/Tempo/ tempo
3. 安装
  cd /Tempo/
  python setup.py install
4. 进入bert目录
  cd Bert-Chinese-Text-Classification-Pytorch
5. 下载checkpoints
  bert模型放在 bert_pretain目录下，三个文件：
  pytorch_model.bin
  bert_config.json
  vocab.txt
6. 运行
  # 训练并测试：
  # bert
  python run.py --model bert
