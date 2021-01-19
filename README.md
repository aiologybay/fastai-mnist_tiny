# About this project
This project is base on fastai1.0, please read requirements.txt to get the information about these packages. This project is not only used to train MNIST_TINY dataset, but also to test and draw paramrtric curves. All these codes are based on Linux(Ubuntu18.04) and train by cpu.

## Main packages version and manual installation
```bash
pip3 install torch==1.7.1 torchvision==0.8.2 fastai==1.0.61
```
## Init the project
I have completed this project and generated some files, if you want to do it from the benginning, please enter the following cammand at the terminal:
```bash
rm -rf log.txt Result_Analysis.png models mnist_tiny/pred.txt
```

## Train and test
```bash
python3 train.py
```

## Draw curves of loss and accuracy
1. Create a file named 'log.txt'.
```bash
touch log.txt
```

2. Copy the outputs of terminal to 'log.txt', it's format such as following:
epoch     train_loss  valid_loss  accuracy  time
0         0.653680    0.413430    0.844063  00:03
1         0.517697    0.445358    0.814020  00:02
2         0.453159    0.396640    0.844063  00:02

2. Run draw_curves.py.
```bash
python3 draw_curves.py
```
## Compare test.txt with pred.txt
```bash
cd mnist_tiny
sort -r test.txt -o test.txt
sort -r pred.txt -o pred.txt
diff test.txt pred.txt
```
Then you'll see which part pictures can't be identified.
This result will ask you to evaluate model correctly.
