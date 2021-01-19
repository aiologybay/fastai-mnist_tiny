from fastai.vision import *

workspace = os.getcwd()
pkl_path = os.path.join(workspace, 'models')
pkl_name = 'weight.pth'

mnist = './mnist_tiny'
tfms = get_transforms(do_flip=False)
data = (ImageList.from_folder(mnist)
        .split_by_folder()
        .label_from_folder()
        .transform(tfms, size=32)
        .databunch()
        .normalize(imagenet_stats))

# Show pictures before training
data.show_batch(row=3, figsize=(4, 4))
plt.pause(2)
plt.close()


def train():
    learn = cnn_learner(data, models.resnet18, metrics=accuracy)  # create a learner
    learn.fit(100)
    if not os.path.isdir('models'):
        os.mkdir('models')

    learn.export(pkl_path + '/' + pkl_name)


def test():
    learn = load_learner(pkl_path, file=pkl_name)
    test_path = os.path.join(mnist, 'test')
    test_list = os.listdir(test_path)
    for i in test_list:
        img = open_image(os.path.join(test_path, i))
        pred_cls, pred_idx, outputs = learn.predict(img)
        print(i, pred_cls)
        with open('./mnist_tiny/pred.txt', 'a+', encoding='utf-8') as f:
            f.write(i + ' ' + str(pred_cls) + '\n')
        f.close()


if __name__ == '__main__':
    train()
    test()

