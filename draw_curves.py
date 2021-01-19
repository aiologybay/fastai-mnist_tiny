import matplotlib.pyplot as plt

# Create list to save data
epoch = []
train_loss = []
valid_loss = []
accuracy = []
with open('./log.txt', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        epo, tloss, vloss, acc = line.rsplit()[0], line.rsplit()[1], line.rsplit()[2], line.rsplit()[3]  # Split data
        epoch.append(int(epo))
        train_loss.append(float(tloss))
        valid_loss.append(float(vloss))
        accuracy.append(float(acc))
    f.close()

# Draw them
plt.title('Result Analysis')
plt.xlabel('epoch')
plt.ylabel('rate')
plt.ylim([0,1])
plt.plot(epoch, train_loss, color='green', label='train_loss')
plt.plot(epoch, valid_loss, color='red', label='valid_loss')
plt.plot(epoch, accuracy, color='blue', label='accuracy')
plt.legend()
plt.savefig('Result_Analysis.png')
plt.show()

