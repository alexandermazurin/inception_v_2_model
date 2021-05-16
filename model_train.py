import numpy as np
import torch
import os
import torch.optim as optimizers
import torch.nn as nn
from torch_inception import Inception_Net

dataset_path = '/Users/a18651298/EEG_SD_Data/val_0aug/{}/'
PATH = 'ready_net.pth'
num_classes = 5
samples_number_per_word = 100
number_of_epochs = 1000
batch_size = 10


def load_tensors_from_dirs_to_data(data, dir_list, root_dir):
    counter = 0
    for file_name in dir_list:
        if counter < samples_number_per_word:
            word_tensor = torch.load(os.path.join(root_dir, file_name))
            data.append(word_tensor)
            counter += 1
    return data


def form_target_vectors(y_dataset):
    for i in range(len(y_dataset)):
        vector = np.zeros(num_classes)
        vector[int(y_dataset[i]) - 1] = 1.
        y_dataset[i] = np.reshape(vector, newshape=num_classes)
    return torch.from_numpy(np.array(y_dataset))


def load_dataset():
    x_dataset = []
    y_dataset = []
    for i in range(num_classes):
        input_dir = dataset_path.format(str(i+1))
        dir_list = os.listdir(input_dir)
        x_dataset = load_tensors_from_dirs_to_data(x_dataset, dir_list, input_dir)
        for j in range(samples_number_per_word):
            y_dataset.append(i+1)
    y_dataset = form_target_vectors(y_dataset)
    return x_dataset, y_dataset


def data_split(data_list, percentage):
    border = int(len(data_list) * (1 - percentage))
    data_train = data_list[:border]
    data_test = data_list[border:]
    return data_train, data_test


def convert_list_of_tensors_to_numpy(tensor_list):
    new_array = np.zeros(shape=(len(tensor_list),
                                tensor_list[0].shape[0], tensor_list[0].shape[1], tensor_list[0].shape[2]))
    for i in range(len(tensor_list)):
        new_array[i] = tensor_list[i]
    return new_array


def shuffle(x_list, y_list):
    indice = np.random.permutation(len(x_list))
    x_new = [x_list[i] for i in indice]
    y_new = [y_list[i] for i in indice]
    return x_new, y_new


def accuracy(x_test_data, y_test_data, model):
    x_tensors = torch.from_numpy(x_test_data)
    total = x_test_data.shape[0]
    correct = 0
    for i in range(total):
        cur_x = x_tensors[i]
        cur_y = y_test_data[i]
        cur_x = cur_x.unsqueeze(0)
        cur_y = cur_y.unsqueeze(0)
        y_pred = model(cur_x)
        _, y_pred = torch.max(y_pred.data, 1)
        _, cur_y = torch.max(cur_y.data, 1)
        correct += int((y_pred.item() == cur_y.item()))
    return 100*correct/total


x_data, y_data = load_dataset()
x_data, y_data = shuffle(x_data, y_data)
x_data_train, x_data_test = data_split(x_data, 1 - 0.9)
y_data_train, y_data_test = data_split(y_data, 1 - 0.9)


if __name__ == '__main__':
    x_data_train = convert_list_of_tensors_to_numpy(x_data_train).astype(np.float32)
    x_data_test = convert_list_of_tensors_to_numpy(x_data_test).astype(np.float32)

    net = Inception_Net(dropout_rate=0.3, num_classes=num_classes)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optimizers.Adam(net.parameters(), lr=0.001)

    for epoch in range(number_of_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                order = np.random.permutation(x_data_train.shape[0])
                for start_index in range(0, x_data_train.shape[0], batch_size):
                    optimizer.zero_grad()
                    batch_indice = order[start_index: start_index + batch_size]

                    x_batch = torch.from_numpy(np.array([np.array(x_data_train[i]) for i in batch_indice]))
                    y_batch = torch.from_numpy(np.array([np.array(y_data_train[i]) for i in batch_indice]))

                    outputs = net(x_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
            else:
                net.eval()
                with torch.no_grad():
                    current_acc = accuracy(x_data_test, y_data_test, net)
                    if current_acc > 50.0:
                        torch.save(net.state_dict(), PATH)
                    if epoch % 100 == 0:
                        torch.save(net.state_dict(), PATH)
                    print('Epoch: ' + str(epoch + 1) +
                          ' Loss: ' + str(loss.item() / 2000) +
                          ' Accuracy: ' + str(current_acc))
    print('Finished Training')
    torch.save(net.state_dict(), PATH)
