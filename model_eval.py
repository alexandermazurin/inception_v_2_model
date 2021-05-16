import torch
from torch_inception import Inception_Net
from model_train import x_data_test, y_data_test, convert_list_of_tensors_to_tensor
num_classes = 5

dataset_path = '/Users/a18651298/EEG_SD_Data/val_0aug/{}'
PATH = 'ready_net.pth'


def get_batches(input_list_x, input_list_y, batch_size=batch_size):
    batches_x = torch.zeros(int(len(input_list_x)/batch_size),
                            batch_size, input_list_x[0].shape[0], input_list_x[0].shape[1], input_list_x[0].shape[2])
    batches_y = torch.zeros(int(len(input_list_x)/batch_size),
                            batch_size, input_list_y[0].shape[0])
    list_length = len(input_list_x)
    indices = [i for i in range(list_length)]
    counter = 0
    while len(indices):
        chosen_indices = random.sample(indices, batch_size)
        batch_x = torch.from_numpy(np.array([np.array(input_list_x[i]) for i in chosen_indices]))
        batch_y = torch.from_numpy(np.array([np.array(input_list_y[i]) for i in chosen_indices]))
        batches_x[counter] = batch_x
        batches_y[counter] = batch_y
        for index in chosen_indices:
            indices.remove(index)
        counter += 1
    return batches_x, batches_y


def accuracy_on_batch(x_batch, y_batch, model):
    total = len(x_batch)
    correct = 0
    for i in range(len(x_batch)):
        cur_x = x_batch[i]
        cur_y = y_batch[i]
        cur_x = cur_x.unsqueeze(0)
        cur_y = cur_y.unsqueeze(0)
        y_pred = model(cur_x)
        _, y_pred = torch.max(y_pred.data, 1)
        _, cur_y = torch.max(cur_y.data, 1)
        correct += int((y_pred.item() == cur_y.item()))
    return 100*correct/total


if __name__ == "__main__":
    net = Inception_Net(dropout_rate=0, num_classes=num_classes)
    net.load_state_dict(torch.load(PATH))

    x_test_tensor = convert_list_of_tensors_to_tensor(x_data_test)
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(x_test_tensor.shape[0]):
            zam = x_test_tensor[i]
            zam = zam.unsqueeze(0)
            zam1 = y_data_test[i]
            zam1 = zam1.unsqueeze(0)
            outputs = net(zam)
            _, predicted = torch.max(outputs.data, 1)
            _, label = torch.max(zam1, 1)
            total += 1
            correct += int((predicted.item() == label.item()))

    print('Accuracy of the network: %d %%' % (100 * correct / total))

