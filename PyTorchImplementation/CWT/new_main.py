import argparse

import numpy as np
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import time
from scipy.stats import mode
import STGCN
import STGCN_yester
import Wavelet_CNN_Source_Network
import gcn


def add_args(parser):
    parser.add_argument('--epoch', type=int, default=15, metavar='N',
                        help='number of training')

    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate')

    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='insert batch size for training(default 128)')

    parser.add_argument('--precision', type=float, default=1e-6, metavar='N',
                        help='reducing learning rate when a metric has stopped improving(default = 0.0000001')

    parser.add_argument('--channel',default='[7,8,8]',metavar='N', help=' 3 channel')

    parser.add_argument('--dropout', type=float, default=0.2, metavar='N',
                        help='probability of elements to be zero')

    args = parser.parse_args()

    return args


def scramble(examples, labels):
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)

    new_labels, new_examples = [], []
    for i in random_vec:
        new_labels.append(labels[i])
        new_examples.append(examples[i])

    return new_examples, new_labels


def train_all(args, examples_training, labels_training, examples_test_0, labels_test_0, examples_test_1,
                      labels_test_1):
    accuracy_test0, accuracy_test1 = [], []
    X_fine_tune_train, Y_fine_tune_train = [], []
    X_test_0, X_test_1, Y_test_0, Y_test_1 = [], [], [], []

    for dataset_index in range(0, 17):
        for label_index in range(len(labels_training)):
            if label_index == dataset_index:
                for example_index in range(len(examples_training[label_index])):
                    if (example_index < 28):
                        X_fine_tune_train.extend(examples_training[label_index][example_index])
                        Y_fine_tune_train.extend(labels_training[label_index][example_index])
        print("{}-th data set open~~~".format(dataset_index))
        for label_index in range(len(labels_test_0)):
            if label_index == dataset_index:
                for example_index in range(len(examples_test_0[label_index])):
                    X_test_0.extend(examples_test_0[label_index][example_index])
                    Y_test_0.extend(labels_test_0[label_index][example_index])

        for label_index in range(len(labels_test_1)):
            if label_index == dataset_index:
                for example_index in range(len(examples_test_1[label_index])):
                    X_test_1.extend(examples_test_1[label_index][example_index])
                    Y_test_1.extend(labels_test_1[label_index][example_index])

    X_fine_tunning, Y_fine_tunning = scramble(X_fine_tune_train, Y_fine_tune_train)


    valid_examples = X_fine_tunning[0:int(len(X_fine_tunning) * 0.1)]
    labels_valid = Y_fine_tunning[0:int(len(Y_fine_tunning) * 0.1)]
    X_fine_tune = X_fine_tunning[int(len(X_fine_tunning) * 0.1):]
    Y_fine_tune = Y_fine_tunning[int(len(Y_fine_tunning) * 0.1):]
    print("total data size :", len(X_fine_tune_train), np.shape(np.array(X_fine_tune_train)))
    X_fine_tune = torch.from_numpy(np.array(X_fine_tune[:81200], dtype=np.float32))
    #X_fine_tune = torch.transpose(X_fine_tune, 1, 2)
    X_fine_tune = torch.transpose(X_fine_tune, 1, 3)
    print("train data :", np.shape(np.array(X_fine_tune)))
    Y_fine_tune = torch.from_numpy(np.array(Y_fine_tune[:81200], dtype=np.float32))
    valid_examples = torch.from_numpy(np.array(valid_examples[:9000], dtype=np.float32))
    #valid_examples = torch.transpose(valid_examples, 1, 2)
    valid_examples = torch.transpose(valid_examples, 1, 3)
    print("valid data :", np.shape(np.array(valid_examples)))
    labels_valid = torch.from_numpy(np.array(labels_valid[:9000], dtype=np.float32))
    # dimension setting
    X_test_0 = torch.from_numpy(np.array(X_test_0, dtype=np.float32))
    #X_test_0 = torch.transpose(X_test_0, 1, 2)
    X_test_0 = torch.transpose(X_test_0, 1, 3)
    X_test_1 = torch.from_numpy(np.array(X_test_1[:90250], dtype=np.float32))
    #X_test_1 = torch.transpose(X_test_1, 1, 2)
    X_test_1 = torch.transpose(X_test_1, 1, 3)
    Y_test_0 = torch.from_numpy(np.array(Y_test_0, dtype=np.float32))
    Y_test_1 = torch.from_numpy(np.array(Y_test_1[:90250], dtype=np.float32))
    print(X_test_0.shape, X_test_1.shape)

    # dataset
    train = TensorDataset(X_fine_tune, Y_fine_tune)
    valid = TensorDataset(valid_examples, labels_valid)
    test_0 = TensorDataset(X_test_0, Y_test_0)
    test_1 = TensorDataset(X_test_1, Y_test_1)

    # data loading
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=args.batch_size, shuffle=True)
    test_0_loader = torch.utils.data.DataLoader(test_0, batch_size=args.batch_size, shuffle=False)
    test_1_loader = torch.utils.data.DataLoader(test_1, batch_size=args.batch_size, shuffle=False)

    # model create
    stgcn = STGCN.St_conv_block(8, 1, 1, eval(args.channel), "scope", 0.4, act_func='glu', channel=1, num_class=7)
    stgcn = gcn.MYOGCN(7, 32, 32, 32, 16, 7, 0.2, 7)

    criterion = nn.NLLLoss(size_average=False)
    optimizer = optim.Adam(stgcn.parameters(), lr=args.lr)  # lr=0.0404709 lr=args.lr
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                     verbose=True, eps=args.precision) #학습이 개선되지 않을때 자동으로 학습률을 조절합니다.
    #training
    model, num_epoch = train_model(stgcn, criterion, optimizer, scheduler,\
                                   {"train": train_loader, "val": valid_loader}, args.epoch, args.precision)
    model.eval()

    # test : set_0
    total = 0
    correct_prediction_test_0 = 0
    for k, data_test_0 in enumerate(test_0_loader):
        inputs_test_0, ground_truth_test_0 = data_test_0
        inputs_test_0, ground_truth_test_0 = Variable(inputs_test_0), Variable(ground_truth_test_0)
        concat_input = inputs_test_0
        for i in range(20): #input data 옆으로 20개 복사 concat
            concat_input = torch.cat([concat_input, inputs_test_0])

        outputs_test_0 = model(concat_input)
        _, predicted = torch.max(outputs_test_0.data, 1)
        correct_prediction_test_0 += (mode(predicted.cpu().numpy())[0][0] ==
                                      ground_truth_test_0.data.cpu().numpy()).sum()
        total += ground_truth_test_0.size(0)
    accuracy_test0.append(100 * float(correct_prediction_test_0) / float(total))
    print("ACCURACY TESƒT_0 FINAL : %.3f %%" % (100 * float(correct_prediction_test_0) / float(total)))

    # test : set_1
    total = 0
    correct_prediction_test_1 = 0
    for k, data_test_1 in enumerate(test_1_loader):
        # get the inputs
        inputs_test_1, ground_truth_test_1 = data_test_1
        inputs_test_1, ground_truth_test_1 = Variable(inputs_test_1), Variable(ground_truth_test_1)
        concat_input = inputs_test_1
        for i in range(20):
            concat_input = torch.cat([concat_input, inputs_test_1])

        outputs_test_1 = model(concat_input)
        _, predicted = torch.max(outputs_test_1.data, 1)
        correct_prediction_test_1 += (mode(predicted.cpu().numpy())[0][0] ==
                                      ground_truth_test_1.data.cpu().numpy()).sum()
        total += ground_truth_test_1.size(0)
    accuracy_test1.append(100 * float(correct_prediction_test_1) / float(total))
    print("ACCURACY TEST_1 FINAL : %.3f %%" % (100 * float(correct_prediction_test_1) / float(total)))

    #result
    print("AVERAGE ACCURACY TEST 0 %.3f" % np.array(accuracy_test0).mean())
    print("AVERAGE ACCURACY TEST 1 %.3f" % np.array(accuracy_test1).mean())
    return accuracy_test0, accuracy_test1, num_epoch


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs, precision):
    since = time.time()
    best_loss = float('inf')
    patience = 30
    patience_increase = 3
    hundred = False

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10, "^ㅁ^bbbbb")

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0
            total = 0
            for i, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()

                if phase == 'train':
                    model.train()
                    outputs = model(inputs) # forward
                    _, predictions = torch.max(outputs.data, 1)
                    labels = labels.long()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()
                else:# phase == 'val'
                    model.eval()
                    accumulated_predicted = Variable(torch.zeros(len(inputs), 7))
                    loss_intermediary = 0.
                    total_sub_pass = 0
                    for repeat in range(1):#range(20)
                        outputs = model(inputs)
                        labels = labels.long()
                        loss = criterion(outputs, labels)
                        if loss_intermediary == 0.:
                            loss_intermediary = loss.item()
                        else:
                            loss_intermediary += loss.item()
                        _, prediction_from_this_sub_network = torch.max(outputs.data, 1)
                        accumulated_predicted[range(len(inputs)),
                                              prediction_from_this_sub_network.cpu().numpy().tolist()] += 1
                        total_sub_pass += 1
                    _, predictions = torch.max(accumulated_predicted.data, 1)
                    loss = loss_intermediary / total_sub_pass

                running_loss += loss
                running_corrects += torch.sum(predictions == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            print('{} Loss: {:.8f} Acc: {:.8}'.format(
                phase, epoch_loss, epoch_acc))

            # earlystopping
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), 'best_weights_source_wavelet.pt')
                    patience = patience_increase + epoch
                if epoch_acc == 1:
                    print("stopped because of 100%")
                    hundred = True

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience or hundred:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    model_weights = torch.load('best_weights_source_wavelet.pt')
    model.load_state_dict(model_weights)
    model.eval()
    return model, num_epochs


if __name__ == "__main__":
    # loading...
    examples_training = np.load("../formatted_datasets/evaluation_example.npy", encoding="bytes", allow_pickle=True)
    labels_training = np.load("../formatted_datasets/evaluation_labels.npy", encoding="bytes", allow_pickle=True)
    examples_validation0 = np.load("../formatted_datasets/test0_evaluation_example.npy", encoding="bytes",
                                   allow_pickle=True)
    labels_validation0 = np.load("../formatted_datasets/test0_evaluation_labels.npy", encoding="bytes",
                                 allow_pickle=True)
    examples_validation1 = np.load("../formatted_datasets/test1_evaluation_example.npy", encoding="bytes",
                                   allow_pickle=True)
    labels_validation1 = np.load("../formatted_datasets/test1_evaluation_labels.npy", encoding="bytes",
                                 allow_pickle=True)


    parser = argparse.ArgumentParser()
    args = add_args(parser)

    accuracy_one_by_one = []
    array_training_error = []
    array_validation_error = []

    test_0, test_1 = [], []

    for i in range(3):  # range(20) 20번 돌려서 평균내는 역할.
        accuracy_test_0, accuracy_test_1, num_epochs = train_all(args, examples_training, labels_training,
                                                                         examples_validation0, labels_validation0,
                                                                         examples_validation1, labels_validation1)
        print(accuracy_test_0)

        test_0.append(accuracy_test_0)
        test_1.append(accuracy_test_1)
        print("TEST 0 SO FAR: ", test_0)
        print("TEST 1 SO FAR: ", test_1)
        print("CURRENT AVERAGE : ", (np.mean(test_0) + np.mean(test_1)) / 2.)

    print("ACCURACY FINAL TEST 0: ", test_0)
    print("ACCURACY FINAL TEST 0: ", np.mean(test_0))
    print("ACCURACY FINAL TEST 1: ", test_1)
    print("ACCURACY FINAL TEST 1: ", np.mean(test_1))

    # with open("Pytorch_results_4_cycles.txt", "a") as myfile:
    #     myfile.write("stgcn STFT: \n")
    #     myfile.write("Epochs:")
    #     myfile.write(str(num_epochs) + '\n')
    #     myfile.write("Test 0: \n")
    #     myfile.write(str(np.mean(test_0, axis=0)) + '\n')
    #     myfile.write(str(np.mean(test_0)) + '\n')
    #
    #     myfile.write("Test 1: \n")
    #     myfile.write(str(np.mean(test_1, axis=0)) + '\n')
    #     myfile.write(str(np.mean(test_1)) + '\n')
    #     myfile.write("\n\n\n")

