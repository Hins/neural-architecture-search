import numpy as np
import csv
from datetime import datetime

import tensorflow as tf
from keras import backend as K
from keras.utils import to_categorical

from controller import Controller, StateSpace
from manager import NetworkManager
from model import model_fn

# create a shared session between Keras and Tensorflow
policy_sess = tf.Session()
K.set_session(policy_sess)

NUM_LAYERS = 2  # number of layers of the state space
MAX_TRIALS = 100  # maximum number of models generated, adjust by xtpan from 250 to 100

MAX_EPOCHS = 1  # maximum number of epochs to train, adjust by xtpan from 10 to 2
CHILD_BATCHSIZE = 512  # batchsize of the child models
EXPLORATION = 0.7  # high exploration for the first 1000 steps
REGULARIZATION = 1e-3  # regularization strength
CONTROLLER_CELLS = 32  # number of cells in RNN controller
EMBEDDING_DIM = 20  # dimension of the embeddings for each state
ACCURACY_BETA = 0.8  # beta value for the moving average of the accuracy
CLIP_REWARDS = 0.0  # clip rewards in the [-0.05, 0.05] range
RESTORE_CONTROLLER = True  # restore controller to continue training
TOP_K_CANDIDATE_ACTION = 5

# construct a state space
state_space = StateSpace()

# add states
state_space.add_state(name='embedding', values=[50, 100, 200])
state_space.add_state(name='bidirection_lstm', values=[64, 128, 256])
state_space.add_state(name='filters', values=[16, 32, 64])
state_space.add_state(name='kernel', values=[1, 3])

# print the state space being searched
state_space.print_state_space()

x_train = []
y_train = []
x_test = []
y_test = []
label_size = 0
with open('nlp/train.dat', 'r') as f:
    for line in f:
        elements = line.strip('\r\n').split('\t')
        x_train.append(elements[0].split(','))
        y_train.append(elements[1])
        if int(elements[1]) > label_size:
            label_size = int(elements[1])
    f.close()
with open('nlp/val.dat', 'r') as f:
    for line in f:
        elements = line.strip('\r\n').split('\t')
        x_test.append(elements[0].split(','))
        y_test.append(elements[1])
        if int(elements[1]) > label_size:
            label_size = int(elements[1])
    f.close()
label_size += 1
print('label size is %d' % label_size)
x_train = np.asarray(x_train, dtype=np.int32)
y_train = np.asarray(y_train, dtype=np.int32)
x_test = np.asarray(x_test, dtype=np.int32)
y_test = np.asarray(y_test, dtype=np.int32)

y_train = np.reshape(y_train, newshape=[y_train.shape[0], 1])
y_train = to_categorical(y_train, num_classes=label_size)
y_test = np.reshape(y_test, newshape=[y_test.shape[0], 1])
y_test = to_categorical(y_test, num_classes=label_size)

dataset = [x_train, y_train, x_test, y_test]  # pack the dataset for the NetworkManager

previous_acc = 0.0
total_reward = 0.0

with policy_sess.as_default():
    # create the Controller and build the internal policy network
    controller = Controller(policy_sess, NUM_LAYERS, state_space,
                            reg_param=REGULARIZATION,
                            exploration=EXPLORATION,
                            controller_cells=CONTROLLER_CELLS,
                            embedding_dim=EMBEDDING_DIM,
                            restore_controller=RESTORE_CONTROLLER)

# create the Network Manager
manager = NetworkManager(dataset, epochs=MAX_EPOCHS, child_batchsize=CHILD_BATCHSIZE, clip_rewards=CLIP_REWARDS,
                         acc_beta=ACCURACY_BETA)

# get an initial random state space if controller needs to predict an
# action from the initial state
state = state_space.get_random_state_space(NUM_LAYERS)
print("Initial Random State : ", state_space.parse_state_space_list(state))
print()

# clear the previous files
controller.remove_files()

best_acc = 0.0
best_state_space = []
# used to dedup action info
action_history_dict = {}

start_time = datetime.now()
# train for number of trails
for trial in range(MAX_TRIALS):
    with policy_sess.as_default():
        K.set_session(policy_sess)
        actions_set = controller.get_action(state, TOP_K_CANDIDATE_ACTION)  # get an action for the previous state

    new_action_flag = False
    for action in actions_set:
        action_str = ','.join([str(item) for item in state_space.parse_state_space_list(action)])
        actions = action
        if action_str not in action_history_dict:
            new_action_flag = True
            break
    if new_action_flag is False:
        print('no new action in %d trial, action_str is %s' % (trial, action_str))
        continue
    action_history_dict[action_str] = 1

    # print the action probabilities
    state_space.print_actions(actions)
    print("Predicted actions : ", state_space.parse_state_space_list(actions))

    # build a model, train and get reward and accuracy from the network manager
    reward, previous_acc = manager.get_rewards(model_fn, state_space.parse_state_space_list(actions))
    print("Rewards : ", reward, "Accuracy : ", previous_acc)
    if previous_acc > best_acc:
        best_acc = previous_acc
        best_state_space = actions

    with policy_sess.as_default():
        K.set_session(policy_sess)

        total_reward += reward
        print("Total reward : ", total_reward)

        # actions and states are equivalent, save the state and reward
        state = actions
        controller.store_rollout(state, reward)

        # train the controller on the saved state and the discounted rewards
        loss = controller.train_step()
        print("Trial %d: Controller loss : %0.6f" % (trial + 1, loss))

        # write the results of this trial into a file
        with open('train_history.csv', mode='a+') as f:
            data = [previous_acc, reward]
            data.extend(state_space.parse_state_space_list(state))
            writer = csv.writer(f)
            writer.writerow(data)
    print()

end_time = datetime.now()
print("Time cost is %d seconds" % (start_time - end_time).seconds)
print("Total Reward : %f, best accuracy is %f" % (total_reward, best_acc))
print("best actions ", state_space.parse_state_space_list(best_state_space))