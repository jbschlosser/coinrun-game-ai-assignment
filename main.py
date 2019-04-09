############################################################

def in_ipynb():
  try:
    result = get_ipython().__class__.__name__
    if 'Shell' in result:
      return True
    else:
      return False
  except:
    return False

IN_PYNB = in_ipynb()

#############################################################

import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image 
# pip install Pillow
# pip install torch
# pip install torchvision

from collections import OrderedDict
from coinrun import setup_utils, make
import coinrun.main_utils as utils
from coinrun.config import Config
if not IN_PYNB:
    from gym.envs.classic_control import rendering
from coinrun import policies, wrappers

import superlogger
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import os
import argparse
import pdb
from tensorboardX import SummaryWriter


###########################################################

# Game seed information
NUM_LEVELS = 1 # repeat the same level over and over
EASY_LEVEL = 1 # Start on a very small map, no enemies
EASY_LEVEL2 = 5 # Very small map, no enemies
MEDIUM_LEVEL = 20 # Medium length, no enemies
MEDIUM_LEVEL2 = 45 # Medium length, no enemies
ONE_MONSTER = 10 # Short map with one monster
HARD_LEVEL = 7 # Longer and with monsters
LAVA_LEVEL = 3 # Longer and with lava and pits

###########################################################
'''
Colab instructions:
New notebook
Edit > Notebook settings > GPU

[1]
import os
del os.environ['LD_PRELOAD']
!apt-get remove libtcmalloc*

[2]
!apt-get update
!apt-get install mpich build-essential qt5-default pkg-config

[3]
import torch
torch.cuda.is_available()

[4]
!git clone https://github.com/markriedl/coinrun-game-ai-assignment.git

[5]
!pip install -r coinrun-game-ai-assignment/requirements.txt

[6]
import sys
sys.path.insert(0, 'coinrun-game-ai-assignment')

[7]
### Testing coinrun with random agent
from coinrun.random_agent import random_agent
random_agent(max_steps=10)

[8]
from main import *
'''



###########################################################
### ARGPARSE

parser = argparse.ArgumentParser(description='Train CoinRun DQN agent.')
parser.add_argument('--render', action="store_true", default=False)
parser.add_argument('--unit_test', action="store_true", default=False)
parser.add_argument('--eval', action="store_true", default=False)
parser.add_argument("--save", help="save the model", default="saved.model")
parser.add_argument("--load", help="load a model", default=None)
parser.add_argument("--episodes", help="number of episodes", type=int, default=1000)
parser.add_argument("--model_path", help="path to saved models", default="saved_models")
parser.add_argument("--seed", help="which level", default=EASY_LEVEL)

args = None
if not IN_PYNB:
    args = parser.parse_args()


writer = SummaryWriter(log_dir=args.model_path)
logger = superlogger.LogWriter(args.model_path)

###########################################################
### CONSTANTS

# if gpu is to be used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', DEVICE)

# Resize the screen to this
RESIZE_CONST = 80

# Defaults
RENDER_SCREEN = args.render if not IN_PYNB else False
SAVE_FILENAME = args.save if not IN_PYNB else 'saved.model'
LOAD_FILENAME = args.load if not IN_PYNB else 'saved.model'
MODEL_PATH = args.model_path if not IN_PYNB else 'saved_models' 
SEED = args.seed if not IN_PYNB else EASY_LEVEL


# Don't play with this
EVAL_EPSILON = 0.1
EVAL_WINDOW_SIZE = 3
EVAL_COUNT = 10
TIMEOUT = 1000
COIN_REWARD = 100

# You may want to change these, but is probably not necessary
BATCH_SIZE = 128            # How many replay experiences to run through neural net at once
GAMMA = 0.99               # How much to discount the future [0..1]
BOOTSTRAP = 5000            # How many steps to run to fill up replay memory before training starts
TARGET_UPDATE = 2           # Delays updating the network for loss calculations. 0=don't delay, or 1+ number of episodes
REPLAY_CAPACITY = 10000     # How big is the replay memory
EPSILON = 1.0               # Use random action if less than epsilon [0..1]
EVAL_INTERVAL = 10          # How many episodes of training before evaluation
RANDOM_SEED = None          # Seed for random number generator, for reproducability, use None for random seed
NUM_EPISODES = args.episodes if not IN_PYNB else 1000   # Max number of training episodes
LOG_INTERVAL = 100

# Linearly decay epsilon from start -> end.
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_PERIOD = 50000
EPSILON_DELTA = (EPSILON_START - EPSILON_END) / EPSILON_DECAY_PERIOD

############################################################
### HELPERS

### Data structure for holding experiences for replay
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

### Function for resizing the screen
resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(),
                    T.Resize(RESIZE_CONST, interpolation=Image.CUBIC),
                    T.ToTensor()])

def printAllTensors():
    import gc
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(obj.type(), obj.size())

### Take the environment and return a tensor containing screen data as a 3D tensor containing (color, height, width) information.
### Optional: the screen may be manipulated, for example, it could be cropped
SIZE = 80
FRAMESTACK = 4
def get_screen(env, screens = [torch.zeros(1, 1, SIZE, SIZE).to(DEVICE) for _ in range(FRAMESTACK)]):
    # Returned screen requested by gym is 512x512x3. Transpose it into torch order (Color, Height, Width).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    ### DO ANY SCREEN MANIPULATIONS NECESSARY (IF ANY)

    ### END SCREEN MANIPULATIONS
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    # Resize, and add a batch dimension (BCHW)
    # screen = torch.from_numpy(screen)
    # screen = resize(screen).unsqueeze(0).to(DEVICE)
    screen = torch.from_numpy(screen).to(DEVICE).unsqueeze(0)
    screen = F.interpolate(screen, size=(SIZE, SIZE))
    gray_or_grey = torch.tensor([0.21, 0.72, 0.07]).to(DEVICE).view(1, 3, 1, 1)
    screen *= gray_or_grey
    screen = torch.sum(screen, dim=1, keepdim=True)
    screens.pop(0)
    screens.append(screen)
    return torch.cat(screens, dim=1)

### Save the model. Extra information can be added to the end of the filename
def save_model(model, filename, extras = None):
    if extras is not None:
        filename = filename + '.' + str(extras)
    print("Saving", filename, "...")
    torch.save(model, os.path.join(MODEL_PATH, filename))
    print("Done saving.")
 
### Load the model. If there are multiple versions with extra information at the
### end of the filename, get the latest.
def load_model(filename, extras = None):
    if extras is not None:
        filename = filename + '.' + str(extras)
    model = None
    candidates = [os.path.join(MODEL_PATH, f) for f in os.listdir(MODEL_PATH) if filename in f]
    if len(candidates) > 0:
        candidates = sorted(candidates, key=lambda f:os.stat(f).st_mtime, reverse=True)
        filename = candidates[0]
        print("Loading", filename, "...")
        model = torch.load(filename)
        print("Done loading.")
    return model

### Give a text description of the outcome of an episode and also a score
### Score is duration, unless the agent died.
def episode_status(duration, reward):
    status = ""
    score = 0
    if duration >= TIMEOUT:
        status = "timeout"
        score = duration
    elif reward < COIN_REWARD:
        status = "died"
        score = TIMEOUT
    else:
        status = "coin"
        score = duration
    return status, score

############################################################
### ReplayMemory

### Store transitions to use to prevent catastrophic forgetting.
### ReplayMemory implements a ring buffer. Items are placed into memory
###    until memory reaches capacity, and then new items start replacing old items
###    at the beginning of the array. 
### Member variables:
###    capacity: (int) number of transitions that can be stored
###    memory: (array) holds transitions (state, action, next_state, reward)
###    position: (int) index of current location in memory to place the next transition.

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        # Hard-code bins for storing samples.
        # Format: position, capacity, list
        NBINS = 3
        portion = self.capacity // NBINS
        caps = [portion for _ in range(NBINS - 1)]
        caps.append(self.capacity - sum(caps))
        self.memory = [(0, caps[i], []) for i in range(NBINS)]

    ### Store a transition in memory.
    ### To implement: put new items at the end of the memory array, unless capacity is reached.
    ###    Combine the arguments into a new Transition object.
    ###    If capacity is reached, start overwriting the beginning of the array.
    ###    Use the position index to keep track of where to put the next item. 
    def push(self, state, action, next_state, reward):
        trans = Transition(state=state,
                           action=action,
                           next_state=next_state,
                           reward=reward)
        # Store transaction in the correct bin.
        bin_idx = 0
        rewval = reward.item()
        if rewval < 2.0: bin_idx = 2
        elif rewval < 10.0: bin_idx = 1
        else: bin_idx = 0
        position, capacity, memory = self.memory[bin_idx]
        if(len(memory) == capacity):
            memory[position] = trans
            position = (position + 1) % capacity
        else:
            memory.append(trans)
        self.memory[bin_idx] = (position, capacity, memory)

    ### Return a batch of transition objects from memory containing batch_size elements.
    def sample(self, batch_size, log_entries = {}):
        nbins = len(self.memory)
        portion = batch_size // nbins
        choose = [portion for _ in range(nbins - 1)]
        choose.append(batch_size - sum(choose))
        samples = []
        sampled = [0 for _ in range(nbins)]
        inmem = [len(mem) for _,_,mem in self.memory]
        while len(samples) < batch_size:
            for bin_idx, num in enumerate(choose):
                _, _, mem = self.memory[bin_idx]
                if len(mem) == 0: continue
                to_pick = min(num, batch_size - len(samples))
                if to_pick == 0: break
                sampled[bin_idx] += to_pick
                picks = np.random.choice(len(mem), size=to_pick, replace=True)
                samples.extend([mem[p] for p in picks])
        for i, (s, im) in enumerate(zip(sampled, inmem)):
            log_entries['bin' + str(i) + '_sampled'] = s
            log_entries['bin' + str(i) + '_total'] = im
        return samples

    ### This allows one to call len() on a ReplayMemory object. E.g. len(replay_memory)
    def __len__(self):
        length = sum([len(mem) for pos, cap, mem in self.memory])
        return length

##########################################################
### DQN

class DQN(nn.Module):

    ### Create all the nodes in the computation graph.
    ### We won't say how to put the nodes together into a computation graph. That is done
    ### automatically when forward() is called.
    def __init__(self, h, w, num_actions):
        super(DQN, self).__init__()
        in_channels = 4 # Framestacking of grayscale frames
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Dropout2d(p=0.2),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(6 * 6 * 64, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_actions)
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        def switch_bn_and_dropout(should_train):
            def _switch_bn_and_dropout(m):
                if type(m) == nn.BatchNorm2d or type(m) == nn.Dropout:
                    if should_train: m.train()
                    else: m.eval()
            return _switch_bn_and_dropout
        switch_to_train = switch_bn_and_dropout(True)
        switch_to_eval = switch_bn_and_dropout(False)
        if x.shape[0] == 1:
            self.trunk.apply(switch_to_train)
            self.classifier.apply(switch_to_train)
        else:
            self.trunk.apply(switch_to_train)
            self.classifier.apply(switch_to_train)
        # Expected x shape: NxCxHxW
        # Required input image size is 84x84.
        x = self.trunk(x)
        q_values = self.classifier(x.view(x.shape[0], -1))
        return q_values

##########################################################
### UNIT TESTING

def testReplayMemory():
    print("Testing ReplayMemory...")
    capacity = 100
    test_replay_memory = ReplayMemory(capacity)
    for i in range(capacity):
        test_replay_memory.push(i, i, i, i)
    assert (len(test_replay_memory) == capacity),"size test failed"
    for i in range(len(test_replay_memory)):
        item = test_replay_memory.memory[i]
        assert (item[0] == i), "item" + str(i) + "not holding the correct value"
    for i in range(capacity//2):
        test_replay_memory.push(capacity+i, capacity+i, capacity+i, capacity+i)
    assert (len(test_replay_memory) == capacity), "size test 2 failed"
    # check items
    for i in range(len(test_replay_memory)):
        item = test_replay_memory.memory[i]
        if i < capacity // 2:
            assert (item[0] == i+capacity), "not holding the correct value after looping (first half)"
        else:
            assert (item[0] == i), "not holding the correct value after looping (second half)"
    print("ReplayMemory test passed.")
    return True


def testMakeBatch():
    print("Testing doMakeBatch...")
    batch_size = 128
    capacity = batch_size * 2
    test_replay_memory = ReplayMemory(capacity)
    state = None
    new_state = None
    action = None
    reward = None
    # Test types and shapes of return values
    for i in range(capacity):
        state = torch.randn(1, 3, 80, 80, device=DEVICE)
        new_state = torch.randn(1, 3, 80, 80, device=DEVICE)
        action = torch.randn(1, 1, device=DEVICE)
        reward = torch.randn(1, 1, device=DEVICE)
        test_replay_memory.push(state, action, new_state, reward)
    states_batch, actions_batch, next_states_batch, rewards_batch, non_final_mask = doMakeBatch(test_replay_memory, batch_size)
    assert(type(states_batch) == torch.Tensor and states_batch.size() == (batch_size, 3, 80, 80)), "states batch not correct shape."
    assert(type(actions_batch) == torch.Tensor and actions_batch.size() == (batch_size, 1)), "actions batch not correct shape."
    assert(type(next_states_batch) == torch.Tensor and next_states_batch.size() == (batch_size, 3, 80, 80)), "next states batch not correct shape."
    assert(type(rewards_batch) == torch.Tensor and rewards_batch.size() == (batch_size, 1)), "rewards batch not correct shape."
    assert(type(non_final_mask) == type(torch.tensor(batch_size, dtype=torch.uint8, device=DEVICE)) and non_final_mask.size()[0] == batch_size), "non-final mask not correct shape."

    # Test mask
    test_replay_memory = ReplayMemory(batch_size)
    for i in range(batch_size):
        state = torch.randn(1, 3, 80, 80)
        new_state = None
        if i % 2 == 0:
            new_state = torch.randn(1, 3, 80, 80, device=DEVICE)
        action = torch.randn(1, 1, device=DEVICE)
        reward = torch.randn(1, 1, device=DEVICE)
        test_replay_memory.push(state, action, new_state, reward)
    states_batch, actions_batch, next_states_batch, rewards_batch, non_final_mask = doMakeBatch(test_replay_memory, batch_size)
    assert(non_final_mask.sum() == batch_size//2), "non_final_mask not masking properly."
    print("doMakeBatch test passed.")
    return True

class UnitTestDQN(nn.Module):
    def __init__(self, h, w, num_actions):
        super(UnitTestDQN, self).__init__()
        self.num_actions = num_actions
    def forward(self, x):
        assert(False), "Network should not be queried when epsilon = 1.0." 
        return None

def testSelectAction():
    print("Testing select_action...")
    from scipy.stats import chisquare
    sample_size = 10000
    num_tests = 100
    pass_rate = 0.9
    screen_height = 40
    screen_width = 40
    epsilon = 1.0
    num_actions = 7
    test_results = {True: 0, False: 0}
    significance_level = 0.02
    net = UnitTestDQN(screen_height, screen_width, num_actions).to(DEVICE)
    state = torch.randn(1, 3, 80, 80, device=DEVICE)
    for j in range(num_tests):
        samples = {}
        for i in range(sample_size):
            action, new_epsilon = select_action(state, net, num_actions, epsilon, steps_done = 0, bootstrap_threshold = 2)
            assert(type(action) == torch.Tensor and action.size() == (1,1)), "Action not correct shape."
            assert(new_epsilon == epsilon), "Epsilon should not change during bootstrapping."
            action = action.item()
            if action not in samples:
                samples[action] = 0
            samples[action] = samples[action] + 1
        expected = [sample_size / num_actions] * num_actions
        statistic, pvalue = chisquare(f_obs=list(samples.values()), f_exp=expected)
        test_results[pvalue >= significance_level] += 1
    assert(test_results[True] > pass_rate * num_tests), "Random sample is not from uniform distribution."    
    print("select_action test passed.")
    return True

def chi_test():
    num_samples = 100000
    samples = {}
    state = torch.randn(1, 3, 80, 80, device=DEVICE)

    for i in range(num_samples):
        action, new_epsilon = select_action(state, net, num_actions, epsilon, steps_done = 0, bootstrap_threshold = 2)
        s = function_to_test()

        if s not in samples:
            samples[s] = 0

        samples[s] += 1

    expected = [num_samples / num_directions] * num_directions

    test_result = chisquare(f_obs=list(samples.values()), f_exp=expected)

    significance_level = 0.02
    return test_result.pvalue >= significance_level



def testPredictQValues():
    print("Testing doPredictQValues...")
    batch_size = 128
    screen_height = 80
    screen_width = 80
    num_actions = 7
    net = DQN(screen_height, screen_width, num_actions).to(DEVICE)
    states_batch = torch.randn(batch_size, 3, 80, 80, device=DEVICE)
    actions_batch = torch.randint(0, 7, (128, 1), device=DEVICE)
    state_action_values = doPredictQValues(net, states_batch, actions_batch)
    assert(type(state_action_values) == torch.Tensor and state_action_values.size() == (128, 1)), "Return value not correct shape."
    print("doPredictQValues test passed.")
    return True

def testPredictNextStateUtilities():
    print("Testing doPredictNextStateUtilities...")
    screen_height = 80
    screen_width = 80
    num_actions = 7
    batch_size = 128
    passed = False
    net = DQN(screen_height, screen_width, num_actions).to(DEVICE)
    # First option to try is that the batch is full sized.
    try:
        next_states_batch = torch.ones(batch_size, 3, 80, 80, device=DEVICE)
        non_final_mask = torch.ones(batch_size, dtype=torch.uint8, device=DEVICE)
        for i in range(batch_size):
            if i % 2 == 1:
                next_states_batch[i].fill_(0)
                non_final_mask[i] = 0
        next_state_values = doPredictNextStateUtilities(net, next_states_batch, non_final_mask, batch_size)
        assert(type(next_state_values) == torch.Tensor and next_state_values.size() == (batch_size, 1)), "Return value not correct shape (attempt 1)."
        for i in range(batch_size):
            if i % 2 == 1:
                assert(next_state_values[i].sum() == 0), "Element " + str(i) + "is not 0.0 when non_final_mask[i] = 0"
        passed = True
    except RuntimeError as e:
        print(e)
        print("Will try alternative test.")
    if not passed:
        # Next option is that batch is not full sized.
        try:
            next_states_batch = torch.ones(batch_size-1, 3, 80, 80, device=DEVICE)
            non_final_mask = torch.ones(batch_size, dtype=torch.uint8, device=DEVICE)
            non_final_mask[0] = 0
            next_state_values = doPredictNextStateUtilities(net, next_states_batch, non_final_mask, batch_size)
            assert(type(next_state_values) == torch.Tensor and next_state_values.size()[0] == batch_size), "Return value not correctd shape (attempt 2)."
            passed = True
        except RuntimeError as e:
            print(e)
            print("No further alternative tests available.")
    if passed:
        print("doPredictNextStateUtilities test passed.")
        return True
    assert(False), "doPredictNextStateUtilities did NOT pass test."

def testComputeExpectedQValues():
    print("Testing doComputeExpectedQValues...")
    batch_size = 128
    gamma = 0.5
    next_state_values = torch.ones(batch_size)
    rewards_batch = torch.ones(batch_size)
    expected_state_action_values = doComputeExpectedQValues(next_state_values, rewards_batch, gamma)
    assert(type(expected_state_action_values) == torch.Tensor and expected_state_action_values.size()[0] == batch_size), "Return value not expected shape."
    for i in range(batch_size):
        assert(expected_state_action_values[i] == 1.5), "Element " + str(i) + " doesn't have the correct value."
    print("doComputeExpectedQValues test passed.")
    return True

def testComputeLoss():
    print("Testing doComputeLoss...")
    batch_size = 128
    state_action_values = torch.randn(batch_size, device=DEVICE)
    expected_state_action_values = torch.randn(batch_size, device=DEVICE)
    loss = doComputeLoss(state_action_values, expected_state_action_values)
    assert(type(loss) == torch.Tensor and len(loss.size()) == 0), "Loss not of expected shape."
    print("doComputeLoss test passed.")
    return True


def unit_test():
    testReplayMemory()
    testMakeBatch()
    testSelectAction()
    testPredictQValues()
    testPredictNextStateUtilities()
    testComputeExpectedQValues()
    testComputeLoss()

##########################################################
### WORKER FUNCTIONS

### Choose and instantiate an optimizer. A default example is given, which you can change.
### Input:
### - parameters: the DQN parameters
### Output:
### - the optimizer object
def initializeOptimizer(parameters):
    optimizer = torch.optim.Adam(parameters, lr=1e-4, weight_decay=1e-4)
    return optimizer

### Select an action to perform. 
### If a random number [0..1] is greater than epsilon, then query the policy_network,
### otherwise use a random action.
### Inputs:
### - state: a tensor of shape 3 x screen_height x screen_width
### - policy_net: a DQN object
### - num_actions: number of actions available
### - epsilon: float [0..1] indicating whether to choose random or use the network
### - steps_done: number of previously executed steps
### - bootstrap_threshold: number of steps that must be executed before training begins
### This function should return:
### - A tensor of shape 1 x 1 that contains the number of the action to execute
### - The new epsilon value to use next time
def select_action(state, policy_net, num_actions, epsilon, steps_done = 0, bootstrap_threshold = 0, log_entries = {}):
    action = None
    new_epsilon = epsilon
    if steps_done > bootstrap_threshold:
        new_epsilon = max(EPSILON_END, epsilon - EPSILON_DELTA)
    q_vals = policy_net(state)
    qs = q_vals[0].detach().cpu().numpy()
    for i, q in enumerate(qs):
        log_entries['q_value' + str(i)] = q
    if torch.rand(size=(1,)).item() < epsilon:
        # Sample a random action uniformly.
        action = torch.randint(0, num_actions, size=(1,))[:,None].long()
        log_entries['choice'] = 'random'
    else:
        action = torch.argmax(q_vals, dim=1)[:,None].long()
        log_entries['choice'] = 'policy'
    log_entries['action'] = action.item()
    return action, new_epsilon

### Ask for a batch of experience replays.
### Inputs:
### - replay_memory: A ReplayMemory object
### - batch_size: size of the batch to return
### Outputs:
### - states_batch: a tensor of shape batch_size x 3 x screen_height x screen_width
### - actions_batch: a tensor of shape batch_size x 1 containing action numbers
### - next_states_batch: a tensor containing screens. 
### - rewards_batch: a tensor of shape batch_size x 1 containing reward values.
### - non_final_mask: a tensor of bytes of length batch_size containing a 0 if the state is terminal or 1 otherwise
def doMakeBatch(replay_memory, batch_size, log_entries={}):
    assert batch_size > 0
    samples = replay_memory.sample(batch_size, log_entries)
    first = samples[0]
    states_batch = torch.empty((batch_size, *first.state.shape[1:]), device=DEVICE)
    actions_batch = torch.empty((batch_size, 1), device=DEVICE).long()
    next_states_batch = torch.empty((batch_size, *first.state.shape[1:]), device=DEVICE)
    rewards_batch = torch.empty((batch_size, 1), device=DEVICE)
    non_final_mask = torch.empty((batch_size,), device=DEVICE).long()
    for i, sample in enumerate(samples):
        states_batch[i] = sample.state
        actions_batch[i] = sample.action
        rewards_batch[i] = sample.reward
        if sample.next_state is None:
            next_states_batch[i] = torch.zeros(sample.state.shape, device=DEVICE)
            non_final_mask[i] = 0
        else:
            next_states_batch[i] = sample.next_state
            non_final_mask[i] = 1
    return states_batch, actions_batch, next_states_batch, rewards_batch, non_final_mask


### Ask the policy_net to predict the Q value for a batch of states and a batch of actions.
### Inputs:
### - policy_net: the DQN
### - states_batch: a tensor of shape batch_size x 3 x screen_height x screen_width containing screens
### - actions_batch: a tensor of shape batch_size x 1 containing action numbers
### Output:
### - A tensor of shape batch_size x 1 containing the Q-value predicted by the DQN in the position indicated by the action
def doPredictQValues(policy_net, states_batch, actions_batch):
    q_values = policy_net(states_batch)
    state_action_values = torch.gather(q_values, 1, actions_batch.long())
    return state_action_values

### Ask the policy_net to predict the utility of a next_state.
### Inputs:
### - policy_net: The DQN
### - next_states_batch: a tensor of shape batch_size x 3 x screen_height x screen_width
### - non_final_mask: a tensor of length batch_size containing 0 for terminal states and 1 for non-terminal states
### - batch_size: the batch size
### Note: Only run non-terminal states through the policy_net
### Output:
### - A tensor of shape batch_size x 1 containing Q-values
def doPredictNextStateUtilities(policy_net, next_states_batch, non_final_mask, batch_size):
    next_state_values = torch.zeros(batch_size, device=DEVICE)
    next_states_masked = next_states_batch[non_final_mask,:,:,:]
    q_values = torch.max(policy_net(next_states_masked), dim=1)[0]
    next_state_values[non_final_mask] = q_values
    next_state_values = next_state_values[:,None]
    return next_state_values.detach()

### Compute the Q-update equation Q(s_t, a_t) = R(s_t+1) + gamma * argmax_a' Q(s_t+1, a')
### Inputs:
### - next_state_values: a tensor of shape batch_size x 1 containing Q values for state s_t+1
### - rewards_batch: a tensor or shape batch_size x 1 containing reward values for state s_t+1
### Output:
### - A tensor of shape batch_size x 1
def doComputeExpectedQValues(next_state_values, rewards_batch, gamma):
    return next_state_values * gamma + rewards_batch

### Compute the loss
### Inputs:
### - state_action_values: a tensor of shape batch_size x 1 containing Q values
### - expected_state_action_values: a tensor of shape batch_size x 1 containing updated Q values
### Output:
### - A tensor scalar value
def doComputeLoss(state_action_values, expected_state_action_values):
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values,
                            reduction='elementwise_mean')
    return loss

### Run backpropagation. Make sure gradients are clipped between -1 and +1.
### Inputs:
### - loss: a tensor scalar
### - parameters: the parameters of the DQN
### There is no output
def doBackprop(loss, parameters):
    loss.backward()
    # Do gradient clipping based on the global norm.
    torch.nn.utils.clip_grad_norm_(parameters, 0.1)


#########################################################
### OPTIMIZE

### Take a DQN and do one forward-backward pass.
### Since this is Q-learning, we will run a forward pass to get Q-values for state-action pairs and then 
### give the true value as the Q-values after the Q-update equation.
def optimize_model(policy_net, target_net, replay_memory, optimizer, batch_size, gamma, writer, log_entries, steps_done):
    if len(replay_memory) < batch_size:
        return
    ### step 1: sample from the replay memory. Get BATCH_SIZE transitions
    ### Step 2: Get a list of non-final next states.
    ###         a. Create a mask, a tensor of length BATCH_SIZE where each element i is 1 if 
    ###            batch.next_state[i] is not None and 0 otherwise.
    ###         b. Create a tensor of shape [BATCH_SIZE, color(3), height, width] by concatenating
    ###            all non-final (not None) batch.next_states together.
    ### Step 3: set up batches for state, action, and reward
    ###         a. Create a tensor of shape [BATCH_SIZE, color(3), height, width] holding states
    ###         b. Create a tensor of shape [BATCH_SIZE, 1] holding actions
    ###         c. Create a tensor of shape [BATCH_SIZE, 1] holding rewards
    states_batch, actions_batch, next_states_batch, rewards_batch, non_final_mask = doMakeBatch(replay_memory, batch_size, log_entries)

    ### Step 4: Get the action values predicted.
    ###         a. Call policy_net(state_batch) to get a tensor of shape [BATCH_SIZE, NUM_ACTIONS] containing Q-values
    ###         b. For each batch, get the Q-value for the corresponding action in action_batch (hint: torch.gather)
    state_action_values = doPredictQValues(policy_net, states_batch, actions_batch)

    ### Step 5: Get the utility values of next_states.
    next_state_values = doPredictNextStateUtilities(target_net, next_states_batch, non_final_mask, batch_size)
    
    ### Step 6: Compute the expected Q values.
    expected_state_action_values = doComputeExpectedQValues(next_state_values, rewards_batch, gamma)

    ### Step 7: Computer Huber loss (smooth L1 loss)
    ###         Compare state action values from step 5 to expected state action values from step 7
    loss = doComputeLoss(state_action_values, expected_state_action_values)
    if steps_done % LOG_INTERVAL == 0:
        writer.add_scalar('train/loss', loss.item(), steps_done)
    log_entries['loss'] = loss.item()
    ### Step 8: Back propagation
    ###         a. Zero out gradients
    ###         b. call loss.backward()
    ###         c. Prevent gradient explosion by clipping gradients between -1 and 1
    ###            (hint: param.grad.data is the gradients. See torch.clamp_() )
    ###         d. Tell the optimizer that another step has occurred: optimizer.step()
    if optimizer is not None:
        optimizer.zero_grad()
        doBackprop(loss, policy_net.parameters())
        optimizer.step()


##########################################################
### MAIN

### Training loop.
### Each episode is a game that runs until the agent gets the coin or the game times out.
### Train for a given number of episodes.
def train(num_episodes = NUM_EPISODES, load_filename = None, save_filename = None, eval_interval = EVAL_INTERVAL, replay_capacity = REPLAY_CAPACITY, bootstrap_threshold = BOOTSTRAP, epsilon = EPSILON, eval_epsilon = EVAL_EPSILON, gamma = GAMMA, batch_size = BATCH_SIZE, target_update = TARGET_UPDATE, random_seed = RANDOM_SEED, num_levels = NUM_LEVELS, seed = SEED):
    # Set the random seed
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)
    # Set up the environment
    setup_utils.setup_and_load(use_cmd_line_args=False, is_high_res=True, num_levels=num_levels, set_seed=seed)
    env = make('standard', num_envs=1)
    if RENDER_SCREEN and not IN_PYNB:
        env.render()

    # Reset the environment
    env.reset()

    # Get screen size so that we can initialize layers correctly based on shape returned from AI gym. 
    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape
    print("screen size: ", screen_height, screen_width)

    # Are we resuming from an existing model?
    policy_net = None
    if load_filename is not None and os.path.isfile(os.path.join(MODEL_PATH, load_filename)):
        print("Loading model...")
        policy_net = load_model(load_filename)
        policy_net = policy_net.to(DEVICE)
        print("Done loading.")
    else:
        print("Making new model.")
        policy_net = DQN(screen_height, screen_width, env.NUM_ACTIONS).to(DEVICE)
    # Make a copy of the policy network for evaluation purposes
    eval_net = DQN(screen_height, screen_width, env.NUM_ACTIONS).to(DEVICE)
    eval_net.load_state_dict(policy_net.state_dict())
    eval_net.eval()
    # Target network is a snapshot of the policy network that lags behind (for stablity)
    target_net = DQN(screen_height, screen_width, env.NUM_ACTIONS).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Instantiate the optimizer
    optimizer = None
    if len(list(policy_net.parameters())) > 0:
        optimizer = initializeOptimizer(policy_net.parameters())
    
    # Instantiate the replay memory
    replay_memory = ReplayMemory(replay_capacity)

    steps_done = 0               # How many steps have been run
    best_eval = float('inf')     # The best model evaluation to date

    ### Do training until episodes complete 
    print("training...")
    i_episode = 0            # The episode number
    
    # Stop when we reach max episodes
    while i_episode < num_episodes:
        print("episode:", i_episode, "epsilon:", epsilon)
        max_reward = 0       # The best reward we've seen this episode
        done = False         # Has the game ended (timed out or got the coin)
        episode_steps = 0    # Number of steps performed in this episode
        # Initialize the environment and state
        env.reset()
        
        # Current screen. There is no last screen because we get velocity on the screen itself.
        state = get_screen(env)

        # Do forever until the loop breaks
        while not done:
            log_entries = OrderedDict()
            log_entries['epsilon'] = epsilon
            log_entries['episode'] = i_episode
            for i in range(state.shape[1]):
                log_entries['screen' + str(i)] = state[0,i,:,:].detach().cpu().numpy()
            # Select and perform an action
            action, epsilon = select_action(state, policy_net, env.NUM_ACTIONS, epsilon, steps_done, bootstrap_threshold, log_entries)
            log_entries['new_epsilon'] = epsilon
            steps_done = steps_done + 1
            episode_steps = episode_steps + 1
            if steps_done % LOG_INTERVAL == 0:
                writer.add_scalar('train/epsilon', epsilon, steps_done)
                writer.add_scalar('train/episodes', i_episode, steps_done)
                writer.add_image('train/screen', state[0,:,:,:], steps_done)
            
            # for debugging
            if RENDER_SCREEN and not IN_PYNB:
                env.render() 

            # Run the action in the environment
            if action is not None: 
                _, reward, done, _ = env.step(np.array([action.item()]))

                # Record if this was the best reward we've seen so far
                max_reward = max(reward, max_reward)
                log_entries['reward'] = reward[0]
                log_entries['max_reward'] = max_reward[0]
                
                # Turn the reward into a tensor  
                reward = torch.tensor([reward], device=DEVICE)

                # Observe new state
                current_screen = get_screen(env)

                # Did the game end?
                if not done:
                    next_state = current_screen
                else:
                    next_state = None

                # Store the transition in memory
                replay_memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # If we are past bootstrapping we should perform one step of the optimization
                log_entries['loss'] = np.nan
                for i in range(3):
                    log_entries['bin' + str(i) + '_sampled'] = np.nan
                    log_entries['bin' + str(i) + '_total'] = np.nan
                if steps_done > bootstrap_threshold:
                    optimize_model(policy_net, target_net if target_update > 0 else policy_net, replay_memory, optimizer, batch_size, gamma, writer, log_entries, steps_done)
                logger.log(log_entries)
            else:
                # Do nothing if select_action() is not implemented and returning None
                env.step(np.array([0]))
                
            # If we are done, print some statistics
            if done:
                print("duration:", episode_steps)
                print("max reward:", max_reward)
                status, _ = episode_status(episode_steps, max_reward)
                print("result:", status)
                print("total steps:", steps_done, '\n')
                writer.add_scalar('train/duration', episode_steps, steps_done)
                writer.add_scalar('train/max_reward', max_reward, steps_done)

        # Should we update the target network?
        if target_update > 0 and i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
                
        # Should we evaluate?
        if steps_done > bootstrap_threshold and i_episode > 0 and i_episode % eval_interval == 0:
            test_average_duration = 0       # Track the average eval duration
            test_average_max_reward = 0     # Track the average max reward
            # copy all the weights into the evaluation network
            eval_net.load_state_dict(policy_net.state_dict())
            # Evaluate 10 times
            for _ in range(EVAL_COUNT):
                # Call the evaluation function
                test_duration, test_max_reward = evaluate(policy_net, eval_epsilon, env)
                status, score = episode_status(test_duration, test_max_reward)
                test_duration = score # Set test_duration to score to factor in death-penalty
                test_average_duration = test_average_duration + test_duration
                test_average_max_reward = test_average_max_reward + test_max_reward
            test_average_duration = test_average_duration / EVAL_COUNT
            test_average_max_reward = test_average_max_reward / EVAL_COUNT
            print("Average duration:", test_average_duration)
            print("Average max reward:", test_average_max_reward)
            writer.add_scalar('eval/duration', test_average_duration, steps_done)
            writer.add_scalar('eval/max_reward', test_average_max_reward, steps_done)
            # If this is the best window average we've seen, save the model
            if test_average_duration < best_eval:
                best_eval = test_average_duration
                if save_filename is not None:
                    save_model(policy_net, save_filename, i_episode)
            print(' ')
        # Only increment episode number if we are done with bootstrapping
        if steps_done > bootstrap_threshold:
          i_episode = i_episode + 1
    print('Training complete')
    if RENDER_SCREEN and not IN_PYNB:
        env.render()
    writer.close()
    env.close()
    return policy_net
 
 
 

### Evaluate the DQN
### If environment is given, use that. Otherwise make a new environment.
def evaluate(policy_net, epsilon = EVAL_EPSILON, env = None):
    setup_utils.setup_and_load(use_cmd_line_args=False, is_high_res=True, num_levels=NUM_LEVELS, set_seed=SEED)
    
    # Make an environment if we don't already have one
    if env is None:
        env = make('standard', num_envs=1)
    if RENDER_SCREEN and not IN_PYNB:
        env.render()

    # Reset the environment
    env.reset()

    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. 
    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape

    # Get the network ready for evaluation (turns off some things like dropout if used)
    policy_net.eval()

    # Current screen. There is no last screen
    state = get_screen(env)

    steps_done = 0         # Number of steps executed
    max_reward = 0         # Max reward seen
    done = False           # Is the game over?
    logger = superlogger.LogWriter(args.model_path + '/eval')

    print("Evaluating...")
    while not done:
        # Select and perform an action
        log_entries = OrderedDict()
        log_entries['epsilon'] = epsilon
        for i in range(state.shape[1]):
            log_entries['screen' + str(i)] = state[0,i,:,:].detach().cpu().numpy()
        action, _ = select_action(state, policy_net, env.NUM_ACTIONS, epsilon, steps_done=0, bootstrap_threshold=0, log_entries=log_entries)
        steps_done = steps_done + 1
        log_entries['steps_done'] = steps_done

        if RENDER_SCREEN and not IN_PYNB:
            env.render()

        # Execute the action
        if action is not None:
            _, reward, done, _ = env.step(np.array([action.item()]))

            # Is this the best reward we've seen?
            max_reward = max(reward, max_reward)
            log_entries['reward'] = reward[0]
            log_entries['max_reward'] = max_reward[0]

            # Observe new state
            state = get_screen(env)
        else:
            # Do nothing if select_action() is not implemented and returning None
            env.step(np.array([0]))
        logger.log(log_entries)

    print("duration:", steps_done)
    print("max reward:", max_reward)
    status, _ = episode_status(steps_done, max_reward)
    print("result:", status, '\n')
    if RENDER_SCREEN and not IN_PYNB:
        env.render()
    return steps_done, max_reward



if __name__== "__main__":
    if not IN_PYNB:
        if args.unit_test:
            unit_test()
        elif args.eval:
            if args.load is not None and os.path.isfile(os.path.join(MODEL_PATH, args.load)):
                eval_net = load_model(args.load) 
                print(eval_net)
                durations = []
                for _ in range(EVAL_COUNT):
                    duration, _ = evaluate(eval_net, EVAL_EPSILON)
                    durations.append(duration)
                print('Average:', np.mean(durations))
        else:
            policy_net = train(save_filename=SAVE_FILENAME, load_filename=LOAD_FILENAME)
            for _ in range(EVAL_COUNT):
                evaluate(policy_net, EVAL_EPSILON)
