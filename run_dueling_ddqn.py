import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense, Lambda
from keras import backend as K
K.set_image_dim_ordering('th')

ENV_NAME = 'SpaceInvaders-v0'  # Environment name
FRAME_WIDTH = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
NUM_EPISODES = 12000  # Number of episodes the agent plays
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
GAMMA = 0.99  # Discount factorg
EXPLORATION_STEPS = 1000000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
# EXPLORATION_STEPS = 5  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = 50000  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 1000000  # Number of replay memory the agent uses for training
BATCH_SIZE = 32  # Mini batch size
TARGET_UPDATE_INTERVAL = 10000  # The frequency with which the target network is updated
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
LEARNING_RATE = 0.0001  # Learning rate used by ADAM
SAVE_INTERVAL = 300000  # The frequency with which the network is saved
SAVE_INTERVAL_UPDATE_STEPS = 10000
NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
LOAD_NETWORK = False
TRAIN = True
FOLDER_TAG = 'Dueling-DDQN-' + ENV_NAME
SAVE_NETWORK_PATH = 'saved_networks/' + FOLDER_TAG
SAVE_SUMMARY_PATH = 'summary/' + FOLDER_TAG
NUM_EPISODES_AT_TEST = 20  # Number of episodes the agent plays at test time

#masking gpus
os.environ["CUDA_VISIBLE_DEVICES"]="0"


class Policy():
    def __init__(self, num_actions, epsilon_start, epsilon_end, exploration_steps):
        self.num_actions = num_actions
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.exploration_steps = exploration_steps
        self.epsilon_step = (epsilon_start-epsilon_end)/(1.0*exploration_steps)

    def get_random_action(self, state):
        return random.randrange(self.num_actions)

    def get_epsilon_greedy_action(self, state, q_values):
        if self.epsilon >= random.random():
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(q_values)
        return action

    def get_epsilon_greedy_action_with_anneal(self, state, q_values):
        action = self.get_epsilon_greedy_action(state, q_values)
        if (self.epsilon-self.epsilon_step) > self.epsilon_end:
            self.epsilon -= self.epsilon_step
        return action

class ReplayMemory():
    def __init__(self, size):
        self.replay_memory_size = size
        self.replay_memory = deque()

    def is_full():
        return len(deque) == self.replay_memory_size

    def append(self, experience):
        self.replay_memory.append(experience)
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.popleft()

    def get_minibatch(self, batch_size):
        minibatch = random.sample(self.replay_memory, batch_size)
        return minibatch

class Agent():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.t = 0
        self.network_updates = 0
        self.policy = Policy(num_actions, INITIAL_EPSILON, FINAL_EPSILON, EXPLORATION_STEPS)
        self.replay_memory = ReplayMemory(NUM_REPLAY_MEMORY)

        # Parameters used for summary
        self.total_reward_unclipped = 0
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # Create q network
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        # Create target network
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op(q_network_weights)

        #mask gpus
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True,
                                        gpu_options=gpu_options))
        self.saver = tf.train.Saver(q_network_weights, max_to_keep=10000)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(SAVE_SUMMARY_PATH, self.sess.graph)

        if not os.path.exists(SAVE_NETWORK_PATH):
            os.makedirs(SAVE_NETWORK_PATH)

        if not os.path.exists(SAVE_NETWORK_PATH + '_evaluation/'):
            os.makedirs(SAVE_NETWORK_PATH + '_evaluation/')            

        self.sess.run(tf.global_variables_initializer())

        # Load network
        if LOAD_NETWORK:
            self.load_network()

        # Initialize target network
        self.sess.run(self.update_target_network)

    def build_network(self):
        model = Sequential()
        model.add(Convolution2D(16, 8, 8, subsample=(4, 4), activation='relu', input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
        model.add(Convolution2D(32, 4, 4, subsample=(2, 2), activation='relu'))
        #model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.num_actions+1))
        model.add(Lambda(lambda a: K.expand_dims(a[:, 0]) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(self.num_actions,)))

        s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
        q_values = model(s)

        return s, q_values, model

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        grads_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grads_update

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
        state = [processed_observation for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=0)

    def run(self, state, action, reward, terminal, observation):
        next_state = np.append(state[1:, :, :], observation, axis=0)
        self.total_reward_unclipped += reward

        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        reward = np.clip(reward, -1, 1)

        # Store transition in replay memory
        experience = (state, action, reward, next_state, terminal)
        self.replay_memory.append(experience)
        
        if self.t >= INITIAL_REPLAY_SIZE:
            # Train network
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()
                self.network_updates += 1

            # Update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % SAVE_INTERVAL == 0:
                save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=self.t)
                print('Successfully saved: ' + save_path)

            if self.network_updates % SAVE_INTERVAL_UPDATE_STEPS == 0:
                save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '_evaluation/' + ENV_NAME, global_step=self.network_updates)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
        self.duration += 1

        if terminal:
            # Write summary
            if self.t >= INITIAL_REPLAY_SIZE:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                        self.duration, self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), self.total_reward_unclipped]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.t < INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode + 1, self.t, self.duration, self.policy.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), mode))

            self.total_reward_unclipped = 0
            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = self.replay_memory.get_minibatch(BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        
        online_q_values_next_state_batch = self.q_values.eval(feed_dict={self.s: np.float32(np.array(next_state_batch) / 255.0)})
        online_q_values_best_next_action_batch = np.argmax(online_q_values_next_state_batch, axis=1)
        target_q_values_next_state_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)})
        
        # Convert action selected by online network to one hot
        next_a_one_hot = tf.one_hot(online_q_values_best_next_action_batch, self.num_actions, 1.0, 0.0)
        q_values_next_batch = tf.reduce_sum(tf.multiply(online_q_values_next_state_batch, next_a_one_hot), reduction_indices=1)

        # print "q_values for next state: " + str(online_q_values_next_state_batch.shape)
        # print "argmax for prev vector:  " + str(online_q_values_best_next_action_batch.shape)
        # print "next a one hot: " + str(next_a_one_hot.shape)
        # print "q_values_next_batch" + str(q_values_next_batch.shape)


        y_batch = reward_batch + (1 - terminal_batch) * GAMMA * q_values_next_batch

        # print "y_batch" + str(q_values_next_batch.shape)

        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: np.float32(np.array(state_batch) / 255.0),
            self.a: action_batch,
            self.y: y_batch.eval()
        })

        self.total_loss += loss

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Loss/Episode', episode_avg_loss)
        episode_total_reward_unclipped = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Total Reward Unclipped/Episode', episode_total_reward_unclipped)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss, episode_total_reward_unclipped]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def get_action_at_test(self, state):
        if random.random() <= 0.05:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))

        self.t += 1

        return action


def preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
    return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))


def main():
    env = gym.make(ENV_NAME)
    agent = Agent(num_actions=env.action_space.n)

    if TRAIN:  # Train mode
        for _ in range(NUM_EPISODES):
            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation

                if agent.t < INITIAL_REPLAY_SIZE:
                    action = agent.policy.get_random_action(state)
                else:
                    action = agent.policy.get_epsilon_greedy_action_with_anneal(state, agent.q_values.eval(feed_dict={agent.s: [np.float32(state / 255.0)]}))

                observation, reward, terminal, _ = env.step(action)
                # env.render()
                processed_observation = preprocess(observation, last_observation)
                state = agent.run(state, action, reward, terminal, processed_observation)
    else:  # Test mode
        # env.monitor.start(ENV_NAME + '-test')
        for _ in range(NUM_EPISODES_AT_TEST):
            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation
                action = agent.get_action_at_test(state)
                observation, _, terminal, _ = env.step(action)
                env.render()
                processed_observation = preprocess(observation, last_observation)
                state = np.append(state[1:, :, :], processed_observation, axis=0)
        # env.monitor.close()


if __name__ == '__main__':
    main()
