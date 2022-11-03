import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from collections import deque
import pandas as pd


class MemoryReplayBuffer:
    def __init__(self, maxlen, batch_size):
        self.buffer = deque(maxlen=maxlen)
        self.batch_size = batch_size
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self):
        indices = np.arange(len(self.buffer))
        sample_indices = np.random.choice(indices, size=self.batch_size)
        samples = np.array(self.buffer, dtype = object)[sample_indices]
    
        return map(np.array, zip(*samples))


class PrioritizedReplayBuffer:
    def __init__(self, maxlen, batch_size):
        self.priority_scale = 0.8
        self.beta = 0.4 # initial beta
        self.beta_increment_per_sampling = 1e-3
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen) 
        self.batch_size = batch_size
    
    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1)) #new experience has higher prob
        
    def get_probabilities(self):
        scaled_priorities = np.array(self.priorities)**self.priority_scale
        probs = scaled_priorities/sum(scaled_priorities)
        return probs
    
    def get_importance(self, probabilities):
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])  # max = 1
        importance = (1/len(self.buffer) * 1/probabilities)**self.beta
        importance_normalized = importance / max(importance)
        return importance_normalized
    
    def sample(self):
        sample_probs = self.get_probabilities()
        indices = np.arange(len(self.buffer))
        sample_indices = random.choices(indices, k = self.batch_size, weights=sample_probs)
        samples = np.array(self.buffer, dtype = object)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        
        return map(np.array, zip(*samples)), importance, indices
    
    def set_priorities(self, indices, errors, offset=0.1):
        for i,e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset
        

class DQN_agent:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_network = self.build_q_network()
        self.t_q_network = self.build_q_network()
        self.batch_size = 32
        self.buffer = MemoryReplayBuffer(1000, self.batch_size)
        self.optimizer = keras.optimizers.Adam(learning_rate = 3e-4, clipnorm=1.0)
        # timestep in an episode
        self.frame_count = 0
        # prob for exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        # for epsilon decay
        self.epsilon_greedy_frames = 50000.0
        # discounted ratio
        self.gamma = 0.99
        self.loss_function = keras.losses.Huber()
    
    def build_q_network(self):
        # Network architecture
        inputs = keras.Input(shape = self.n_states)
        x = layers.Conv2D(32, 3, strides = 1, padding = "same", activation = 'relu')(inputs)
        x = layers.Conv2D(64, 3, strides = 1, padding = "same", activation = 'relu')(x)

        #x = layers.Conv2D(64, 3, strides = 1, padding = "same", activation = 'relu')(x)
        #x = layers.Conv2D(64, 3, strides = 1, padding = "same", activation = 'relu')(x)
        x = layers.Flatten()(x)

        x = layers.Dense(units = 128, activation = 'relu')(x)
        q_value = layers.Dense(units = self.n_actions)(x)

        return keras.Model(inputs = inputs, outputs = q_value)
    
    def choose_action(self, state):
        # exploration and exploitation
        if self.epsilon >= 0.90:
            action = np.random.choice(self.n_actions)
        else:
            if  self.epsilon >= np.random.rand(1)[0]:
                action = np.random.choice(self.n_actions)
            else:
                action_values = self.q_network(np.expand_dims(state, axis=0))
                action = tf.argmax(action_values[0]).numpy()

        return action

    def decay_epsilon(self):
        # decay probability of taking random action
        self.epsilon -= (1.0 - self.epsilon_min)/self.epsilon_greedy_frames
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def store(self, state, action, next_state, reward, done):
        # store training data
        self.buffer.add((state, action, reward, next_state, done))
    

    def train_q_network(self):
        # sample
        (states, actions, rewards, next_states, dones) = self.buffer.sample()

        next_values = self.q_network.predict(next_states)
        next_action = tf.math.argmax(next_values, 1)
        mask_next_action = tf.one_hot(next_action, self.n_actions)
        future_q_value = self.t_q_network.predict(next_states)
        future_rewards = tf.reduce_sum(tf.multiply(future_q_value, mask_next_action), axis=1)

        # set last q value to 0
        future_rewards = future_rewards*(1 - dones)
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards + self.gamma * future_rewards

        masks = tf.one_hot(actions, self.n_actions)

        with tf.GradientTape() as tape:
          # Train the model on the states and updated Q-values
          q_values = self.q_network(states)
          # only update q-value which is chosen
          q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
          # calculate loss between new Q-value and old Q-value
          loss = self.loss_function(updated_q_values, q_action)
        
        # Backpropagation
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

    def update_target_network(self):
        # update per update_target_network steps
        self.t_q_network.set_weights(self.q_network.get_weights())


class DQN_agent_PER:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_network = self.build_q_network()
        self.t_q_network = self.build_q_network()
        self.batch_size = 32
        self.buffer = PrioritizedReplayBuffer(1000, self.batch_size)
        self.optimizer = keras.optimizers.Adam(learning_rate = 3e-4, clipnorm=1.0)
        # timestep in an episode
        self.frame_count = 0
        # prob for exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        # for epsilon decay
        self.epsilon_greedy_frames = 50000.0
        # discounted ratio
        self.gamma = 0.99
    
    def build_q_network(self):
        # Network architecture
        inputs = keras.Input(shape = self.n_states)
        x = layers.Conv2D(32, 3, strides = 1, padding = "same", activation = 'relu')(inputs)
        x = layers.Conv2D(64, 3, strides = 1, padding = "same", activation = 'relu')(x)

        #x = layers.Conv2D(64, 3, strides = 1, padding = "same", activation = 'relu')(x)
        #x = layers.Conv2D(64, 3, strides = 1, padding = "same", activation = 'relu')(x)
        x = layers.Flatten()(x)

        x = layers.Dense(units = 128, activation = 'relu')(x)
        q_value = layers.Dense(units = self.n_actions)(x)

        return keras.Model(inputs = inputs, outputs = q_value)
    
    def choose_action(self, state):
        # exploration and exploitation
        if self.epsilon >= 0.9:
            action = np.random.choice(self.n_actions)
        else:
            if  self.epsilon >= np.random.rand(1)[0]:
                action = np.random.choice(self.n_actions)
            else:
                action_values = self.q_network(np.expand_dims(state, axis=0))
                action = tf.argmax(action_values[0]).numpy()

        return action

    def decay_epsilon(self):
        # decay probability of taking random action
        self.epsilon -= (1.0 - self.epsilon_min)/self.epsilon_greedy_frames
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def store(self, state, action, next_state, reward, done):
        # store training data
        self.buffer.add((state, action, reward, next_state, done))
    

    def train_q_network(self):
        # sample
        (states, actions, rewards, next_states, dones), importance, indices = self.buffer.sample()

        next_values = self.q_network.predict(next_states)
        next_action = tf.math.argmax(next_values, 1)
        mask_next_action = tf.one_hot(next_action, self.n_actions)
        future_q_value = self.t_q_network.predict(next_states)
        future_rewards = tf.reduce_sum(tf.multiply(future_q_value, mask_next_action), axis=1)

        # set last q value to 0
        future_rewards = future_rewards*(1 - dones)
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards + self.gamma * future_rewards

        masks = tf.one_hot(actions, self.n_actions)

        with tf.GradientTape() as tape:
          # Train the model on the states and updated Q-values
          q_values = self.q_network(states)
          # only update q-value which is chosen
          q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
          # calculate loss between new Q-value and old Q-value
          loss = tf.reduce_mean(importance * tf.math.square(q_action - updated_q_values))
        
        # set priorities
        errors = updated_q_values - q_action
        self.buffer.set_priorities(indices, errors)
        
        # Backpropagation
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

    def update_target_network(self):
        # update per update_target_network steps
        self.t_q_network.set_weights(self.q_network.get_weights())