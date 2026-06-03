import numpy as np
import tensorflow as tf
import random

SEED = 42
BATCH_SIZE = 64
TAU = 1e-3

def get_action(q_values, epsilon):
    if random.random() > epsilon:
        return np.argmax(q_values.numpy()[0])
    return random.choice(np.arange(q_values.shape[1]))

def check_update_conditions(t, num_steps_for_update, memory_buffer):
    return (t + 1) % num_steps_for_update == 0 and len(memory_buffer) >= BATCH_SIZE

def get_experiences(memory_buffer):
    batch = random.sample(memory_buffer, k=BATCH_SIZE)
    states      = tf.cast(np.array([e.state      for e in batch]), tf.float32)
    actions     = tf.cast(np.array([e.action     for e in batch]), tf.float32)
    rewards     = tf.cast(np.array([e.reward     for e in batch]), tf.float32)
    next_states = tf.cast(np.array([e.next_state for e in batch]), tf.float32)
    done_vals   = tf.cast(np.array([e.done       for e in batch], dtype=np.uint8), tf.float32)
    return (states, actions, rewards, next_states, done_vals)

def update_target_network(q_network, target_q_network):
    for target_w, q_w in zip(target_q_network.weights, q_network.weights):
        target_w.assign(TAU * q_w + (1.0 - TAU) * target_w)

def get_new_eps(epsilon):
    return max(epsilon * 0.995, 0.01)
