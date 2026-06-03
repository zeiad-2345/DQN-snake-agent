import time
from collections import deque, namedtuple
import numpy as np
import tensorflow as tf
import utils
from snake_env import SnakeEnv

from keras import Sequential
from keras.layers import Dense, Input
from keras.losses import MeanSquaredError
from keras.optimizers import Adam

tf.random.set_seed(utils.SEED)
np.random.seed(utils.SEED)

MEMORY_SIZE = 100_000
GAMMA = 0.995
ALPHA = 1e-3
NUM_STEPS_FOR_UPDATE = 4

experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


def compute_loss(experiences, gamma, q_network, target_q_network, mse_fn):
    states, actions, rewards, next_states, done_vals = experiences
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    y_targets = rewards + gamma * max_qsa * (1 - done_vals)
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
    return mse_fn(y_targets, q_values)


_EMIT_INTERVAL = 0.05  # cap snake state at ~20 fps over WebSocket


def train(config, stop_event, emit_fn):
    gamma        = float(config.get('gamma', GAMMA))
    alpha        = float(config.get('alpha', ALPHA))
    num_episodes = int(config.get('num_episodes', 2000))

    env    = SnakeEnv()
    mse_fn = MeanSquaredError()

    q_network = Sequential([
        Input(shape=(11,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3, activation='linear'),
    ])
    target_q_network = Sequential([
        Input(shape=(11,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3, activation='linear'),
    ])

    optimizer = Adam(learning_rate=alpha)
    target_q_network.set_weights(q_network.get_weights())

    @tf.function
    def agent_learn(experiences):
        with tf.GradientTape() as tape:
            loss = compute_loss(experiences, gamma, q_network, target_q_network, mse_fn)
        gradients = tape.gradient(loss, q_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
        utils.update_target_network(q_network, target_q_network)

    memory_buffer     = deque(maxlen=MEMORY_SIZE)
    epsilon           = 1.0
    total_point_history = []
    num_p_av          = 100
    best_score        = 0
    start             = time.time()
    last_emit         = 0.0

    for i in range(num_episodes):
        if stop_event.is_set():
            break

        state, _ = env.reset()
        total_points = 0.0

        for t in range(1000):
            if stop_event.is_set():
                break

            state_qn = np.expand_dims(state, axis=0)
            q_values = q_network(state_qn)
            action   = utils.get_action(q_values, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory_buffer.append(experience(state, action, reward, next_state, done))

            now = time.time()
            if now - last_emit >= _EMIT_INTERVAL:
                emit_fn('snake_state', {
                    'snake':     [list(s) for s in env.snake],
                    'food':      list(env.food),
                    'grid_size': env.grid_size,
                    'score':     len(env.snake) - 3,
                    'episode':   i + 1,
                })
                last_emit = now

            if utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer):
                batch = utils.get_experiences(memory_buffer)
                agent_learn(batch)

            state = next_state.copy()
            total_points += reward
            if done:
                break

        ep_score  = len(env.snake) - 3
        best_score = max(best_score, ep_score)

        total_point_history.append(total_points)
        av = float(np.mean(total_point_history[-num_p_av:]))
        epsilon = utils.get_new_eps(epsilon)

        print(f"\rEpisode {i+1} | Avg last {num_p_av}: {av:.2f}", end="")
        if (i + 1) % num_p_av == 0:
            print(f"\rEpisode {i+1} | Avg last {num_p_av}: {av:.2f}")

        emit_fn('episode_update', {
            'episode':        i + 1,
            'total_episodes': num_episodes,
            'reward':         float(total_points),
            'avg_reward':     av,
            'epsilon':        float(epsilon),
            'elapsed':        float(time.time() - start),
            'score':          ep_score,
            'best_score':     best_score,
        })

        if av >= 200.0:
            q_network.save('snake_model.keras')
            emit_fn('training_complete', {'episode': i + 1, 'avg_reward': av})
            break

    emit_fn('training_stopped', {'total_time': float(time.time() - start)})
    env.close()


if __name__ == '__main__':
    import threading
    train(
        config={'gamma': GAMMA, 'alpha': ALPHA, 'num_episodes': 2000},
        stop_event=threading.Event(),
        emit_fn=lambda event, data: None,
    )
