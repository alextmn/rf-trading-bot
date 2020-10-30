import numpy as np

# import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, CuDNNLSTM, LSTM,Conv1D, MaxPooling1D
from keras.optimizers import Adam

# keras-rl agent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

# trader environment
from enviroment import OhlcvEnv
# custom normalizer
from normalizer import NormalizerProcessor

def create_model(shape, nb_actions):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh",input_shape=shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    #model.add(Dense(nb_actions, activation='sigmoid'))
    model.add(Dense(nb_actions, activation='softmax'))
    return model

def main():
    # OPTIONS
    ENV_NAME = 'OHLCV-v0'
    TIME_STEP = 20

    # Get the environment and extract the number of actions.
    PATH_TRAIN = "./data/train/"
    PATH_TEST = "./data/test/"
    env = OhlcvEnv(TIME_STEP, path=PATH_TRAIN)
    env_test = OhlcvEnv(TIME_STEP, path=PATH_TEST)

    # random seed
    np.random.seed(123)
    env.seed(123)

    nb_actions = env.action_space.n
    model = create_model(shape=env.shape, nb_actions=nb_actions)
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and even the metrics!
    memory = SequentialMemory(limit=1000, window_length=TIME_STEP)
    # policy = BoltzmannQPolicy()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
            attr='eps', value_max=1., value_min=.2, value_test=.05, nb_steps=3000)
    #policy = EpsGreedyQPolicy()
    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=200,
                   enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy,
                   processor=NormalizerProcessor())
    dqn.compile(Adam(lr=1e-3), metrics=['crossentropy'])

    while True:
        # train
        dqn.load_weights('model/duel_dqn_weights-a-2.h5f')
        dqn.fit(env, nb_steps=17511 * 10, nb_max_episode_steps=17511,
          visualize=False, verbose=2)
        dqn.save_weights('model/duel_dqn_weights-a-2.h5f', overwrite=True)
        #try:
            # validate
        info = dqn.test(env_test, nb_episodes=1, visualize=False)
        #env.save_history()
        # n_long, n_short, total_reward, portfolio = info['n_trades']['long'], info['n_trades']['short'], info[
        #     'total_reward'], int(info['portfolio'])
        # np.array([info]).dump(
        #     './info/duel_dqn_{0}_weights_{1}LS_{2}_{3}_{4}.info'.format(ENV_NAME, portfolio, n_long, n_short,
        #                                                             total_reward))
        # dqn.save_weights(
        #     './model/duel_dqn_{0}_weights_{1}LS_{2}_{3}_{4}.h5f'.format(ENV_NAME, portfolio, n_long, n_short, total_reward),
        #     overwrite=True)
        # except KeyboardInterrupt:
        #     continue
        break

if __name__ == '__main__':
    main()