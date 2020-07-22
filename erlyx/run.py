from erlyx.environment import BaseEnvironment
from erlyx.callbacks.base import CallbackTuple
from erlyx.agents import BaseAgent

from tqdm.auto import tqdm


def run(environment: BaseEnvironment, agent: BaseAgent, n_episodes: int, callbacks=None, use_tqdm=True):
    callbacks = CallbackTuple(callbacks or [])
    callbacks.call(event='on_train_begin')
    episode_iterator = range(n_episodes)
    if use_tqdm:
        episode_iterator = tqdm(episode_iterator)
    for _ in episode_iterator:
        episode, observation = environment.new_episode()
        if callbacks.call(event='on_episode_begin', initial_observation=observation):
            break
        done = False
        while not done:
            action = agent.select_action(observation)
            observation, reward, done = episode.step(action)
            if callbacks.call(event='on_step_end', action=action, observation=observation, reward=reward, done=done):
                break
        if callbacks.call(event='on_episode_end'):
            break
    callbacks.call(event='on_train_end')
