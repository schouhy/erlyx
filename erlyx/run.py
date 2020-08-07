from erlyx.environment import BaseEnvironment
from erlyx.callbacks.base import CallbackTuple
from erlyx.agents import BaseAgent

from tqdm.auto import tqdm


def run_episodes(environment: BaseEnvironment, agent: BaseAgent, n_episodes: int, callbacks=None, use_tqdm=True):
    callbacks = CallbackTuple(callbacks or [])
    callbacks.call(event='on_train_begin')
    episode_iterator = range(n_episodes)
    if use_tqdm:
        episode_iterator = tqdm(episode_iterator)
    for _ in episode_iterator:
        episode, observation = environment.new_episode()
        if callbacks.call(event='on_episode_begin', initial_observation=observation): break
        done = False
        while not done:
            if callbacks.call(event='on_step_begin'): break
            action = agent.select_action(observation)
            observation, reward, done = episode.step(action)
            if callbacks.call(event='on_step_end', action=action, observation=observation, reward=reward,
                              done=done): break
        if callbacks.call(event='on_episode_end'): break
    callbacks.call(event='on_train_end')


def run_steps(environment: BaseEnvironment, agent: BaseAgent, n_steps: int, callbacks=None, use_tqdm=True):
    callbacks = CallbackTuple(callbacks or [])
    callbacks.call(event='on_train_begin')
    if use_tqdm:
        progress_bar = tqdm(total=n_steps)
    episode_counter = 0
    while episode_counter < n_steps:
        episode, observation = environment.new_episode()
        if callbacks.call(event='on_episode_begin', initial_observation=observation): break
        done = False
        while not done:
            if callbacks.call(event='on_step_begin'): break
            action = agent.select_action(observation)
            observation, reward, done = episode.step(action)
            if callbacks.call(event='on_step_end', action=action, observation=observation, reward=reward, done=done): break
            episode_counter += 1
            if use_tqdm:
                progress_bar.update(1)
        if callbacks.call(event='on_episode_end'): break
    callbacks.call(event='on_train_end')
