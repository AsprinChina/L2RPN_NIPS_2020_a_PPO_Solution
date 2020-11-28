"""
In this file, we do the following thing repeatedly:
    1. choose a scenario
    2. while not game over:
    3.     if not overflow:
    4.         step a "reconnect disconnected line" or "do nothing" action
    5.     else:
    6.         search a greedy action to minimize the max rho (~60k possible actions)
    7.         save the tuple of (None, observation, action) to a csv file.

author: chen binbin
mail: cbb@cbb1996.com
"""
import os
import time
import grid2op
import numpy as np
import pandas as pd
from copy import deepcopy


def topology_search(env):
    obs = env.get_obs()
    min_rho, overflow_id = obs.rho.max(), obs.rho.argmax()
    print("step-%s, line-%s(from bus-%d to bus-%d) overflows, max rho is %.5f" %
          (dst_step, overflow_id, env.line_or_to_subid[overflow_id],
           env.line_ex_to_subid[overflow_id], obs.rho.max()))
    all_actions = env.action_space.get_all_unitary_topologies_change(env.action_space)
    action_chosen = env.action_space({})
    tick = time.time()
    for action in all_actions:
        if not env._game_rules(action, env):
            continue
        obs_, _, done, _ = obs.simulate(action)
        if (not done) and (obs_.rho.max() < min_rho):
            min_rho = obs_.rho.max()
            action_chosen = action
    print("find a greedy action and max rho decreases to %.5f, search duration: %.2f" %
          (min_rho, time.time() - tick))
    return action_chosen


def save_sample(save_path='./'):
    if action == env.action_space({}):
        return None  # not necessary to save a "do nothing" action
    act_or, act_ex, act_gen, act_load = [], [], [], []
    for key, val in action.as_dict()['change_bus_vect'][action.as_dict()['change_bus_vect']['modif_subs_id'][0]].items():
        if val['type'] == 'line (extremity)':
            act_ex.append(key)
        elif val['type'] == 'line (origin)':
            act_or.append(key)
        elif val['type'] == 'load':
            act_load.append(key)
        else:
            act_gen.append(key)
    pd.concat(
        (
            pd.DataFrame(
                np.array(
                    [env.chronics_handler.get_name(), dst_step, None, None, None,
                     str(np.where(obs.rho > 1)[0].tolist()),
                     str([i for i in np.around(obs.rho[np.where(obs.rho > 1)], 2)]),
                     action.as_dict()['change_bus_vect']['modif_subs_id'][0], act_or, act_ex, act_gen, act_load,
                     obs.rho.max(), obs.rho.argmax(), obs_.rho.max(), obs_.rho.argmax()]).reshape([1, -1])),
            pd.DataFrame(np.concatenate((obs.to_vect(), obs_.to_vect(), action.to_vect())).reshape([1, -1]))
        ),
        axis=1
    ).to_csv(os.path.join(save_path, 'Experiences2.csv'), index=0, header=0, mode='a')


def find_best_line_to_reconnect(obs, original_action):
    disconnected_lines = np.where(obs.line_status == False)[0]
    if not len(disconnected_lines):
        return original_action
    o, _, _, _ = obs.simulate(original_action)
    min_rho = o.rho.max()
    line_to_reconnect = -1
    for line in disconnected_lines:
        if not obs.time_before_cooldown_line[line]:
            reconnect_array = np.zeros_like(obs.rho)
            reconnect_array[line] = 1
            reconnect_action = deepcopy(original_action)
            reconnect_action.update({'set_line_status': reconnect_array})
            if not is_legal(reconnect_action, obs):
                continue
            o, _, _, _ = obs.simulate(reconnect_action)
            if o.rho.max() < min_rho:
                line_to_reconnect = line
                min_rho = o.rho.max()
    if line_to_reconnect != -1:
        reconnect_array = np.zeros_like(obs.rho)
        reconnect_array[line_to_reconnect] = 1
        original_action.update({'set_line_status': reconnect_array})
    return original_action


def is_legal(action, obs):
    if 'change_bus_vect' not in action.as_dict():
        return True
    substation_to_operate = int(action.as_dict()['change_bus_vect']['modif_subs_id'][0])
    if obs.time_before_cooldown_sub[substation_to_operate]:
        return False
    for line in [eval(key) for key, val in action.as_dict()['change_bus_vect'][str(substation_to_operate)].items() if 'line' in val['type']]:
        if obs.time_before_cooldown_line[line] or not obs.line_status[line]:
            return False
    return True


if __name__ == "__main__":
    # hyper-parameters
    DATA_PATH = '../training_data_track1'  # for demo only, use your own dataset
    SCENARIO_PATH = '../training_data_track1/chronics'
    SAVE_PATH = './'
    NUM_EPISODE = 100

    for episode in range(NUM_EPISODE):
        try:
            # if lightsim2grid is available, use it.
            from lightsim2grid import LightSimBackend
            backend = LightSimBackend()
            env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, backend=backend)
        except:
            env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH)
        env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
        for chronic in range(len(os.listdir(SCENARIO_PATH))):
            env.reset()
            # dst_step = random.randint(0, 8000)
            dst_step = 0
            print('Scenario to test is [%s]，start from step-%d... ...' % (env.chronics_handler.get_name(), dst_step))
            env.fast_forward_chronics(dst_step)
            obs, done = env.get_obs(), False
            while not done:
                if obs.rho.max() >= 1:
                    action = topology_search(env)
                    obs_, reward, done, _ = env.step(action)
                    save_sample(SAVE_PATH)
                    obs = obs_
                else:
                    action = env.action_space({})
                    action = find_best_line_to_reconnect(obs, action)
                    obs, reward, done, _ = env.step(action)
                dst_step += 1
