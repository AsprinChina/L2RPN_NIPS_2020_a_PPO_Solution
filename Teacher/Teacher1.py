"""
In this file, we do the following thing repeatedly:
    1. choose a scenario
    2. sample a time-step every 6 hours
    3. disconnect a line which is under possible attack
    4. search a greedy action to minimize the max rho (~60k possible actions)
    5. save the tuple of (attacked line, observation, action) to a csv file.

author: chen binbin
mail: cbb@cbb1996.com
"""
import os
import time
import random
import grid2op
import numpy as np
import pandas as pd


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


def save_sample(save_path):
    if action == env.action_space({}):
        return None  # not necessary to save a "do nothing" action
    act_or, act_ex, act_gen, act_load = [], [], [], []
    for key, val in action.as_dict()['change_bus_vect'][
        action.as_dict()['change_bus_vect']['modif_subs_id'][0]].items():
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
                    [env.chronics_handler.get_name(), dst_step, line_to_disconnect,
                     env.line_or_to_subid[line_to_disconnect],
                     env.line_ex_to_subid[line_to_disconnect], str(np.where(obs.rho > 1)[0].tolist()),
                     str([i for i in np.around(obs.rho[np.where(obs.rho > 1)], 2)]),
                     action.as_dict()['change_bus_vect']['modif_subs_id'][0], act_or, act_ex, act_gen, act_load,
                     obs.rho.max(), obs.rho.argmax(), obs_.rho.max(), obs_.rho.argmax()]).reshape([1, -1])),
            pd.DataFrame(np.concatenate((obs.to_vect(), obs_.to_vect(), action.to_vect())).reshape([1, -1]))
        ),
        axis=1
    ).to_csv(os.path.join(save_path, 'Experiences1.csv'), index=0, header=0, mode='a')


if __name__ == "__main__":
    # hyper-parameters
    DATA_PATH = '../training_data_track1'  # for demo only, use your own dataset
    SCENARIO_PATH = '../training_data_track1/chronics'
    SAVE_PATH = './'
    LINES2ATTACK = [45, 56, 0, 9, 13, 14, 18, 23, 27, 39]
    NUM_EPISODES = 1000  # each scenario runs 100 times for each attack (or to say, sample 100 points)

    for episode in range(NUM_EPISODES):
        # traverse all attacks
        for line_to_disconnect in LINES2ATTACK:
            try:
                # if lightsim2grid is available, use it.
                from lightsim2grid import LightSimBackend
                backend = LightSimBackend()
                env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, backend=backend)
            except:
                env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH)
            env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
            # traverse all scenarios
            for chronic in range(len(os.listdir(SCENARIO_PATH))):
                env.reset()
                dst_step = episode * 72 + random.randint(0, 72)  # a random sampling every 6 hours
                print('\n\n' + '*' * 50 + '\nScenario[%s]: at step[%d], disconnect line-%d(from bus-%d to bus-%d]' % (
                    env.chronics_handler.get_name(), dst_step, line_to_disconnect,
                    env.line_or_to_subid[line_to_disconnect], env.line_ex_to_subid[line_to_disconnect]))
                # to the destination time-step
                env.fast_forward_chronics(dst_step - 1)
                obs, reward, done, _ = env.step(env.action_space({}))
                if done:
                    break
                # disconnect the targeted line
                new_line_status_array = np.zeros(obs.rho.shape, dtype=np.int)
                new_line_status_array[line_to_disconnect] = -1
                action = env.action_space({"set_line_status": new_line_status_array})
                obs, reward, done, _ = env.step(action)
                if obs.rho.max() < 1:
                    # not necessary to do a dispatch
                    continue
                else:
                    # search a greedy action
                    action = topology_search(env)
                    obs_, reward, done, _ = env.step(action)
                    save_sample(SAVE_PATH)
