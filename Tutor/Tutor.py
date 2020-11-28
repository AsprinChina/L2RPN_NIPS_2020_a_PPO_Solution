"""
In this file, an expert agent (named Tutor), which does a greedy search
in the reduced action space (208 actions), is built.
It receives an observation, and returns the action that decreases the rho
most, as well as its index [api: Tutor.act(obs)].

author: chen binbin
mail: cbb@cbb1996.com
"""
import os
import time
import numpy as np
from grid2op.Agent import BaseAgent


class Tutor(BaseAgent):
    def __init__(self, action_space, action_space_directory):
        BaseAgent.__init__(self, action_space=action_space)
        self.actions62 = np.load(os.path.join(action_space_directory, 'actions62.npy'))
        self.actions146 = np.load(os.path.join(action_space_directory, 'actions146.npy'))

    @staticmethod
    def reconnect_array(obs):
        new_line_status_array = np.zeros_like(obs.rho)
        disconnected_lines = np.where(obs.line_status==False)[0]
        for line in disconnected_lines[::-1]:
            if not obs.time_before_cooldown_line[line]:
                # this line is disconnected, and, it is not cooling down.
                line_to_reconnect = line
                new_line_status_array[line_to_reconnect] = 1
                break  # reconnect the first one
        return new_line_status_array

    def array2action(self, total_array, reconnect_array):
        action = self.action_space({'change_bus': total_array[236:413]})
        action._change_bus_vect = action._change_bus_vect.astype(bool)
        action.update({'set_line_status': reconnect_array})
        return action

    @staticmethod
    def is_legal(action, obs):
        substation_to_operate = int(action.as_dict()['change_bus_vect']['modif_subs_id'][0])
        if obs.time_before_cooldown_sub[substation_to_operate]:
            # substation is cooling down
            return False
        for line in [eval(key) for key, val in action.as_dict()['change_bus_vect'][str(substation_to_operate)].items() if 'line' in val['type']]:
            if obs.time_before_cooldown_line[line] or not obs.line_status[line]:
                # line is cooling down, or line is disconnected
                return False
        return True

    def act(self, observation):
        tick = time.time()
        reconnect_array = self.reconnect_array(observation)

        if observation.rho.max() < 0.925:
            # secure, return "do nothing" in bus switches.
            return self.action_space({'set_line_status': reconnect_array}), -1

        # not secure, do a greedy search
        min_rho = observation.rho.max()
        print('%s: overload! line-%d has a max. rho of %.2f' % (str(observation.get_time_stamp()), observation.rho.argmax(), observation.rho.max()))
        action_chosen = None
        return_idx = -1
        # hierarchy-1: 62 actions.
        for idx, action_array in enumerate(self.actions62):
            a = self.array2action(action_array, reconnect_array)
            if not self.is_legal(a, observation):
                continue
            obs, _, done, _ = observation.simulate(a)
            if done:
                continue
            if obs.rho.max() < min_rho:
                min_rho = obs.rho.max()
                action_chosen = a
                return_idx = idx
        if min_rho <= 0.999:
            print('    Action %d decreases max. rho to %.2f, search duration is %.2fs' % (return_idx, min_rho, time.time() - tick))
            return action_chosen if action_chosen else self.array2action(np.zeros(494), reconnect_array), return_idx
        # hierarchy-2: 146 actions.
        for idx, action_array in enumerate(self.actions146):
            a = self.array2action(action_array, reconnect_array)
            if not self.is_legal(a, observation):
                continue
            obs, _, done, _ = observation.simulate(a)
            if done:
                continue
            if obs.rho.max() < min_rho:
                min_rho = obs.rho.max()
                action_chosen = a
                return_idx = idx + 62
        print('    Action %d decreases max. rho to %.2f, search duration is %.2fs' % (return_idx, min_rho, time.time() - tick))
        return action_chosen if action_chosen else self.array2action(np.zeros(494), reconnect_array), return_idx
