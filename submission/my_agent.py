import os
import datetime
import numpy as np
import tensorflow as tf
from copy import deepcopy
from grid2op.Agent import BaseAgent


class MyAgent(BaseAgent):
    def __init__(self, action_space, this_directory_path='./'):
        """Initialize a new agent."""
        BaseAgent.__init__(self, action_space=action_space)
        self.actions62 = np.load(os.path.join(this_directory_path, 'actions62.npy'))
        self.actions146 = np.load(os.path.join(this_directory_path, 'actions146.npy'))
        self.actions = np.concatenate((self.actions62, self.actions146), axis=0)
        self.actions1255 = np.load(os.path.join(this_directory_path, 'actions1255.npy'))
        chosen = list(range(2, 7)) + list(range(7, 73)) + list(range(73, 184)) + list(range(184, 656))
        chosen += list(range(656, 715)) + list(range(715, 774)) + list(range(774, 833)) + list(range(833, 1010))
        chosen += list(range(1010, 1069)) + list(range(1069, 1105)) + list(range(1105, 1164)) + list(range(1164, 1223))
        self.chosen = chosen
        self.ppo = tf.keras.models.load_model(os.path.join(this_directory_path, './ppo-ckpt'))
        self.last_step = datetime.datetime.now()
        self.recovery_stack = []
        self.overflow_steps = 0

    def find_best_line_to_reconnect(self, obs, original_action):
        disconnected_lines = np.where(obs.line_status == False)[0]
        if not len(disconnected_lines):
            return original_action
        if (obs.time_before_cooldown_line[disconnected_lines] > 0).all():
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
                if not self.is_legal(reconnect_action, obs):
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

    def array2action(self, array):
        action = self.action_space({'change_bus': array[236:413]})
        action._change_bus_vect = action._change_bus_vect.astype(bool)
        return action

    def is_legal(self, action, obs):
        adict = action.as_dict()
        if 'change_bus_vect' not in adict:
            return True
        substation_to_operate = int(adict['change_bus_vect']['modif_subs_id'][0])
        if obs.time_before_cooldown_sub[substation_to_operate]:
            return False
        for line in [eval(key) for key, val in adict['change_bus_vect'][str(substation_to_operate)].items()
                     if 'line' in val['type']]:
            if obs.time_before_cooldown_line[line] or not obs.line_status[line]:
                return False
        return True

    def act(self, observation, reward, done):
        tnow = observation.get_time_stamp()
        if self.last_step + datetime.timedelta(minutes=5) != tnow:
            print('\n\nscenario changes！')
            self.recovery_stack = []
        self.last_step = tnow

        if observation.rho.max() >= 1:
            self.overflow_steps += 1
        else:
            self.overflow_steps = 0

        # case: secure
        threshold_this_step = 0.999
        if observation.rho.max() < threshold_this_step:  # fixed threshold
            if (self.recovery_stack == []) or (not self.is_legal(self.array2action(self.actions1255[self.recovery_stack[0]]), observation)):
                a = self.action_space({})
            else:
                o, r, d, i = observation.simulate(self.array2action(self.actions1255[self.recovery_stack[0]]))
                if (not d) and (o.rho.max() < 0.98):
                    aid = self.recovery_stack.pop(0)
                    a = self.array2action(self.actions1255[aid])
                else:
                    a = self.action_space({})
            return self.find_best_line_to_reconnect(observation, a)

        # case: dangerous
        o, _, d, _ = observation.simulate(self.action_space({}))
        min_rho = o.rho.max() if not d else 10
        print('%s, heavy load, line-%d load is %.2f' % (str(observation.get_time_stamp()), observation.rho.argmax(), observation.rho.max()))

        action_chosen = None
        idx_chosen = None
        features = np.concatenate(([0], observation.to_vect()))[None, self.chosen]
        _, a_pred, _ = self.ppo.predict_step(features)
        a_pred = a_pred._numpy()
        sorted_actions = a_pred[0, :].argsort()[::-1]
        for k, idx in enumerate(sorted_actions):
            if idx in [3, 39]:
                continue
            a = self.array2action(self.actions[idx, :])
            if not self.is_legal(a, observation):
                continue
            obs, _, done, _ = observation.simulate(a)
            if done:
                continue
            if obs.rho.max() <= 0.95:
                print('take action %d, max-rho to %.2f, simulation times: %d' % (idx, obs.rho.max(), k + 1))
                return self.find_best_line_to_reconnect(observation, a)
            if obs.rho.max() < min_rho:
                min_rho = obs.rho.max()
                action_chosen = a
                idx_chosen = idx

        if (min_rho < 0.98) or (self.overflow_steps < 2):
            print('take action %s, max rho to %.2f, simulation times: 208' % (idx_chosen, min_rho))
            return self.find_best_line_to_reconnect(observation, action_chosen) if action_chosen else self.find_best_line_to_reconnect(observation, self.action_space({}))

        # second-round search with mild effect
        id_second_search = None
        min_rho0 = min_rho
        for idx, action_array in enumerate(self.actions1255):
            a = self.array2action(action_array)
            if not self.is_legal(a, observation):
                continue
            obs, _, done, _ = observation.simulate(a)
            if done:
                continue
            if obs.rho.max() < min(min_rho, min_rho0 - 0.03):
                min_rho = obs.rho.max()
                action_chosen = a
                id_second_search = idx
        if id_second_search:
            self.recovery_stack.append(id_second_search)
        return self.find_best_line_to_reconnect(observation, action_chosen) if action_chosen else self.find_best_line_to_reconnect(observation, self.action_space({}))


def make_agent(env, this_directory_path):
    my_agent = MyAgent(env.action_space, this_directory_path)
    return my_agent
