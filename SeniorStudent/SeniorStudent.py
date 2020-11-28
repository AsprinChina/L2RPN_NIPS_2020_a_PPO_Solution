"""
In this file, a multi-process training for PPO model is designed.
training process:
    The environment steps “do nothing” action (except reconnection of lines)
    until encountering a dangerous scenario, then its observation is sent to
    the Senior Student to get a “do something” action. After stepping this
    action, the reward is calculated and fed back to the Senior Student for
    network updating.

author: chen binbin
mail: cbb@cbb1996.com
"""
import os
import time
import grid2op
import numpy as np
from PPO import PPO
import tensorflow as tf
from PPO_Reward import PPO_Reward
from multiprocessing import cpu_count
from grid2op.Environment import SingleEnvMultiProcess


class Run_env(object):
    def __init__(self, envs, agent, n_steps=2000, n_cores=12, gamma=0.99, lam=0.95, action_space_path='../'):
        self.envs = envs
        self.agent = agent
        self.n_steps = n_steps
        self.n_cores = n_cores
        self.gamma = gamma
        self.lam = lam
        self.chosen = list(range(2, 7)) + list(range(7, 73)) + list(range(73, 184)) + list(range(184, 656))
        self.chosen += list(range(656, 715)) + list(range(715, 774)) + list(range(774, 833)) + list(range(833, 1010))
        self.chosen += list(range(1010, 1069)) + list(range(1069, 1105)) + list(range(1105, 1164)) + list(range(1164, 1223))
        self.chosen = np.asarray(self.chosen, dtype=np.int32) - 1  # (1221,)
        self.actions62 = np.load(os.path.join(action_space_path, 'actions62.npy'))
        self.actions146 = np.load(os.path.join(action_space_path, 'actions146.npy'))
        self.actions = np.concatenate((self.actions62, self.actions146), axis=0)
        self.batch_reward_records = []
        self.aspace = self.envs.action_space[0]
        self.rec_rewards = []
        self.worker_alive_steps = np.zeros(NUM_CORE)
        self.alive_steps_record = []

    def run_n_steps(self, n_steps=None):
        def swap_and_flatten(arr):
            shape = arr.shape
            return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

        self.n_steps = n_steps if n_steps is not None else self.n_steps
        # mb for mini-batch
        mb_obs, mb_rewards, mb_actions = [[] for _ in range(NUM_CORE)], [[] for _ in range(NUM_CORE)], [[] for _ in range(NUM_CORE)]
        mb_values, mb_dones, mb_neg_log_p = [[] for _ in range(NUM_CORE)], [[] for _ in range(NUM_CORE)], [[] for _ in range(NUM_CORE)]

        # start sampling
        obs_objs = self.envs.get_obs()
        obss = np.asarray([obs.to_vect()[self.chosen] for obs in obs_objs])  # (12, 1221,)
        dones = np.asarray([False for _ in range(NUM_CORE)])  # (12,)
        agent_step_rs = np.asarray([0 for _ in range(NUM_CORE)], dtype=np.float64)  # (12,)
        for _ in range(self.n_steps):
            self.worker_alive_steps += 1
            actions = np.asarray([None for _ in range(NUM_CORE)])  # 均为shape=(12,)的np array
            values = np.asarray([None for _ in range(NUM_CORE)])
            neg_log_ps = np.asarray([None for _ in range(NUM_CORE)])
            for id in range(NUM_CORE):
                if obss[id, 654:713].max() >= ACTION_THRESHOLD:
                    actions[id], values[id], neg_log_ps[id], _ = map(lambda x: x._numpy(), self.agent.step(tf.constant(obss[[id], :])))
                    if dones[id] == False and len(mb_obs[id]) > 0:
                        mb_rewards[id].append(agent_step_rs[id])
                    agent_step_rs[id] = 0
                    mb_obs[id].append(obss[[id], :])
                    mb_dones[id].append(dones[[id]])
                    dones[id] = False
                    mb_actions[id].append(actions[id])
                    mb_values[id].append(values[id])
                    mb_neg_log_p[id].append(neg_log_ps[id])
                else:
                    pass
            actions_array = [self.array2action(self.actions[i][0]) if i is not None else self.array2action(np.zeros(494), Run_env.reconnect_array(obs_objs[idx])) for idx, i in enumerate(actions)]
            obs_objs, rs, env_dones, infos = self.envs.step(actions_array)
            obss = np.asarray([obs.to_vect()[self.chosen] for obs in obs_objs])
            for id in range(NUM_CORE):
                if env_dones[id]:
                    # death or end
                    self.alive_steps_record.append(self.worker_alive_steps[id])
                    self.worker_alive_steps[id] = 0
                    if 'GAME OVER' in str(infos[id]['exception']):
                        dones[id] = True
                        mb_rewards[id].append(agent_step_rs[id] - 300)  # 上一个agent step的reward
                    else:
                        dones[id] = True
                        mb_rewards[id].append(agent_step_rs[id] + 500)
            agent_step_rs += rs
        # end sampling

        # batch to trajectory
        for id in range(NUM_CORE):
            if mb_obs[id] == []:
                continue
            if dones[id]:
                mb_dones[id].append(np.asarray([True]))
                mb_values[id].append(np.asarray([0]))
            else:
                mb_obs[id].pop()
                mb_actions[id].pop()
                mb_neg_log_p[id].pop()
        obs2ret, done2ret, action2ret, value2ret, neglogp2ret, return2ret = ([] for _ in range(6))
        for id in range(NUM_CORE):
            if mb_obs[id] == []:
                continue
            mb_obs_i = np.asarray(mb_obs[id], dtype=np.float32)
            mb_rewards_i = np.asarray(mb_rewards[id], dtype=np.float32)
            mb_actions_i = np.asarray(mb_actions[id], dtype=np.float32)
            mb_values_i = np.asarray(mb_values[id][:-1], dtype=np.float32)
            mb_neg_log_p_i = np.asarray(mb_neg_log_p[id], dtype=np.float32)
            mb_dones_i = np.asarray(mb_dones[id][:-1], dtype=np.bool)
            last_done = mb_dones[id][-1][0]
            last_value = mb_values[id][-1][0]

            # calculate R and A
            mb_advs_i = np.zeros_like(mb_values_i)
            last_gae_lam = 0
            for t in range(len(mb_obs[id]))[::-1]:
                if t == len(mb_obs[id]) - 1:
                    # last step
                    next_non_terminal = 1 - last_done
                    next_value = last_value
                else:
                    next_non_terminal = 1 - mb_dones_i[t + 1]
                    next_value = mb_values_i[t + 1]
                # calculate delta：r + gamma * v' - v
                delta = mb_rewards_i[t] + self.gamma * next_value * next_non_terminal - mb_values_i[t]
                mb_advs_i[t] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            mb_returns_i = mb_advs_i + mb_values_i
            obs2ret.append(mb_obs_i)
            action2ret.append(mb_actions_i)
            value2ret.append(mb_values_i)
            done2ret.append(mb_dones_i)
            neglogp2ret.append(mb_neg_log_p_i)
            return2ret.append(mb_returns_i)
        obs2ret = np.concatenate(obs2ret, axis=0)
        action2ret = np.concatenate(action2ret, axis=0)
        value2ret = np.concatenate(value2ret, axis=0)
        done2ret = np.concatenate(done2ret, axis=0)
        neglogp2ret = np.concatenate(neglogp2ret, axis=0)
        return2ret = np.concatenate(return2ret, axis=0)
        self.rec_rewards.append(sum([sum(i) for i in mb_rewards]) / action2ret.shape[0])
        return *map(swap_and_flatten, (obs2ret, return2ret, done2ret, action2ret, value2ret, neglogp2ret)), (sum([sum(i) for i in mb_rewards]) / action2ret.shape[0])

    @staticmethod
    def reconnect_array(obs):
        new_line_status_array = np.zeros_like(obs.rho)
        disconnected_lines = np.where(obs.line_status == False)[0]
        for line in disconnected_lines[::-1]:
            if not obs.time_before_cooldown_line[line]:
                line_to_reconnect = line  # reconnection
                new_line_status_array[line_to_reconnect] = 1
                break
        return new_line_status_array

    def array2action(self, total_array, reconnect_array=None):
        action = self.aspace({'change_bus': total_array[236:413]})
        action._change_bus_vect = action._change_bus_vect.astype(bool)
        if reconnect_array is None:
            return action
        action.update({'set_line_status': reconnect_array})
        return action


if __name__ == '__main__':
    # hyper-parameters
    ACTION_THRESHOLD = 0.9
    DATA_PATH = '../training_data_track1'  # for demo only, use your own dataset
    SCENARIO_PATH = '../training_data_track1/chronics'
    EPOCHS = 1000
    NUM_ENV_STEPS_EACH_EPOCH = 20000 # larger is better
    NUM_CORE = cpu_count()
    print('CPU counts：%d' % NUM_CORE)

    # Build single-process environment
    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, backend=backend, reward_class=PPO_Reward)
    except:
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, reward_class=PPO_Reward)
    env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
    # Convert to multi-process environment
    envs = SingleEnvMultiProcess(env=env, nb_env=NUM_CORE)
    envs.reset()

    # Build PPO agent
    agent = PPO(coef_entropy=1e-3, coef_value_func=0.01)

    # Build a runner
    runner = Run_env(envs, agent, action_space_path='../ActionSpace')

    logfile = ('./log/log-%s.txt' % time.strftime('%m-%d-%H-%M', time.localtime()))
    with open(logfile, 'w') as f:
        f.writelines('epoch, ave_r, ave_alive, policy_loss, value_loss, entropy, kl, clipped_ratio, time\n')

    print('start training... ...')
    for update in range(EPOCHS):
        # update learning rate
        lr_now = 6e-5 * np.linspace(1, 0.025, 500)[np.clip(update, 0, 499)]
        if update < 5:
            lr_now = 1e-4
        clip_range_now = 0.2

        # generate a new batch
        tick = time.time()
        obs, returns, dones, actions, values, neg_log_p, ave_r = runner.run_n_steps(NUM_ENV_STEPS_EACH_EPOCH)
        returns /= 20
        print('sampling number in this epoch: %d' % obs.shape[0])

        # update policy-value-network
        n = obs.shape[0]
        advs = returns - values
        advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)
        for _ in range(2):
            ind = np.arange(n)
            np.random.shuffle(ind)
            for batch_id in range(10):
                slices = (tf.constant(arr[ind[batch_id::10]]) for arr in (obs, returns, actions, values, neg_log_p, advs))
                policy_loss, value_loss, entropy, approx_kl, clip_ratio = agent.train(*slices,
                                                                                      lr=lr_now,
                                                                                      clip_range=clip_range_now)

        # logging
        print('epoch-%d, policy loss: %5.3f, value loss: %.5f, entropy: %.5f, approximate kl-divergence: %.5g, clipped ratio: %.5g' % (update, policy_loss, value_loss, entropy, approx_kl, clip_ratio))
        print('epoch-%d, ave_r: %5.3f, ave_alive: %5.3f, duration: %5.3f' % (update, ave_r, np.average(runner.alive_steps_record[-1000:]), time.time() - tick))
        with open(logfile, 'a') as f:
            f.writelines('%d, %.2f, %.2f, %.3f, %.3f, %.3f, %.3f, %.3f, %.2f\n' % (update, ave_r, np.average(runner.alive_steps_record[-1000:]), policy_loss, value_loss, entropy, approx_kl, clip_ratio, time.time() - tick))
        runner.agent.model.model.save('./ckpt/%d-%.2f' % (update, ave_r))
