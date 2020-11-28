"""
In this file, we test the performance of our final-submitted agent on
the off-line dataset. Some tricks are integrated in it.

author: chen binbin
mail: cbb@cbb1996.com
"""
import os
import grid2op
from submission.my_agent import MyAgent


if __name__ == '__main__':
    # hyper-parameters
    DATA_PATH = './training_data_track1'  # for demo only, use your own dataset
    SCENARIO_PATH = './training_data_track1/chronics'

    # build grid2op environment
    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, backend=backend)
    except:
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH)
    env.seed(100)

    # build agent
    agent = MyAgent(env.action_space, './submission/')

    # test the agent's performance
    for chronic in range(len(os.listdir(SCENARIO_PATH))):
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs, 0, 0)
            obs, reward, done, info = env.step(action)
        if done:
            if 'GAME OVER' in str(info['exception']):
                print('%s: GAME OVER!' % str(obs.get_time_stamp()))
            else:
                print('Agent survives for all steps!')
