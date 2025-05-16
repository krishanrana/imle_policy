
import gym
import d4rl
import pdb


env = gym.make('kitchen-complete-v0')
env.reset()

# pdb.set_trace() 

for _ in range(1000):
    act = env.action_space.sample()
    act[:9] = 0.0
    act[3] = 5
    # pdb.set_trace() 
    obs, rew, done, info = env.step(act)


    pdb.set_trace()

    print(obs)

    

    # env.render()



