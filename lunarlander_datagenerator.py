import numpy as np
import gym
import cv2
import matplotlib.pyplot as plt

def generate_lunarlander_data(
    rollouts=700,
    timesteps_per_rollout=150,
    image_size=(64,64),
    save_file=None
):
    '''
    Creates a .npz file containing images, observations, actions, etc
    of random rollouts of LunarLander-v2

    Args:
        rollouts (int): How many runs will be recorded
        timesteps_per_rollout (int): Nr of timesteps recorded per rollout
        image_size (int, int): Size of images to be stored in pixels
        save_file (str / None): Name of the file to store the dataset in 
    '''

    if save_file is None:
        save_file = "LunarLander-v2_{}_Dataset".format(
            rollouts*timesteps_per_rollout
        )

    imgs = []
    observations = []
    rewards = []
    dones = []
    actions = []
            
    env = gym.make("LunarLander-v2")

    for _ in range(rollouts):
        env.reset()

        for _ in range(timesteps_per_rollout):
            img = env.render('rgb_array')      
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            small_image = np.array(
                np.true_divide(
                    cv2.resize(
                        img, image_size, 
                        interpolation=cv2.INTER_CUBIC
                    ),
                    255
                ), 
                dtype = np.float16
            )
            
            imgs.append(small_image)
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            actions.append(action)
    np.savez(
        save_file, 
        parameter_rollouts=rollouts, 
        parameter_timesteps_per_rollout=timesteps_per_rollout,
        parameter_image_size=[image_size[0],image_size[1]],
        imgs=imgs, rewards=rewards, actions=actions, 
        dones=dones, observations=observations
    )


if __name__ == "__main__":
    '''
    If run directly this will generate the data 
    needed to run all LunarLander-v2 experiments
    '''
    generate_lunarlander_data()