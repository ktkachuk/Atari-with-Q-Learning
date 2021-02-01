

import numpy as np
import tensorflow as tf
import gym
import copy
from datetime import datetime
import os
import pickle as pkl
import gc
import time


def preprocess_image(image):
    """
    Rescale a given image to the size of 84 x 84 and convert to grayscale

    Args:
        image: Input image to be processed
    """
    img = tf.image.rgb_to_grayscale(tf.image.resize(image, (110,84)))
    img = tf.squeeze(img)[18:102]
    return img



def play_atari(model, episodes, game = 'Breakout-v0', render = True, n_frames = 4):
    """
    Plays an Atari game with a given agent

    Args:
        model: the trained agent network
        episodes: number of episodes to play
        game: name of the gym environment / the game
        render: show the game on screen
        n_frames: number of frames to stack for the models input
    """
    total_reward = 0
    env = gym.make(game)
    
    #Run all episodes
    for e in range(episodes):
        #Start episode
        frame = env.reset()
        frame = preprocess_image(frame)
        
        #Make initial input for the agent.
        #as you make recall out model takes n  = 4 frames as input
        observation = np.stack([frame for f in range(n_frames)],axis = 2)
        observation = observation[np.newaxis,:,:,:,np.newaxis]
        done = False
        step = 0
        
        #Run the episode
        while not done:
            step +=1
            
            #Get action from agent network
            q_values = model(observation)
            action = tf.math.argmax(q_values, axis = 1).numpy().item()
            
            #Execute action in the environment
            frame, reward, done, info = env.step(action)
            
            #Render frame
            if render:
                env.render()
                time.sleep(0.05)
                
            total_reward += reward

            #Include new frame into given observation and remove the oldes frame
            frame = preprocess_image(frame)
            observation = np.roll(observation, -1, axis = 3)
            observation[:,:,:,3,:] = frame[np.newaxis,:,:,np.newaxis]

        #Close environment
        env.close()




def main():
    model_path = r'.\model'
    network = tf.keras.models.load_model(model_path)
    play_atari(network, 1)



if __name__ == "__main__":
    main()