import os
import time
import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch
import imageio
from tqdm import tqdm

from rl_algorithm import DQN
from custom_env import ImageEnv
from utils import seed_everything, YOUR_CODE_HERE
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    # environment hyperparameters
    parser.add_argument('--env_name', type=str, default='ALE/MsPacman-v5')
    parser.add_argument('--state_dim', type=tuple, default=(4, 84, 84))
    parser.add_argument('--image_hw', type=int, default=84, help='The height and width of the image')
    parser.add_argument('--num_envs', type=int, default=4)
    # DQN hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epsilon', type=float, default=0.99)
    parser.add_argument('--epsilon_min', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--buffer_size', type=int, default=int(1e5))
    parser.add_argument('--target_update_interval', type=int, default=10000)
    # training hyperparameters
    parser.add_argument('--max_steps', type=int, default=int(3e6))
    parser.add_argument('--eval_interval', type=int, default=5000)
    # others
    parser.add_argument('--save_root', type=Path, default='./submissions')
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    # evaluation
    parser.add_argument('--eval', action="store_true", help='evaluate the model')
    parser.add_argument('--eval_model_path', type=str, default=None, help='the path of the model to evaluate')
    if not parser.parse_args().eval:
        f = open(f'{parser.parse_args().save_root}/parameter.txt', 'w')
        f.write(str(parser.parse_args()))
        f.close()
    return parser.parse_args()

def validation(agent, num_evals=5):
    eval_env = gym.make('ALE/MsPacman-v5')
    eval_env = ImageEnv(eval_env)
    
    scores = 0
    for i in range(num_evals):
        (state, _), done = eval_env.reset(), False
        while not done:
            action = agent.act(state, training=False)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            
            state = next_state
            scores += reward
            done = terminated or truncated
    return np.round(scores / num_evals, 4)

def train(agent, env):
    history = {'Step': [], 'AvgScore': [], 'AvgLoss': [], 'AvgReward': []}
    rewards = []
    losses = []
    highest_score = 0
    (state, _) = env.reset()
    
    for _ in tqdm(range(args.max_steps)):
        
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        result = agent.process((state, action, reward, next_state, terminated)) 
        # You can track q-losses over training from `result` variable.
        
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()

        rewards.append(result["reward"])
        if "value_loss" in result:
            losses.append(result["value_loss"])

        if agent.total_steps % args.eval_interval == 0:
            avg_score = validation(agent)
            avg_reward = np.mean(rewards[-args.eval_interval:])
            avg_loss = np.mean(losses[-args.eval_interval:])
            history['Step'].append(agent.total_steps)
            history['AvgScore'].append(avg_score)
            history['AvgLoss'].append(avg_loss)
            history['AvgReward'].append(avg_reward)
            
            # log info to plot your figure
            print(f"Step: {agent.total_steps}, AvgScore: {avg_score}, AvgReward: {avg_reward}, AvgLoss: {avg_loss}")
            utils.log_info(history, args.save_root / 'training_log.csv')
            # save model
            if avg_score >highest_score :
                highest_score = avg_score
                global highest_eps 
                highest_eps = True
            torch.save(agent.network.state_dict(), save_dir / 'pacman_dqn.pt')
            print("Model saved!")

def evaluate(agent, eval_env, capture_frames=True):
    seed_everything(0, eval_env) # don't modify
    
    # load the model
    if agent is None:
        action_dim = eval_env.action_space.n
        state_dim = (args.num_envs, args.image_hw, args.image_hw)
        agent = DQN(state_dim=state_dim, action_dim=action_dim)
        agent.network.load_state_dict(torch.load(args.eval_model_path))
    
    (state, _), done = eval_env.reset(), False

    scores = 0
    # Record the frames
    global highest_eps
    if capture_frames:
        writer = imageio.get_writer(save_dir / 'mspacman.mp4', fps=10)

    while not done:
        if capture_frames:
            writer.append_data(eval_env.render())
            
            # highest_eps = False
        else:
            eval_env.render()
        
        action = agent.act(state, training=False)
        next_state, reward, terminated, truncated, info = eval_env.step(action)
        state = next_state
        scores += reward
        done = terminated or truncated
    if capture_frames:
        # print("writer closed")
        writer.close()
    highest_eps = False
    print("The score of the agent: ", scores)

def main():
    env = gym.make(args.env_name)
    env = ImageEnv(env, stack_frames=args.num_envs, image_hw=args.image_hw)

    action_dim = env.action_space.n
    state_dim = (args.num_envs, args.image_hw, args.image_hw)
    agent = DQN(state_dim=state_dim, action_dim=action_dim,\
                    lr=args.lr, epsilon=args.epsilon, epsilon_min=args.epsilon_min,\
                    gamma=args.gamma, batch_size=args.batch_size,\
                    warmup_steps=args.warmup_steps,buffer_size=args.buffer_size,\
                    target_update_interval = args.target_update_interval)
    
    # train
    train(agent, env)
    
    # evaluate
    eval_env = gym.make(args.env_name, render_mode='rgb_array')
    eval_env = ImageEnv(eval_env, stack_frames=args.num_envs, image_hw=args.image_hw)
    # agent.network.load_state_dict(torch.load(save_dir / 'pacman_dqn.pt'))
    evaluate(agent, eval_env)
highest_eps = False
if __name__ == "__main__":
    args = parse_args()
    
    
    save_dir = args.save_root
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    
    if args.eval:
        eval_env = gym.make(args.env_name, render_mode='rgb_array')
        eval_env = ImageEnv(eval_env, stack_frames=args.num_envs, image_hw=args.image_hw)
        evaluate(agent=None, eval_env=eval_env, capture_frames=False)
    else:
        main()
