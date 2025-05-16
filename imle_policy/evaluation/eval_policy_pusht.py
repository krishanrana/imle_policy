import torch
import numpy as np
import collections
from tqdm import tqdm
from imle_policy.envs.pusht_env import PushTImageEnv
from imle_policy.dataloaders.dataset_pusht import normalize_data, unnormalize_data, PolicyDataset
from imle_policy.models.vision_network import get_resnet, replace_bn_with_gn
from imle_policy.models.rs_imle_network import SimpleActionGenerator
import torch
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import pdb
from imle_policy.models.rs_imle_network import GeneratorConditionalUnet1D
from imle_policy.models.diffusion_network import ConditionalUnet1D
import pdb
import os
import json
import wandb

def evaluate(args, nets, stats, method='rs_imle'):

    # seed everything
    np.random.seed(args['seed_start'])
    torch.manual_seed(args['seed_start'])


    print("Evaluating policy...")

    if method == 'rs_imle':
        if args['use_traj_consistency']:
            print("Using trajectory consistency")

    # limit enviornment interaction to 200 steps before termination
    env = PushTImageEnv()
    max_rewards = list()
    success_list = list()

    device = torch.device(args['device'])

    if method == 'diffusion':
        noise_scheduler = DDPMScheduler(
                            num_train_timesteps=args['num_diffusion_iters'],
                            # the choise of beta schedule has big impact on performance
                            # we found squared cosine works the best
                            beta_schedule='squaredcos_cap_v2',
                            # clip output to [-1,1] to improve stability
                            clip_sample=True,
                            # our network predicts noise (instead of denoised action)
                            prediction_type='epsilon')
    
        
    for i in tqdm(range(args['num_trails'])): 

        # get first observation
        obs, info = env.reset(seed=args['seed_start']+i)

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * args['obs_horizon'], maxlen=args['obs_horizon'])
        # save visualization and rewards
        rewards = list()
        done = False
        step_idx = 0
        infer_idx = 0

        # initialize prev_traj for rs_imle
        if method == 'rs_imle':
            # prev_traj = torch.zeros((1, args['pred_horizon'], args['action_dim']), device=device)
            prev_traj = torch.randn((1, args['pred_horizon'], args['action_dim']), device=device)

        while not done:
            B = 1
            infer_idx += 1
            # stack the last obs_horizon number of observations
            images = np.stack([x['image'] for x in obs_deque])
            agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

            # normalize observation
            nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
            nimages = images/255.0

            # device transfer
            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
            # (2,3,96,96)
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)

            # infer action
            with torch.no_grad():
                # get image features
                image_features = nets['vision_encoder_0'](nimages)
                # image_features = nets['vision_encoder'](nimages)
                obs_features = torch.cat([image_features, nagent_poses], dim=-1)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
                
                if method == 'rs_imle':
                    if args['use_traj_consistency']:
                        noise = torch.randn((32, args['pred_horizon'], args['action_dim']), device=device)
                        naction = nets['policy_net'](obs_cond, noise)

                        prev_traj_end = prev_traj[:,8:].reshape(1,-1)
                        gen_traj_start = naction[:,:8].reshape(32,-1)
                                                    
                        # Pick the generated trajectory that has its start closest to the end of the prev traj
                        distances = torch.cdist(gen_traj_start,prev_traj_end)
                        min_dist, min_idx = distances.min(dim=0)
                        naction = naction[min_idx]

                        if infer_idx % 5 == 0:
                            prev_traj = torch.randn((1, args['pred_horizon'], args['action_dim']), device=device)
                        else:
                            prev_traj = naction                             
                    else:
                        noise = torch.randn((1, args['pred_horizon'], args['action_dim']), device=device)
                        naction = nets['policy_net'](obs_cond, noise)


                elif method == 'diffusion':
                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (B, args['pred_horizon'], args['action_dim']), device=device)
                    naction = noisy_action

                    # init scheduler
                    noise_scheduler.set_timesteps(args['num_diffusion_iters'])

                    for k in noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = nets['policy_net'](
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )

                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                elif method == "flow_matching":
                    noisy_action = torch.randn(
                        (B, args['pred_horizon'], args['action_dim']), device=device)
                    naction = noisy_action

                    ts = torch.linspace(0.0, 1.0, args['num_flow_iters']+1, device=device)[:-1]
                    dt = 1.0 / args['num_flow_iters']
                    for t in ts:
                        timestep = (t * args['timestep_integer_scaler'])
                        timestep = timestep.long()

                        # predict noise
                        pred = nets['policy_net'](
                            sample=naction,
                            timestep=timestep,
                            global_cond=obs_cond
                        )
                        naction = naction + pred * dt



            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])

            # only take action_horizon number of actions
            start = args['obs_horizon'] - 1
            end = start + args['action_horizon']
            action = action_pred[start:end,:]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                obs, reward, done, _, info = env.step(action[i])
                obs_deque.append(obs)
                rewards.append(reward)
                # env.render(mode="human")
                # update progress bar
                step_idx += 1
                if step_idx > args['max_steps']:
                    done = True
                if done:
                    break
                        

        if max(rewards) > 0.95:
            success_list.append(1)
        else:
            success_list.append(0)

        max_rewards.append(max(rewards))
        # print("Current Average max reward:", np.mean(max_rewards))
        # print("Current Success Rate:", np.mean(success_list))

    print("Average max reward:", np.mean(max_rewards))
    print("Success Rate:", np.mean(success_list))

    return np.mean(max_rewards), np.mean(success_list)



if __name__ == "__main__":
    

    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)

    args_dict = {
        'method': 'diffusion',
        'task': 'pusht',
        'run_name': "diffusion",
        'vision_feature_dim': 512,
        'lowdim_obs_dim': 2,
        'obs_dim': 514,
        'action_dim': 2,
        'noise_dim': 32,
        'pred_horizon': 16,
        'num_diffusion_iters': 100,
        'obs_horizon': 2,
        'action_horizon': 8,
        'dataset_path': "../dataset/pkl_datasets/pusht.pkl",
        'device': 'cuda',
        'max_steps': 300,
        'seed_start': 100000,
        'num_trails': 50,
        'num_cameras': 1,
        'use_traj_consistency': False
    }

    stats = torch.load(f"../saved_weights/{args_dict['run_name']}/stats.pth")

    nets = nn.ModuleDict()


    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    # nets[f'vision_encoder'] = vision_encoder
    nets[f'vision_encoder_0'] = vision_encoder


    if args_dict['method'] == 'diffusion':
        policy_net = ConditionalUnet1D(
                        input_dim=args_dict['action_dim'],
                        global_cond_dim=args_dict['obs_dim']*args_dict['obs_horizon'])
        
        noise_scheduler = DDPMScheduler(
                            num_train_timesteps=args_dict['num_diffusion_iters'],
                            beta_schedule='squaredcos_cap_v2',
                            clip_sample=True,
                            prediction_type='epsilon')

    elif args_dict['method'] == 'rs_imle':
        policy_net = GeneratorConditionalUnet1D(
                    input_dim=args_dict['action_dim'],
                    global_cond_dim=args_dict['obs_dim']*args_dict['obs_horizon'])
        
    elif args_dict['method'] == 'flow_matching':
        policy_net = ConditionalUnet1D(
                    input_dim=args_dict['action_dim'],
                    global_cond_dim=args_dict['obs_dim']*args_dict['obs_horizon'])


    nets['policy_net'] = policy_net


    ckpt_path = f"../saved_weights/{args_dict['run_name']}/best_ema_net_weights.pth"
    state_dict = torch.load(ckpt_path, map_location='cuda')
    nets.load_state_dict(state_dict)

    # device transfer
    device = torch.device('cuda')
    _ = nets.to(device)

    evaluate(args_dict, nets, stats, method=args_dict['method'])
