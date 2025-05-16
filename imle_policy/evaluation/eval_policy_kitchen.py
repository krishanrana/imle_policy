

import torch
import numpy as np
import collections
from tqdm import tqdm
from imle_policy.dataloaders.dataset_kitchen import normalize_data, unnormalize_data
from imle_policy.models.vision_network import get_resnet, replace_bn_with_gn
from imle_policy.models.rs_imle_network import SimpleActionGenerator
import torch
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import pdb
import d4rl
import gym

def evaluate(args, nets, stats, method='rs_imle'):

    # seed everything
    np.random.seed(args['seed_start'])
    torch.manual_seed(args['seed_start'])


    print("Evaluating policy...")

    if method == 'rs_imle':
        if args['use_traj_consistency']:
            print("Using trajectory consistency")

    # limit enviornment interaction to 200 steps before termination
    env = gym.make('kitchen-mixed-v0')
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
        # seed everything
        np.random.seed(args['seed_start']+i)
        torch.manual_seed(args['seed_start']+i)
        obs = env.reset()

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * args['obs_horizon'], maxlen=args['obs_horizon'])
        # save visualization and rewards
        rewards = list()
        success = False
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
            agent_obs = np.stack([x for x in obs_deque])

            # normalize observation
            nagent_obs = normalize_data(agent_obs, stats=stats['obs'])

            # (2,3,96,96)
            nagent_obs = torch.from_numpy(nagent_obs).to(device, dtype=torch.float32)
            # (2,2)

            # infer action
            with torch.no_grad():
                # get image features
                obs_cond = nagent_obs.unsqueeze(0).flatten(start_dim=1)

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
                obs, reward, done, info = env.step(action[i])
                obs_deque.append(obs)
                rewards.append(reward)
                # env.render(mode="human")
                # update progress bar
                step_idx += 1

                if reward == 4:
                    success = True
                    done = True
                if step_idx > args['max_steps']:
                    done = True
                if done:
                    break
                        

        if success:
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
        'run_name': "confused-gorge-10_rs_imle",
        'checkpoint': 800,
        'vision_feature_dim': 512,
        'lowdim_obs_dim': 2,
        'obs_dim': 514,
        'action_dim': 2,
        'noise_dim': 32,
        'pred_horizon': 16,
        'num_diffusion_iters': 100,
        'obs_horizon': 2,
        'action_horizon': 8,
        'dataset_path': "dataset/pusht_cchi_v7_replay.zarr.zip",
        'device': 'cuda',
        'max_steps': 300,
        'seed_start': 100000,
        'num_trails': 50
    }


    dataset = PushTImageDataset(
        dataset_path=args_dict['dataset_path'],
        pred_horizon=args_dict['pred_horizon'],
        obs_horizon=args_dict['obs_horizon'],
        action_horizon=args_dict['action_horizon']
    )
    stats = dataset.stats

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )


    generator = SimpleActionGenerator(
        state_dim=args_dict['obs_dim']*args_dict['obs_horizon'],
        action_dim=args_dict['action_dim'],
        noise_dim=args_dict['noise_dim']
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'generator': generator
    })

    ckpt_path = f"saved_weights/{args_dict['run_name']}/ema_net_weights_{args_dict['checkpoint']}.pth"
    state_dict = torch.load(ckpt_path, map_location='cuda')
    nets.load_state_dict(state_dict)

    # device transfer
    device = torch.device('cuda')
    _ = nets.to(device)

    evaluate(args_dict, nets, stats)
