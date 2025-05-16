

import torch
import numpy as np
import collections
from tqdm import tqdm
import robosuite as suite
import cv2
from imle_policy.dataloaders.dataset_robomimic import normalize_data, unnormalize_data
from imle_policy.models.vision_network import get_resnet, replace_bn_with_gn
from imle_policy.models.rs_imle_network import SimpleActionGenerator
import torch
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from imle_policy.utils.rotation_transforms import quaternion_to_rotation_6D, rotation_6D_to_quaternion
import pdb



def process_obs(obs, task):

    pos_array = obs['robot0_eef_pos']
    quat_array = obs['robot0_eef_quat']
    quat_array = quaternion_to_rotation_6D(quat_array)
    gripper_array = obs['robot0_gripper_qpos']

    p_obs = np.concatenate([pos_array, quat_array, gripper_array])

    if task == 'ToolHang':
        front_image = np.moveaxis(obs['sideview_image'][::-1], -1, 0)
    else: 
        front_image = np.moveaxis(obs['agentview_image'][::-1], -1, 0)


    hand_image = np.moveaxis(obs['robot0_eye_in_hand_image'][::-1], -1, 0)

    obs = {
        'obs': p_obs,
        'front_image': front_image,
        'hand_image': hand_image}
    
    return obs

    
def evaluate(args, nets, stats, method='rs_imle'):

    # seed everything
    np.random.seed(args['seed_start'])
    torch.manual_seed(args['seed_start'])


    print("Evaluating policy...")
    env = suite.make(
        env_name=args['task'], # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names = ['robot0_eye_in_hand', 'sideview' if args['task'] == 'ToolHang' else 'agentview'],  # use agentview for visualization
        camera_heights = 240 if args['task'] == 'ToolHang' else 84,
        camera_widths = 240 if args['task'] == 'ToolHang' else 84,
        reward_shaping=True,
        controller_configs = {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], 
                              'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'damping': 1, 'impedance_mode': 'fixed', 
                              'kp_limits': [0, 300], 'damping_limits': [0, 10], 'position_limits': None, 'orientation_limits': None, 
                              'uncouple_pos_ori': True, 'control_delta': True, 'interpolation': None, 'ramp_ratio': 0.2},
        control_freq =  20,
        ignore_done = True  
    )

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
        np.random.seed(args['seed_start']+i)
        obs = env.reset()
        obs = process_obs(obs, args['task'])

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * args['obs_horizon'], maxlen=args['obs_horizon'])
        # save visualization and rewards
        rewards = list()
        done = False
        step_idx = 0
        infer_idx = 0
        success = False

        while not done:
            B = 1
            infer_idx += 1
            # stack the last obs_horizon number of observations
            front_image = np.stack([x['front_image'] for x in obs_deque])
            hand_image = np.stack([x['hand_image'] for x in obs_deque])
            obs = np.stack([x['obs'] for x in obs_deque])

            # normalize observation
            nobs = normalize_data(obs, stats=stats['obs'])
            nimage_front = front_image/255.0
            nimage_hand = hand_image/255.0

            # device transfer
            nimage_front = torch.from_numpy(nimage_front).to(device, dtype=torch.float32)
            nimage_hand = torch.from_numpy(nimage_hand).to(device, dtype=torch.float32)
            # (2,3,96,96)
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
            # (2,2)


            # infer action
            with torch.no_grad():
                # get image features
                image_features_front = nets['vision_encoder_0'](nimage_front)
                image_features_hand = nets['vision_encoder_1'](nimage_hand)
                image_features = torch.cat([image_features_front, image_features_hand], dim=-1)


                obs_features = torch.cat([image_features, nobs], dim=-1)
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
                obs, reward, done, info = env.step(action[i])
                obs = process_obs(obs, args['task'])
                obs_deque.append(obs)
                rewards.append(reward)
                # env.render(mode="human")
                # update progress bar

                if env._check_success():
                    success = True

                step_idx += 1
                if step_idx > args['max_steps']:
                    done = True
                if done:
                    break
                        

        if success:
            success_list.append(1)
        else:
            success_list.append(0)

        max_rewards.append(max(rewards))

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
