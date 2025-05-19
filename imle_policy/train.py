from torch import nn
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import numpy as np
import wandb
import copy
import os
import sys
import argparse
import json
import logging

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imle_policy.models.rs_imle_network import GeneratorConditionalUnet1D
from imle_policy.models.diffusion_network import ConditionalUnet1D
from imle_policy.models.vision_network import get_resnet, replace_bn_with_gn
from imle_policy.utils.losses import rs_imle_loss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Policy')
    parser.add_argument('--method', type=str, default='rs_imle',
                      help='Training method (rs_imle, diffusion, flow_matching)')
    parser.add_argument('--epsilon', type=float, default=0.03,
                      help='IMLE epsilon parameter')
    parser.add_argument('--n_samples_per_condition', type=int, default=20,
                      help='Number of samples per condition')
    parser.add_argument('--dataset_percentage', type=float, default=1.0,
                      help='Percentage of dataset to use')
    parser.add_argument('--task', type=str, default='pusht',
                      help='Task to train on (pusht, Lift, PickPlaceCan, etc.)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--use_traj_consistency', type=bool, default=False,
                      help='Use trajectory consistency')
    parser.add_argument('--wandb_run_name', type=str, default='testing',
                      help='Wandb run name')
    return parser.parse_args()

def load_config(task):
    # Get the directory where train_policy.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_dir, 'configs', f'{task}_config.json')
    
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}. Please make sure the config file exists in rs_imle_policy/configs/ directory.")
        
    with open(config_file_path, 'r') as file:
        return json.load(file)

def setup_wandb(args_dict):
    wandb.init(project=args_dict['wandb_run_name'])

    if wandb.run.name is None: # if user does not login to wandb
        wandb.run.name = args_dict['wandb_run_name']
    
    if args_dict['method'] == 'diffusion':
        run_name = wandb.run.name + "_" + args_dict['method'] + f"_dataset_percentage_{args_dict['dataset_percentage']}_{args_dict['task']}"
    elif args_dict['method'] == 'rs_imle':
        run_name = wandb.run.name + "_" + args_dict['method'] + f"_eps_{args_dict['epsilon']}_;_n_samples_{args_dict['n_samples_per_condition']}__dataset_percentage_{args_dict['dataset_percentage']}_{args_dict['task']}"
    elif args_dict['method'] == 'flow_matching':
        run_name = wandb.run.name + "_" + args_dict['method'] + f"_num_flow_iters_{args_dict['num_flow_iters']}_dataset_percentage_{args_dict['dataset_percentage']}_{args_dict['task']}"
    
    wandb.run.name = run_name
    os.makedirs(f'saved_weights/{run_name}', exist_ok=True)
    wandb.config.update(args_dict)
    return run_name

def get_dataset_class(task):
    if task == 'pusht':
        from imle_policy.dataloaders.dataset_pusht import PolicyDataset
        from imle_policy.evaluation.eval_policy_pusht import evaluate
    elif task == 'ToolHang':
        from imle_policy.dataloaders.dataset_robomimic_hdf5 import PolicyDataset
        from imle_policy.evaluation.eval_policy_robomimic import evaluate
    elif task == 'TwoArmTransport':
        from imle_policy.dataloaders.dataset_robomimic_hdf5 import PolicyDataset
        from imle_policy.evaluation.eval_policy_robomimic_two_arm import evaluate
    elif task == 'ur3_blockpush':
        from imle_policy.dataloaders.dataset_ur3_blockpush import PolicyDataset
        from imle_policy.evaluation.eval_policy_ur3_blockpush import evaluate
    elif task == 'kitchen':
        from imle_policy.dataloaders.dataset_kitchen import PolicyDataset
        from imle_policy.evaluation.eval_policy_kitchen import evaluate
    else:
        from imle_policy.dataloaders.dataset_robomimic import PolicyDataset
        from imle_policy.evaluation.eval_policy_robomimic import evaluate
    return PolicyDataset, evaluate

def create_networks(args_dict):
    nets = nn.ModuleDict()
    
    if args_dict['task'] != "ur3_blockpush" or args_dict['task'] != "kitchen":
        for i in range(args_dict['num_cameras']):
            vision_encoder = get_resnet('resnet18')
            vision_encoder = replace_bn_with_gn(vision_encoder)
            nets[f'vision_encoder_{i}'] = vision_encoder

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
    return nets, noise_scheduler if args_dict['method'] == 'diffusion' else None

def process_batch(nbatch, nets, device, args_dict):
    if args_dict['task'] == 'pusht':
        # data normalized in dataset
        nimage = nbatch['image'][:,:args_dict['obs_horizon']].to(device)
        nagent_pos = nbatch['agent_pos'][:,:args_dict['obs_horizon']].to(device)

        # encoder vision features
        image_features = nets['vision_encoder_0'](
            nimage.flatten(end_dim=1))
        image_features = image_features.reshape(
            *nimage.shape[:2],-1)

        # concatenate vision feature and low-dim obs
        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)

    elif args_dict['task'] == 'ur3_blockpush':
        nobs = nbatch['obs'][:,:args_dict['obs_horizon']].to(device)
        obs_cond = nobs.flatten(start_dim=1)

    elif args_dict['task'] == 'kitchen':
        nobs = nbatch['obs'][:,:args_dict['obs_horizon']].to(device)
        obs_cond = nobs.flatten(start_dim=1)

    elif args_dict['task'] == 'TwoArmTransport':
        nimage_front_0 = nbatch['front_images_0'][:,:args_dict['obs_horizon']].to(device)
        nimage_hand_0 = nbatch['hand_images_0'][:,:args_dict['obs_horizon']].to(device)
        nimage_front_1 = nbatch['front_images_1'][:,:args_dict['obs_horizon']].to(device)
        nimage_hand_1 = nbatch['hand_images_1'][:,:args_dict['obs_horizon']].to(device)

        nagent_obs = nbatch['obs'][:,:args_dict['obs_horizon']].to(device)

        # encoder vision features front
        image_features_front_0 = nets['vision_encoder_0'](nimage_front_0.flatten(end_dim=1))
        image_features_front_0 = image_features_front_0.reshape(*nimage_front_0.shape[:2],-1)
        # encoder vision features hand
        image_features_hand_0 = nets['vision_encoder_1'](nimage_hand_0.flatten(end_dim=1))
        image_features_hand_0 = image_features_hand_0.reshape(*nimage_hand_0.shape[:2],-1)

        image_features_front_1 = nets['vision_encoder_2'](nimage_front_1.flatten(end_dim=1))
        image_features_front_1 = image_features_front_1.reshape(*nimage_front_1.shape[:2],-1)

        image_features_hand_1 = nets['vision_encoder_3'](nimage_hand_1.flatten(end_dim=1))
        image_features_hand_1 = image_features_hand_1.reshape(*nimage_hand_1.shape[:2],-1)

        image_features = torch.cat([image_features_front_0, image_features_hand_0, image_features_front_1, image_features_hand_1], dim=-1)

        # concatenate vision feature and low-dim obs
        obs_features = torch.cat([image_features, nagent_obs], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)
    else:
        nimage_front = nbatch['front_images'][:,:args_dict['obs_horizon']].to(device)
        nimage_hand = nbatch['hand_images'][:,:args_dict['obs_horizon']].to(device)
        nagent_obs = nbatch['obs'][:,:args_dict['obs_horizon']].to(device)

        # encoder vision features front
        image_features_front = nets['vision_encoder_0'](nimage_front.flatten(end_dim=1))
        image_features_front = image_features_front.reshape(*nimage_front.shape[:2],-1)

        # encoder vision features hand
        image_features_hand = nets['vision_encoder_1'](nimage_hand.flatten(end_dim=1))
        image_features_hand = image_features_hand.reshape(*nimage_hand.shape[:2],-1)

        image_features = torch.cat([image_features_front, image_features_hand], dim=-1)

        # concatenate vision feature and low-dim obs
        obs_features = torch.cat([image_features, nagent_obs], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)

    return obs_cond

def train_diffusion_step(nets, noise_scheduler, obs_cond, naction, B, device):
    # sample noise to add to actions
    noise = torch.randn(naction.shape, device=device)

    # sample a diffusion iteration for each data point
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (B,), device=device
    ).long()

    noisy_actions = noise_scheduler.add_noise(
        naction, noise, timesteps)

    # predict the noise residual
    noise_pred = nets['policy_net'](
        noisy_actions, timesteps, global_cond=obs_cond)

    return nn.functional.mse_loss(noise_pred, noise)

def train_rs_imle_step(nets, obs_cond, naction, B, args_dict, device):
    noise = torch.randn(B * args_dict['n_samples_per_condition'], *naction.shape[1:], device=device)
    repeated_obs_cond = obs_cond.repeat_interleave(args_dict['n_samples_per_condition'], dim=0)

    pred_actions = nets['policy_net'](repeated_obs_cond, noise)
    pred_actions = pred_actions.reshape(B, args_dict['n_samples_per_condition'], *naction.shape[1:])

    # Compute IMLE loss
    return rs_imle_loss(naction, pred_actions, args_dict['epsilon'])

def train_flow_matching_step(nets, obs_cond, naction, B, args_dict, device):
    noise = torch.randn(naction.shape, device=device)
    t = torch.rand(B, device=device)
    t_shaped = t.reshape(-1, *([1] * (noise.dim() - 1)))
    xt = t_shaped * naction + (1 - t_shaped) * noise
    vector = naction - noise
    timesteps = (t * args_dict['timestep_integer_scaler']).long()
    pred = nets['policy_net'](
        xt, timesteps, global_cond=obs_cond)

    return nn.functional.mse_loss(pred, vector)

def save_checkpoint(args_dict, nets, ema, epoch_idx, best_mean_success, stats, run_name, train_step, evaluate_fn):
    ema_nets = copy.deepcopy(nets)
    ema.copy_to(ema_nets.parameters())

    if (args_dict['task'] != 'pusht_real') and (args_dict['task'] != 'shoe_rack_real'):
        mean_cov, mean_success = evaluate_fn(args_dict, ema_nets, stats, method=args_dict['method'])

        if mean_success > best_mean_success:
            best_mean_success = mean_success
            torch.save(nets.state_dict(), f'saved_weights/{run_name}/best_net_weights.pth')
            torch.save(ema_nets.state_dict(), f'saved_weights/{run_name}/best_ema_net_weights.pth')

        wandb.log({'mean_max_reward': mean_cov, 'mean_success_rate': mean_success}, step=train_step)
    else:
        torch.save(nets.state_dict(), f'saved_weights/{run_name}/net_weights_{epoch_idx}.pth')
        torch.save(ema_nets.state_dict(), f'saved_weights/{run_name}/ema_net_weights_{epoch_idx}.pth')
    
    return best_mean_success

def train(args_dict, nets, dataloader, device, noise_scheduler=None, stats=None, run_name=None, evaluate_fn=None):
    train_step = 0
    best_mean_success = 0
    
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * args_dict['num_epochs']
    )

    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    with tqdm(range(args_dict['num_epochs']), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = list()
            wandb.log({'epoch': epoch_idx}, step=train_step)
            
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    obs_cond = process_batch(nbatch, nets, device, args_dict)
                    naction = nbatch['action'].to(device)
                    B = naction.shape[0]

                    if args_dict['method'] == 'diffusion':
                        loss = train_diffusion_step(nets, noise_scheduler, obs_cond, naction, B, device)
                    elif args_dict['method'] == 'rs_imle':
                        loss, loss_logs = train_rs_imle_step(nets, obs_cond, naction, B, args_dict, device)
                        wandb.log(loss_logs, step=train_step)
                    elif args_dict['method'] == 'flow_matching':
                        loss = train_flow_matching_step(nets, obs_cond, naction, B, args_dict, device)

                    if loss == 0:
                        wandb.log({"zero_loss": 1}, step=train_step)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(nets.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()
                        ema.step(nets.parameters())
                        wandb.log({"zero_loss": 0}, step=train_step)

                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
                    wandb.log({'loss': loss_cpu}, step=train_step)
                    train_step += 1

            if (epoch_idx % 50 == 0):
                best_mean_success = save_checkpoint(
                    args_dict, nets, ema, epoch_idx, best_mean_success,
                    stats, run_name, train_step, evaluate_fn
                )

            tglobal.set_postfix(loss=np.mean(epoch_loss))

def main():
    args = parse_args()
    args_dict = vars(args)
    
    # Load task-specific config
    task_config = load_config(args.task)
    args_dict.update(task_config)
    
    # Setup wandb
    run_name = setup_wandb(args_dict)
    
    # Set random seeds
    np.random.seed(args_dict['seed'])
    torch.manual_seed(args_dict['seed'])
    
    # Get dataset class and evaluation function
    PolicyDataset, evaluate_fn = get_dataset_class(args.task)
    
    # Create dataset and dataloader
    dataset = PolicyDataset(
        dataset_path=args_dict['dataset_path'],
        pred_horizon=args_dict['pred_horizon'],
        obs_horizon=args_dict['obs_horizon'],
        action_horizon=args_dict['action_horizon'],
        dataset_percentage=args_dict['dataset_percentage']
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args_dict['batch_size'],
        num_workers=11,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Save dataset stats
    stats = dataset.stats
    torch.save(stats, f'saved_weights/{run_name}/stats.pth')
    
    # Create networks
    device = torch.device('cuda')
    nets, noise_scheduler = create_networks(args_dict)
    nets = nets.to(device)
    
    # Train
    train(args_dict, nets, dataloader, device, noise_scheduler, stats, run_name, evaluate_fn)

if __name__ == "__main__":
    main()

