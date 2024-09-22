import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy.spatial.transform import Rotation

key_names = [
    'current_color',
    'current_depth',
    'current_intrinsics',
    'current_extrinsics',
    'next_color',
    'next_depth',
    'next_intrinsics',
    'next_extrinsics',
    'current_pts',
    'current_cols',
    'next_pts',
    'next_cols'
]

def custom_collate(batch):
    result = {}
    for key in batch[0].keys():
        if key.endswith('pts') or key.endswith('cols') or key.endswith('instruction'):
            # For keys ending with 'pts' or 'cols', keep as a list
            result[key] = [item[key] for item in batch]
        else:
            # For other keys, stack them
            result[key] = torch.stack([item[key] for item in batch])
    return result

def get_dataset(bs, num_workers=0):
    dataset = RobotDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers, collate_fn=custom_collate)
    return dataloader

class RobotDataset(Dataset):
    def __init__(self, root_dir='data', transform=None, device='cuda:0'):
        self.root_dir = root_dir
        self.transform = transform
        self.episodes = []
        self.device = device

        self.x_bounds = (0.1, 0.5)
        self.y_bounds = (-0.4, 0.4)
        self.z_bounds = (-0.2, 0.8)
        
        # Collect all episode files
        for task in os.listdir(root_dir):
            task_dir = os.path.join(root_dir, task)
            if os.path.isdir(task_dir):
                for episode in os.listdir(task_dir):
                    if episode.endswith('.npz'):
                        self.episodes.append(os.path.join(task_dir, episode))

    def rgbd2pc(self, color: torch.Tensor, depth: torch.Tensor, intrinsic: torch.Tensor, extrinsic: torch.Tensor):
        '''
        color: 3, h, w
        depth: h, w
        '''
        h, w = depth.shape
        i, j = torch.meshgrid(torch.arange(w, device=color.device), 
                              torch.arange(h, device=color.device), indexing='xy')
        z = depth * 1.
        valid_mask = z > 0  # Check for valid depth values

        x = (i - intrinsic[0, 2]) * z / intrinsic[0, 0]
        y = (j - intrinsic[1, 2]) * z / intrinsic[1, 1]
        points = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
        colors = color.reshape(3, -1).permute(1, 0) # npts, 3

        mask = valid_mask.reshape(-1) & (points[:, 2] <= 1)
        points = points[mask]
        colors = colors[mask]

        R = extrinsic[:3, :3] * 1.
        t = extrinsic[:3, 3] * 1.
        points_transformed = (R @ points.T).T + t

        crop_mask = (points_transformed[:, 0] >= self.x_bounds[0]) & (points_transformed[:, 0] <= self.x_bounds[1]) & (points_transformed[:, 1] >= self.y_bounds[0]) \
            & (points_transformed[:, 1] <= self.y_bounds[1]) & (points_transformed[:, 2] >= self.z_bounds[0]) & (points_transformed[:, 2] <= self.z_bounds[1])
        cropped_points = points_transformed[crop_mask]
        cropped_colors = colors[crop_mask]

        return cropped_points, cropped_colors

    def process_euler(self, rot):
        r = Rotation.from_euler("xyz", rot, degrees=True)
        euler = r.as_euler("xyz", degrees=True)
        return euler
        
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx, overlap_extrinsic=True):
        episode_path = self.episodes[idx]
        data = np.load(episode_path, allow_pickle=True)
        frames = data['frames']
        instruction = str(data['instruction'])
        if overlap_extrinsic:
            extrinsics = np.load('calibration/extrinsic.npy')
        else:
            extrinsics = current_frame['camera_data'][0]['extrinsics']
        
        # Randomly select a frame index (except the last one)
        frame_idx = np.random.randint(0, len(frames) - 1)
        
        # Get current and next frame
        current_frame = frames[frame_idx]
        next_frame = frames[frame_idx + 1]
        
        # Process current frame
        current_camera_data = current_frame['camera_data'][0]
        current_color = current_camera_data['color'][..., ::-1]  # BGR to RGB
        current_depth = current_camera_data['depth'] * 0.00025  # mm to m
        current_intrinsics = current_camera_data['intrinsics']
        current_extrinsics = extrinsics * 1.
        current_gripper_pos = current_frame['gripper_pose']['position']
        current_gripper_rot = current_frame['gripper_pose']['rotation']
        current_gripper_rot = self.process_euler(current_gripper_rot)
        current_gripper_state = current_frame['gripper_state']
        current_ignore_collision = int(current_frame['ignore_collision'])
        
        # Process next frame
        next_camera_data = next_frame['camera_data'][0]
        next_color = next_camera_data['color'][..., ::-1]  # BGR to RGB
        next_depth = next_camera_data['depth'] * 0.00025  # mm to m
        next_intrinsics = next_camera_data['intrinsics']
        next_extrinsics = extrinsics * 1.
        next_gripper_pos = next_frame['gripper_pose']['position']
        next_gripper_rot = next_frame['gripper_pose']['rotation']
        next_gripper_rot = self.process_euler(next_gripper_rot)
        next_gripper_state = next_frame['gripper_state']
        next_ignore_collision = int(next_frame['ignore_collision'])
        
        # Convert numpy arrays to PyTorch tensors
        current_color = torch.from_numpy(current_color.copy()).permute(2, 0, 1).float().to(self.device) / 255.0
        current_depth = torch.from_numpy(current_depth).float().to(self.device)
        next_color = torch.from_numpy(next_color.copy()).permute(2, 0, 1).float().to(self.device) / 255.0
        next_depth = torch.from_numpy(next_depth).float().to(self.device)

        current_intrinsics = torch.tensor(current_intrinsics, dtype=torch.float, device=self.device)
        current_extrinsics = torch.tensor(current_extrinsics, dtype=torch.float, device=self.device)
        next_intrinsics = torch.tensor(next_intrinsics, dtype=torch.float, device=self.device)
        next_extrinsics = torch.tensor(next_extrinsics, dtype=torch.float, device=self.device)

        current_gripper_pos = torch.tensor(current_gripper_pos, dtype=torch.float, device=self.device)
        current_gripper_rot = torch.tensor(current_gripper_rot, dtype=torch.float, device=self.device)
        current_gripper_state = torch.tensor(current_gripper_state, dtype=torch.float, device=self.device)
        next_gripper_pos = torch.tensor(next_gripper_pos, dtype=torch.float, device=self.device)
        next_gripper_rot = torch.tensor(next_gripper_rot, dtype=torch.float, device=self.device)
        next_gripper_state = torch.tensor(next_gripper_state, dtype=torch.float, device=self.device)

        current_ignore_collision = torch.tensor(current_ignore_collision, dtype=torch.float, device=self.device)
        next_ignore_collision = torch.tensor(next_ignore_collision, dtype=torch.float, device=self.device)

        current_pts, current_cols = self.rgbd2pc(current_color, current_depth, current_intrinsics, current_extrinsics)
        next_pts, next_cols = self.rgbd2pc(next_color, next_depth, next_intrinsics, next_extrinsics)
        
        # Apply transforms if any
        if self.transform:
            current_color = self.transform(current_color)
            next_color = self.transform(next_color)
        
        return {
            'instruction': instruction,
            'current_color': current_color,
            'current_depth': current_depth,
            'current_intrinsics': current_intrinsics,
            'current_extrinsics': current_extrinsics,
            'next_color': next_color,
            'next_depth': next_depth,
            'next_intrinsics': next_intrinsics,
            'next_extrinsics': next_extrinsics,
            'current_pts': current_pts,
            'current_cols': current_cols,
            'next_pts': next_pts,
            'next_cols': next_cols,
            'current_gripper_pos': current_gripper_pos,
            'current_gripper_rot': current_gripper_rot,
            'current_gripper_state': current_gripper_state,
            'next_gripper_pos': next_gripper_pos,
            'next_gripper_rot': next_gripper_rot,
            'next_gripper_state': next_gripper_state,
            'current_ignore_collision': current_ignore_collision,
            'next_ignore_collision': next_ignore_collision,
        }




# Usage example:
if __name__ == '__main__':
    dataset = get_dataset(bs=4)
    for batch in dataset:
        print(batch)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)