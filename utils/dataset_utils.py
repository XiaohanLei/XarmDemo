import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

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
        if key.endswith('pts') or key.endswith('cols'):
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

        return points_transformed, colors
        
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode_path = self.episodes[idx]
        data = np.load(episode_path, allow_pickle=True)
        frames = data['frames']
        instruction = str(data['instruction'])
        
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
        current_extrinsics = current_camera_data['extrinsics']
        current_gripper_pos = current_frame['gripper_pose']['position']
        current_gripper_rot = current_frame['gripper_pose']['rotation']
        current_gripper_state = current_frame['gripper_state']
        
        # Process next frame
        next_camera_data = next_frame['camera_data'][0]
        next_color = next_camera_data['color'][..., ::-1]  # BGR to RGB
        next_depth = next_camera_data['depth'] * 0.00025  # mm to m
        next_intrinsics = next_camera_data['intrinsics']
        next_extrinsics = next_camera_data['extrinsics']
        next_ripper_pos = next_frame['gripper_pose']['position']
        next_gripper_rot = next_frame['gripper_pose']['rotation']
        next_gripper_state = next_frame['gripper_state']
        
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
        next_ripper_pos = torch.tensor(next_ripper_pos, dtype=torch.float, device=self.device)
        next_gripper_rot = torch.tensor(next_gripper_rot, dtype=torch.float, device=self.device)
        next_gripper_state = torch.tensor(next_gripper_state, dtype=torch.float, device=self.device)

        current_pts, current_cols = self.rgbd2pc(current_color, current_depth, current_intrinsics, current_extrinsics)
        next_pts, next_cols = self.rgbd2pc(next_color, next_depth, next_intrinsics, next_extrinsics)
        
        # Apply transforms if any
        if self.transform:
            current_color = self.transform(current_color)
            next_color = self.transform(next_color)
        
        return {
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
            'next_ripper_pos': next_ripper_pos,
            'next_gripper_rot': next_gripper_rot,
            'next_gripper_state': next_gripper_state
        }




# Usage example:
if __name__ == '__main__':
    dataset = get_dataset(bs=4)
    for batch in dataset:
        print(batch)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)