import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

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

    def rgbd2pc(self, color: torch.tensor, depth: torch.tensor, intrinsic, extrinsic):
        '''
        color: 3, h, w
        depth: h, w
        '''
        h, w = depth.shape
        i, j = np.meshgrid(np.arange(w), np.arange(h))
        z = depth * 1.
        valid_mask = z > 0  # Check for valid depth values

        x = (i - intrinsic[0, 2]) * z / intrinsic[0, 0]
        y = (j - intrinsic[1, 2]) * z / intrinsic[1, 1]
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        colors = color.reshape(3, -1).permute(1, 0) # npts, 3

        mask = valid_mask.reshape(-1) & (points[:, 2] <= 1)
        points = points[mask]
        colors = colors[mask]

        points_transformed = points
        
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
        
        # Process next frame
        next_camera_data = next_frame['camera_data'][0]
        next_color = next_camera_data['color'][..., ::-1]  # BGR to RGB
        next_depth = next_camera_data['depth'] * 0.00025  # mm to m
        next_intrinsics = next_camera_data['intrinsics']
        next_extrinsics = next_camera_data['extrinsics']
        
        # Convert numpy arrays to PyTorch tensors
        current_color = torch.from_numpy(current_color).permute(2, 0, 1).float() / 255.0
        current_depth = torch.from_numpy(current_depth).unsqueeze(0).float()
        next_color = torch.from_numpy(next_color).permute(2, 0, 1).float() / 255.0
        next_depth = torch.from_numpy(next_depth).unsqueeze(0).float()
        
        # Apply transforms if any
        if self.transform:
            current_color = self.transform(current_color)
            next_color = self.transform(next_color)
        
        return {
            'current_color': current_color,
            'current_depth': current_depth,
            'current_intrinsics': torch.tensor(current_intrinsics, dtype=torch.float),
            'current_extrinsics': torch.tensor(current_extrinsics, dtype=torch.float),
            'next_color': next_color,
            'next_depth': next_depth,
            'next_intrinsics': torch.tensor(next_intrinsics, dtype=torch.float),
            'next_extrinsics': torch.tensor(next_extrinsics, dtype=torch.float),
        }

# Usage example:
# dataset = RobotDataset(root_dir='path/to/your/data')
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)