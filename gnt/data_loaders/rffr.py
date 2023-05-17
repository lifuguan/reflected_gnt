import os
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms as T
from .colmap_read_model import \
    read_cameras_binary, read_images_binary, read_points3d_binary

import torch.utils.data as data
from abc import ABC, abstractmethod

class Configurable(object):
    @staticmethod
    def modify_commandline_options(parser):
        """Add new options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser

        Returns:
            the modified parser.
        """
        return parser

class BaseDataset(data.Dataset, ABC, Configurable):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt, mode):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.mode = mode

    @abstractmethod
    def __len__(self):
        """Return the total number of data points in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg

 
def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, pose_avg


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4*np.pi, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))
        
        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0]) # (3)
        x = normalize(np.cross(y_, z)) # (3)
        y = np.cross(z, x) # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)] # (3, 4)

    return np.stack(poses_spiral, 0) # (n_poses, 3, 4)


class RFFRDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--num_masks', type=int, default=-1, help="Number of gt masks used for training, -1 for using all available masks.")
        parser.set_defaults(white_bkgd=False, noise_std=1.)
        return parser

    def __init__(self, opt, mode, scenes=()):
        mode = 'val'
        self.opt = opt
        self.mode = mode
        self.root_dir = os.path.join(opt.dataset_root, scenes[0])
        self.split = mode
        assert self.split in ['train', 'val', 'test', 'test_train', 'test_val']
        self.img_wh = opt.img_wh
        self.val_num = 1
        self.patch_size = opt.patch_size
        self.white_back = opt.white_bkgd
        
        self.define_transforms()
        self.read_meta()

    def read_meta(self):
        # Step 1: rescale focal length according to training resolution
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        H = camdata[1].height
        W = camdata[1].width
        self.focal = camdata[1].params[0] * self.img_wh[0]/W

        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        perm = np.argsort([imdata[k].name for k in imdata])
        # read successfully reconstructed images and ignore others
        self.image_names = sorted([imdata[k].name for k in imdata])
        self.image_paths = [os.path.join(self.root_dir, 'images', name)
                            for name in self.image_names]
        self.mask_paths = [os.path.join(self.root_dir, 'refl_masks', name.split('.')[0] + '.png')
                            for name in self.image_names]
        with open(os.path.join(self.root_dir, 'train.txt')) as f:
            self.train_image_names = list(filter(None, f.read().split('\n')))
        with open(os.path.join(self.root_dir, 'val.txt')) as f:
            self.val_image_names = list(filter(None, f.read().split('\n')))
        self.train_idxs = [self.image_names.index(name) for name in self.train_image_names]
        self.val_idxs = [self.image_names.index(name) for name in self.val_image_names]

        w2c_mats = []
        bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4) cam2world matrices
        
        # read bounds
        self.bounds = np.zeros((len(poses), 2)) # (N_images, 2)
        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        pts_world = np.zeros((1, 3, len(pts3d))) # (1, 3, N_points)
        visibilities = np.zeros((len(poses), len(pts3d))) # (N_images, N_points)
        for i, k in enumerate(pts3d):
            pts_world[0, :, i] = pts3d[k].xyz
            for j in pts3d[k].image_ids:
                visibilities[j-1, i] = 1
        # calculate each point's depth w.r.t. each camera
        # it's the dot product of "points - camera center" and "camera frontal axis"
        depths = ((pts_world-poses[..., 3:4])*poses[..., 2:3]).sum(1) # (N_images, N_points)
        for i in range(len(poses)):
            visibility_i = visibilities[i]
            zs = depths[i][visibility_i==1]
            self.bounds[i] = [np.percentile(zs, 0.1), np.percentile(zs, 99.9)]
        # permute the matrices to increasing order
        poses = poses[perm]
        self.bounds = self.bounds[perm]
        
        # COLMAP poses has rotation in form "right down front", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)
        self.poses, _ = center_poses(poses)

        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = [i for i in np.argsort(distances_from_center) if i in self.val_idxs][0] # choose val image as the closest to center image

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        # TODO: change this hard-coded factor
        # See https://github.com/kwea123/nerf_pl/issues/50
        near_original = self.bounds.min()
        scale_factor = near_original*0.75 # 0.75 is the default parameter
                                          # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal, self.opt.use_pixel_centers) # (H, W, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
            # split data to patches instead of individual pixels
            self.n_img_patches = (self.img_wh[0] - self.patch_size + 1) * (self.img_wh[1] - self.patch_size + 1)
            self.n_patches = self.n_img_patches * (len(self.train_idxs) - 1)
            self.all_rays = []
            self.all_rgbs = []
            self.all_masks = []
            self.all_masks_valid = []
            count = 0            
            for i, image_path in enumerate(self.image_paths):
                if i not in self.train_idxs: # exclude the val images
                    continue
                c2w = torch.FloatTensor(self.poses[i])

                img = Image.open(image_path).convert('RGB')
                assert img.size[1]*self.img_wh[0] == img.size[0]*self.img_wh[1], \
                    f'''{image_path} has different aspect ratio than img_wh, 
                        please check your data!'''
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                self.all_rgbs += [img]

                # load gt masks if exist
                if os.path.exists(self.mask_paths[i]):
                    mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_UNCHANGED)[:,:,3]
                    mask = cv2.resize(mask, self.img_wh, interpolation=cv2.INTER_NEAREST)
                    mask = mask.astype(np.float32)
                    mask[mask > 0] = 1
                    mask = torch.from_numpy(mask).view(-1, 1)
                    self.all_masks += [mask]                
                
                    if self.opt.num_masks == -1 or count < self.opt.num_masks:
                        self.all_masks_valid += [torch.ones_like(mask)]
                    else:
                        self.all_masks_valid += [torch.zeros_like(mask)]

                    count += 1
                else:
                    mask = torch.zeros_like(img)[:,[0]]
                    self.all_masks += [mask]
                    self.all_masks_valid += [torch.zeros_like(mask)]

                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                self.focal, 1.0, rays_o, rays_d)
                                    # near plane is always at 1.0
                                    # near and far in NDC are always 0 and 1
                                    # See https://github.com/bmild/nerf/issues/34

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             near*torch.ones_like(rays_o[:, :1]),
                                             far*torch.ones_like(rays_o[:, :1]),
                                             rays_d],
                                             1)] # (h*w, 11)
                                 
            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
            self.all_masks = torch.cat(self.all_masks, 0)
            self.all_masks_valid = torch.cat(self.all_masks_valid, 0)          
        
        elif self.split == 'val':
            print('val image is', self.image_paths[val_idx])
            self.val_idx = val_idx

        else: # for testing, create a parametric rendering path
            if self.split.endswith('train') or self.split.endswith('val'): # test on training set
                self.poses_test = self.poses
            else:
                focus_depth = 3.5 # hardcoded, this is numerically close to the formula
                                  # given in the original repo. Mathematically if near=1
                                  # and far=infinity, then this number will converge to 4
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                self.poses_test = create_spiral_poses(radii, focus_depth)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays) // self.patch_size**2
        if self.split == 'val':
            return self.val_num
        if self.split == 'test_train':
            return len(self.train_idxs)
        if self.split == 'test_val':
            return len(self.val_idxs)
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            if self.patch_size == 1:
                sample = {'rays': self.all_rays[idx],
                        'rgbs': self.all_rgbs[idx],
                        'masks': self.all_masks[idx],
                        'masks_valid': self.all_masks_valid[idx]}
            else:
                i_patch = torch.randint(high=self.n_patches, size=(1,))[0].item()
                total_step, i_patch = i_patch // self.n_img_patches, i_patch % self.n_img_patches
                row, col = i_patch // (self.img_wh[0] - self.patch_size + 1), i_patch % (self.img_wh[0] - self.patch_size + 1)
                start_idx = total_step * self.img_wh[0] * self.img_wh[1] + row * self.img_wh[0] + col
                idxs = start_idx + torch.cat([torch.arange(self.patch_size) + i * self.img_wh[0] for i in range(self.patch_size)])
                sample = {
                    'rays': self.all_rays[idxs],
                    'rgbs': self.all_rgbs[idxs],
                    'masks': self.all_masks[idxs],
                    'masks_valid': self.all_masks_valid[idxs]
                }
        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.poses[self.val_idx])
            elif self.split == 'test_train':
                c2w = torch.FloatTensor(self.poses[self.train_idxs[idx]])
            elif self.split == 'test_val':
                c2w = torch.FloatTensor(self.poses[self.val_idxs[idx]])
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])

            rays_o, rays_d = get_rays(self.directions, c2w)
            
            viewdir = rays_d

            near, far = 0, 1
            rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0], self.focal, 1.0, rays_o, rays_d)
            viewdir = rays_d

            rays = torch.cat([rays_o, rays_d, 
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1]),
                              viewdir],
                              1) # (h*w, 11)

            sample = {'rays': rays,
                      'c2w': c2w}

            if self.split in ['val', 'test_train', 'test_val']:
                if self.split == 'val':
                    idx = self.val_idx
                if self.split == 'test_train':
                    idx = self.train_idxs[idx]
                if self.split == 'test_val':
                    idx = self.val_idxs[idx]
                img = Image.open(self.image_paths[idx]).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3)
                sample['rgbs'] = img

                if os.path.exists(self.mask_paths[idx]):
                    mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)[:,:,3]
                    mask = cv2.resize(mask, self.img_wh, interpolation=cv2.INTER_NEAREST)
                    mask = mask.astype(np.float32)
                    mask[mask > 0] = 1
                    mask = torch.from_numpy(mask).view(-1, 1)
                    sample['masks'] = mask
                    sample['masks_valid'] = torch.ones_like(mask)       
                else:
                    mask = torch.zeros_like(img)[:,[0]]
                    sample['masks'] = mask
                    sample['masks_valid'] = torch.zeros_like(mask)

        return sample

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[...,0] / rays_o[...,2]
    oy_oz = rays_o[...,1] / rays_o[...,2]
    
    # Projection
    o0 = -1./(W/(2.*focal)) * ox_oz
    o1 = -1./(H/(2.*focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
    d2 = 1 - o2
    
    rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
    
    return rays_o, rays_d


def get_ray_directions(H, W, focal, use_pixel_centers=True):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length
        use_pixel_corners:
            If True, generate rays through the center of each pixel. Note: While
            this is the correct way to handle rays, it is not the way rays are
            handled in the original NeRF paper. Setting this TRUE yields ~ +1 PSNR
            compared to Vanilla NeRF.
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions