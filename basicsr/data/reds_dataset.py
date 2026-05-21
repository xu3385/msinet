# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.flow_util import dequantize_flow


class REDSDataset(data.Dataset):
    """REDS dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, seperated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_flow (str, optional): Data root path for flow.
            meta_info_file (str): Path for meta information file.
            val_partition (str): Validation partition types. 'REDS4' or
                'official'.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(REDSDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(
            opt['dataroot_lq'])
        self.flow_root = Path(
            opt['dataroot_flow']) if opt['dataroot_flow'] is not None else None
        assert opt['num_frame'] % 2 == 1, (
            f'num_frame should be odd number, but got {opt["num_frame"]}')
        self.num_frame = opt['num_frame']
        self.num_half_frames = opt['num_frame'] // 2

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend(
                    [f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # remove the video clips used in validation
        if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(
                f'Wrong validation partition {opt["val_partition"]}.'
                f"Supported ones are ['official', 'REDS4'].")
        self.keys = [
            v for v in self.keys if v.split('/')[0] not in val_partition
        ]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [
                    self.lq_root, self.gt_root, self.flow_root
                ]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        interval_str = ','.join(str(x) for x in opt['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000
        center_frame_idx = int(frame_name)

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - self.num_half_frames * interval
        end_frame_idx = center_frame_idx + self.num_half_frames * interval
        # each clip has 100 frames starting from 0 to 99
        while (start_frame_idx < 0) or (end_frame_idx > 99):
            center_frame_idx = random.randint(0, 99)
            start_frame_idx = (
                center_frame_idx - self.num_half_frames * interval)
            end_frame_idx = center_frame_idx + self.num_half_frames * interval
        frame_name = f'{center_frame_idx:08d}'
        neighbor_list = list(
            range(center_frame_idx - self.num_half_frames * interval,
                  center_frame_idx + self.num_half_frames * interval + 1,
                  interval))
        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        assert len(neighbor_list) == self.num_frame, (
            f'Wrong length of neighbor list: {len(neighbor_list)}')

        # get the GT frame (as the center frame)
        if self.is_lmdb:
            img_gt_path = f'{clip_name}/{frame_name}'
        else:
            img_gt_path = self.gt_root / clip_name / f'{frame_name}.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        # get flows
        if self.flow_root is not None:
            img_flows = []
            # read previous flows
            for i in range(self.num_half_frames, 0, -1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_p{i}'
                else:
                    flow_path = (
                        self.flow_root / clip_name / f'{frame_name}_p{i}.png')
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(
                    img_bytes, flag='grayscale',
                    float32=False)  # uint8, [0, 255]
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(
                    dx, dy, max_val=20,
                    denorm=False)  # we use max_val 20 here.
                img_flows.append(flow)
            # read next flows
            for i in range(1, self.num_half_frames + 1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_n{i}'
                else:
                    flow_path = (
                        self.flow_root / clip_name / f'{frame_name}_n{i}.png')
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(
                    img_bytes, flag='grayscale',
                    float32=False)  # uint8, [0, 255]
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(
                    dx, dy, max_val=20,
                    denorm=False)  # we use max_val 20 here.
                img_flows.append(flow)

            # for random crop, here, img_flows and img_lqs have the same
            # spatial size
            img_lqs.extend(img_flows)

        # randomly crop
        img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, gt_size, scale,
                                             img_gt_path)
        if self.flow_root is not None:
            img_lqs, img_flows = img_lqs[:self.num_frame], img_lqs[self.
                                                                   num_frame:]

        # augmentation - flip, rotate
        img_lqs.append(img_gt)
        if self.flow_root is not None:
            img_results, img_flows = augment(img_lqs, self.opt['use_flip'],
                                             self.opt['use_rot'], img_flows)
        else:
            img_results = augment(img_lqs, self.opt['use_flip'],
                                  self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        if self.flow_root is not None:
            img_flows = img2tensor(img_flows)
            # add the zero center flow
            img_flows.insert(self.num_half_frames,
                             torch.zeros_like(img_flows[0]))
            img_flows = torch.stack(img_flows, dim=0)

        # img_lqs: (t, c, h, w)
        # img_flows: (t, 2, h, w)
        # img_gt: (c, h, w)
        # key: str
        if self.flow_root is not None:
            return {'lq': img_lqs, 'flow': img_flows, 'gt': img_gt, 'key': key}
        else:
            return {'lq': img_lqs, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)

class REDSStereoImageDataset(data.Dataset):
    """REDS dataset for Stereo Image Super-Resolution training.
    
    Loads a single pair of stereo images (Left/Right) at a time.
    """

    def __init__(self, opt):
        super(REDSStereoImageDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                # 将该视频片段下的所有帧都加入索引列表
                # 配合 DataLoader 的 shuffle=True，即可实现“随机片段、随机帧”的效果
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # 移除验证集片段
        if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.')
        
        if opt['test_mode']:
            self.keys = [v for v in self.keys if v.split('/')[0] in val_partition]
        else:
            self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
        elif self.io_backend_opt['type'] == 'sharedict':
            self.is_sharedict = True
            self.io_backend_opt['imgdirs'] = [self.lq_root, self.gt_root]
        self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

    def __getitem__(self, index):
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # --- 加载图像 (Load Images) ---
        # 这里的逻辑从“循环加载邻帧”改为了“只加载当前帧”
        
        # 1. 构建路径
        if self.is_lmdb:
            img_lq_path = f'{clip_name}/{frame_name}'
            img_gt_path = f'{clip_name}/{frame_name}'
        else:
            # REDS Stereo 命名规则: frame_0.png (左), frame_1.png (右)
            img_lq_path_l = self.lq_root / clip_name / f'{frame_name}_0.png'
            img_lq_path_r = self.lq_root / clip_name / f'{frame_name}_1.png'
            img_gt_path_l = self.gt_root / clip_name / f'{frame_name}_0.png'
            img_gt_path_r = self.gt_root / clip_name / f'{frame_name}_1.png'
            # 这里的 path 变量仅用于 crop 时的记录，随便用一个即可
            img_gt_path = str(img_gt_path_l) 

        # 2. 读取 LQ (Left + Right)
        if self.is_lmdb:
            # LMDB 读取逻辑需根据实际 key 格式调整，这里保留原结构
            # 假设 LMDB key 已经是分开存的或者你需要在这里做处理
            pass 
        else:
            # Load LQ Left
            img_bytes = self.file_client.get(img_lq_path_l, 'lq')
            img_lq_l = imfrombytes(img_bytes, float32=True) if self.io_backend_opt.get('store_undecoded', True) else img_bytes.astype(np.float32) / 255.
            
            # Load LQ Right
            img_bytes = self.file_client.get(img_lq_path_r, 'lq')
            img_lq_r = imfrombytes(img_bytes, float32=True) if self.io_backend_opt.get('store_undecoded', True) else img_bytes.astype(np.float32) / 255.

            # Concat LQ -> (H, W, 6)
            img_lq = np.concatenate([img_lq_l, img_lq_r], axis=-1)

            # Load GT Left
            img_bytes = self.file_client.get(img_gt_path_l, 'gt')
            img_gt_l = imfrombytes(img_bytes, float32=True) if self.io_backend_opt.get('store_undecoded', True) else img_bytes / 255.

            # Load GT Right
            img_bytes = self.file_client.get(img_gt_path_r, 'gt')
            img_gt_r = imfrombytes(img_bytes, float32=True) if self.io_backend_opt.get('store_undecoded', True) else img_bytes / 255.

            # Concat GT -> (H, W, 6)
            img_gt = np.concatenate([img_gt_l, img_gt_r], axis=-1)

        # --- 裁剪与增强 (Crop & Augment) ---
        
        # 1. 确保输入 paired_random_crop 的是列表形式 [img_gt], [img_lq]
        #    这样通常能保证返回的也是列表
        img_gts, img_lqs = paired_random_crop([img_gt], [img_lq], gt_size, scale, img_gt_path)

        # 2. 安全检查：如果返回的不是列表（是 numpy array），强制转为列表
        #    这步是为了防止 'numpy.ndarray' object has no attribute 'extend' 报错
        if not isinstance(img_gts, list):
            img_gts = [img_gts]
        if not isinstance(img_lqs, list):
            img_lqs = [img_lqs]

        # 3. 合并列表：将 GT 和 LQ 放入同一个列表进行增强
        #    注意：这里不再修改 img_lqs 本身，而是创建一个新列表 img_full_list
        img_full_list = img_lqs + img_gts
        
        # 4. 数据增强 (Flip / Rotate)
        img_results = augment(img_full_list, self.opt['use_hflip'], self.opt['use_rot'])

        # 5. 转为 Tensor
        img_results = img2tensor(img_results)
        
        # 6. 拆分回 LQ 和 GT
        #    img_results 列表的前半部分是 LQ，后半部分是 GT
        #    因为我们是单帧训练，split_idx 应该为 1
        split_idx = len(img_lqs) 
        img_lqs_ret = torch.stack(img_results[:split_idx], dim=0)
        img_gts_ret = torch.stack(img_results[split_idx:], dim=0)

        # 7. 去除 batch 维度 (1, C, H, W) -> (C, H, W)
        img_lq = img_lqs_ret.squeeze(0)
        img_gt = img_gts_ret.squeeze(0)

        return {'lq': img_lq, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)