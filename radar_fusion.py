import sys
import shutil
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix, transform_pcl
from vod.visualization import get_radar_velocity_vectors
import numpy as np
from utils import read_kitti_label_file, if_points_in_moving_objects
import os
from tqdm import tqdm

bad_frames = [1312, 1313, 1314, 1315, 1316, 1317, 544, 545, 546, 547, 548, 4652, 4653, 4654, 4655, 4656, 8481, 8482, 8483, 8484, 8485, 9518, 9519, 9520, 9521, 3277, 3278, 3279, 3280, 6759, 6760 ,6761, 6762, 8198, 8199, 8200, 8201, 4049, 4050, 4051,4052]

"""
编写自己的transform_pcl, 保留雷达点的其他属性
"""

def single_frame_compensation(frame: int, frame_diff: int, kitti_loc):
    """
    对单帧中的运动点进行补偿。返回补偿后的雷达点云，world frame
    frame_diff: 离要融合的帧差几帧，一帧0.1s 10Hz
    返回世界坐标系下的雷达点云，N x 7
    """
    frame_data = FrameDataLoader(kitti_locations=kitti_loc, frame_number=str(frame).zfill(5))
    frame_transform = FrameTransformMatrix(frame_data)
    cf_radar_pc = transform_pcl(frame_data.radar_data, frame_transform.t_camera_radar)[:, :3]  # N x 3
    # radar 点的其他4个属性 RCS, v_r, v_r_compensated, time
    rest_attributes = frame_data.radar_data[:, 3:]
    compensated_radial_velo = frame_data.radar_data[:, 5]

    moving_points_velo = [] # 运动点的radial velocity
    moving_points_rest4 = [] # 点的其他4个属性
    static_points_rest4 = []

    _, moving_objs = read_kitti_label_file(frame_data.raw_labels)
    moving_points, static_points = [], []
    for idx, point in enumerate(cf_radar_pc):
        res, _ = if_points_in_moving_objects(point, moving_objs, frame_transform.t_lidar_camera, frame_transform.t_camera_lidar)
        if res:
            moving_points.append(point)
            moving_points_velo.append(compensated_radial_velo[idx])
            moving_points_rest4.append(rest_attributes[idx])
        else:
            static_points.append(point)
            static_points_rest4.append(rest_attributes[idx])

    # 如果没有运动点，直接返回世界坐标系下的原始雷达点云
    if len(moving_points) == 0:
        print(f'No moving points in frame {str(frame).zfill(5)}.')
        wf_radar_pc = transform_pcl(cf_radar_pc, frame_transform.t_map_camera)[:,:3]
        wf_radar_pc = np.hstack([wf_radar_pc, rest_attributes])
        return wf_radar_pc
    else:
        cf_moving_points = np.array(moving_points) # N x 3
        cf_static_points = np.array(static_points)
        moving_points_velo = np.array(moving_points_velo) # N x 1
        # 补偿动态点，加上radial velocity向量
        moving_points_compensated = cf_moving_points + frame_diff * 0.1 * get_radar_velocity_vectors(cf_moving_points, moving_points_velo)
        cf_radar_pc_after_comp = np.vstack([cf_static_points, moving_points_compensated])
        wf_radar_pc_after_comp = transform_pcl(cf_radar_pc_after_comp, frame_transform.t_map_camera)[:, :3]
        full_points_rest4 = np.vstack([static_points_rest4, moving_points_rest4])  # N x 4
        wf_radar_pc_after_comp = np.hstack([wf_radar_pc_after_comp, full_points_rest4])
        return wf_radar_pc_after_comp

# 问题: bin文件里存储的点位于雷达坐标系下！！！
def fuse_3_frames(frame3:int, kitti_loc = KittiLocations(root_dir='/datasets/vod')):
    print(f'Fusing frame {str(frame3-2).zfill(5)}, frame {str(frame3-1).zfill(5)} and frame {str(frame3).zfill(5)} ...') if frame3 >= 3 else sys.exit('Frame number should be larger than 3.')
    frame3_data = FrameDataLoader(kitti_locations=kitti_loc, frame_number=str(frame3).zfill(5))
    frame3_transform = FrameTransformMatrix(frame3_data)

    rest_4_attributes = frame3_data.radar_data[:, 3:]
    cf3_pc = transform_pcl(frame3_data.radar_data, frame3_transform.t_camera_radar)[:,:3]
    wf3_pc = transform_pcl(cf3_pc, frame3_transform.t_map_camera)[:,:3]
    wf3_pc = np.hstack([wf3_pc, rest_4_attributes]) # N x 7
    wf2_pc_compensated = single_frame_compensation(frame3 - 1, 1, kitti_loc)
    wf1_pc_compensated = single_frame_compensation(frame3 - 2, 2, kitti_loc)

    fused_wf_pc = np.vstack([wf1_pc_compensated, wf2_pc_compensated, wf3_pc])
    rest_attributes = fused_wf_pc[:, 3:]
    fused_cf_pc = transform_pcl(fused_wf_pc, frame3_transform.t_camera_map)[:,:3]
    fused_rf_pc = transform_pcl(fused_cf_pc, frame3_transform.t_radar_camera)[:,:3]
    complete_fused_rf_pc = np.hstack([fused_rf_pc, rest_attributes])
    return complete_fused_rf_pc # N x 7

def fuse_5_frames(frame5: int, kitti_loc = KittiLocations(root_dir='/datasets/vod')):
    frame5_data = FrameDataLoader(kitti_locations=kitti_loc, frame_number=str(frame5).zfill(5))
    frame5_transform = FrameTransformMatrix(frame5_data)

    rest_4_attributes = frame5_data.radar_data[:, 3:]
    cf5_pc = transform_pcl(frame5_data.radar_data, frame5_transform.t_camera_radar)[:,:3]
    wf5_pc = transform_pcl(cf5_pc, frame5_transform.t_map_camera)[:, :3]
    wf5_pc = np.hstack([wf5_pc, rest_4_attributes])
    wf4_pc_compensated = single_frame_compensation(frame5 - 1, 1, kitti_loc)
    wf3_pc_compensated = single_frame_compensation(frame5 - 2, 2, kitti_loc)
    wf2_pc_compensated = single_frame_compensation(frame5 - 3, 3, kitti_loc)
    wf1_pc_compensated = single_frame_compensation(frame5 - 4, 4, kitti_loc)

    fused_wf_pc = np.vstack([wf1_pc_compensated, wf2_pc_compensated, wf3_pc_compensated, wf4_pc_compensated, wf5_pc])
    rest_attributes = fused_wf_pc[:, 3:]
    fused_cf_pc = transform_pcl(fused_wf_pc, frame5_transform.t_camera_map)[:,:3]
    fused_rf_pc = transform_pcl(fused_cf_pc, frame5_transform.t_radar_camera)[:,:3]
    complete_rf_pc = np.hstack([fused_rf_pc, rest_attributes])
    return complete_rf_pc # N x 7

def fuse_and_save(label_dir, target_dir, frames=15):
    label_files = os.listdir(label_dir)
    print(f'Total {len(label_files)} label files.')
    count = 0
    for bin_file in tqdm(label_files, desc='Fusing and saving..'):
        frame_num = int(bin_file[:-4])
        if frame_num >= frames and frame_num not in bad_frames:
            data = fuse_5_frames(frame_num)
            data.astype(np.float32).tofile(f'{target_dir}/{bin_file[:-4]}.bin')
            count += 1
    print(f'{count} frames fused and saved.')

    # 将自叠加文件夹中没有的bin文件从原始叠加中复制过来
    my_bins = os.listdir(target_dir)
    vod_bins = os.listdir('/datasets/vod/radar_5frames/training/velodyne')
    print(f'Difference: {len(vod_bins) - len(my_bins)}')
    for bin_file in vod_bins:
        if bin_file not in my_bins:
            print(f'Copying {bin_file}...')
            shutil.copy2(f'/datasets/vod/radar_5frames/training/velodyne/{bin_file}', target_dir)

def replace_different_size_files():
    f1 = os.listdir('/datasets/vod/radar_5frames/training/velodyne')
    f2 = os.listdir('/datasets/vod/my_radar_5frames/training/velodyne')
    print(f1)
    print(f2)
    for i in range(len(f1)):
        if f1[i] != f2[i]:
            print('Different files')
            break
        if os.path.getsize(f'/datasets/vod/radar_5frames/training/velodyne/{f1[i]}') != os.path.getsize(f'/datasets/vod/my_radar_5frames/training/velodyne/{f2[i]}'):
            # replace f2[i] with f1[i]
            print(f'Replacing {f2[i]} with {f1[i]}...')
            shutil.copy2(f'/datasets/vod/radar_5frames/training/velodyne/{f1[i]}', '/datasets/vod/my_radar_5frames/training/velodyne')

if __name__ == '__main__':
    # source = '/datasets/vod/lidar/training/label_2'
    # target = '/datasets/vod/my_radar_5frames/training/velodyne'
    # fuse_and_save(source, target)

    replace_different_size_files()
