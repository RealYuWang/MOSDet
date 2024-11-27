import numpy as np
from k3d import points
from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
import cumm.tensorview as tv

class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        self._voxel_generator = VoxelGenerator(
            vsize_xyz=vsize_xyz,
            coors_range_xyz=coors_range_xyz,
            num_point_features=num_point_features,
            max_num_points_per_voxel=max_num_points_per_voxel,
            max_num_voxels=max_num_voxels
        )

    def generate(self, points):
        voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
        tv_voxels, tv_coordinates, tv_num_points = voxel_output
        # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
        voxels = tv_voxels.numpy()
        coordinates = tv_coordinates.numpy()
        num_points = tv_num_points.numpy()

        return voxels, coordinates, num_points

if __name__ == '__main__':
    vsize_xyz = [0.16, 0.16, 5]
    coors_range_xyz = [0, -25.6, -3, 51.2, 25.6, 2]
    num_point_features = 7
    max_num_points_per_voxel = 10
    max_num_voxels = 16000
    voxel_generator = VoxelGeneratorWrapper(vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels)

    points = np.load('points.npy')
    voxels, coordinates, num_points = voxel_generator.generate(points)
    # print(voxels.shape)
    # print(coordinates.shape)
    print(coordinates)
    # print(num_points.shape)
