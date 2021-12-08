import numpy as np
import glob
from skimage.measure import marching_cubes_lewiner
import open3d as o3d
import cv2
import time

depth_dir_name = "CS284_hw5_data/depth/*"
pose_dir_name = "CS284_hw5_data/pose/*"

K = np.array([[259.2, 0, 160],
              [0, 259.2, 120],
              [0, 0, 1]])

K_inv= np.linalg.inv(K)

scale = 1.3476e5

voxel_size = 0.01
map_size = 0.3

class TSDFMap:
    def __init__(self, voxel_size, map_size):
        self.resolution = int(map_size / voxel_size)
        self.map_size = map_size
        self.origin = -np.array([map_size/2, map_size/2, map_size/2]).reshape(3, -1)
        self.voxel_size = voxel_size
        self.trunc_margin = 1 * voxel_size

        self.map = np.full((self.resolution, self.resolution, self.resolution), 1)
        self.weight = np.zeros((self.resolution, self.resolution, self.resolution))

    def voxel2point(self, voxels):
        points = (voxels + 0.5) * self.voxel_size + self.origin
        return points

    def updateMap(self, depth_image, pose):

        points = np.zeros((0,3))
        for voxel_x in range(self.resolution):
            for voxel_y in range(self.resolution):
                for voxel_z in range(self.resolution):
                    voxel = np.array([voxel_x, voxel_y, voxel_z]).reshape(3, 1)
                    point_world = self.voxel2point(voxel)
                    uv, point_cam = self.projectPoint(point_world, pose)
                    uv = uv.astype("int")

                    if (uv[0] >= depth_image.shape[1] or uv[1] >= depth_image.shape[0] or uv[0] < 0 or uv[1] < 0):
                        continue

                    points = np.concatenate([points, np.append(uv, 1).reshape(1,3)])

                    lamd = np.linalg.norm(K_inv @ np.append(uv, 1).reshape(3,1))
                    sdf = np.linalg.norm(point_cam) / lamd - depth_image[uv[1], uv[0]] / scale

                    if (sdf >= -self.trunc_margin):
                        tsdf = min(1, sdf / self.trunc_margin) * np.sign(sdf)
                    else:
                        continue

                    old_weight = self.weight[voxel_x, voxel_y, voxel_z]
                    old_tsdf = self.map[voxel_x, voxel_y, voxel_z]

                    self.weight[voxel_x, voxel_y, voxel_z] += 1
                    self.map[voxel_x, voxel_y, voxel_z] = (old_weight * old_tsdf + tsdf) / (old_weight + 1)

    def projectPoint(self, pt_world, pose):
        pts_cam = pose[:3, :3] @ pt_world + pose[:3, 3].reshape(3, 1)
        pts_normalized = pts_cam / pts_cam[2, :].reshape(1, -1)
        uv = K @ pts_normalized
        return uv[:2, :], pts_cam

    def visualize(self):
        verts, faces, normals, values = marching_cubes_lewiner(self.map)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

def get_camera_pt(depth_im, color_im, cam_intr):
    depth_im = depth_im / scale
    valid_pts = np.logical_and(depth_im > 0, depth_im < 5)
    pix_x,  pix_y = np.where(valid_pts)
    pix_z = depth_im[pix_x, pix_y]
    # color = color_im[pix_x, pix_y, :]/255.0
    pts = np.concatenate([pix_x.reshape(1, -1), pix_y.reshape(1,-1), np.ones((1, pix_x.shape[0]))])
    pts = K_inv @ pts * pix_z

    pix_y = (pix_y - cam_intr[0, 2]) * pix_z / cam_intr[0, 0]
    pix_x = (pix_x - cam_intr[1, 2]) * pix_z / cam_intr[1, 1]
    pixyz = np.concatenate((np.expand_dims(pix_y, axis=1), np.expand_dims(pix_x, axis=1), np.expand_dims(pix_z, axis=1)), axis=1)

    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(pixyz)
    # o3d.visualization.draw_geometries([cloud])
    return pixyz

def read_data():
    depth_dir = glob.glob(depth_dir_name)
    pose_dir = glob.glob(pose_dir_name)
    depth_dir.sort()
    pose_dir.sort()

    depth_list = []
    pose_list = []
    for i in range(len(depth_dir)):
        depth_img = cv2.imread(depth_dir[i], flags=-1)
        pose = np.loadtxt(pose_dir[i], delimiter=",")
        depth_list.append(depth_img)
        pose_list.append(pose)

    return depth_list, pose_list

def main():
    depth_list, pose_list = read_data()

    pts = np.empty((0, 3))
    for i in range(len(depth_list)):
        pt = get_camera_pt(depth_list[i], depth_list[i], K).T
        pt = np.concatenate([pt, np.ones((1, pt.shape[1]))])
        pt = np.linalg.inv(pose_list[i]) @ pt
        pts = np.concatenate([pts, pt[:3, :].T])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw_geometries([cloud])


    map = TSDFMap(voxel_size, map_size)
    time_start = time.time()
    for i in range(len(depth_list)):
        print(i, "/", len(depth_list))
        map.updateMap(depth_list[i], pose_list[i])
    time_end = time.time()

    print("total time:", time_end - time_start)
    print("average time:", (time_end - time_start) / 14)
    map.visualize()

if __name__ == "__main__":
    main()
