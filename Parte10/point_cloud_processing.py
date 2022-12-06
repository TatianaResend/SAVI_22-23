#!/usr/bin/env python3

import open3d as o3d
import numpy as np
from copy import deepcopy
from matplotlib import cm
from more_itertools import locate
import math


class PointCloudProcessing():
    def __init__(self):
        pass

    def loadPointCLoud(self,filename):
        print("load")
        self.pcd = o3d.io.read_point_cloud(filename)
        self.original = deepcopy(self.pcd)

    def preProcess(self):
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.02)
        print('After downsampling point cloud has ' + str(len(self.pcd.points)) + ' points')

        #estimate normals
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        self.pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))    
    
    def transform(self,r,p,y,tx,ty,tz):
        r = math.pi*r / 180.0
        p = math.pi*p / 180.0
        y = math.pi*y / 180.0

        rotation = self.pcd.get_rotation_matrix_from_xyz((r,p,y))
        self.pcd.rotate(rotation, center=(0,0,0))

        self.pcd= self.pcd.translate((tx,ty,tz))

    def crop(self,min_x,min_y,min_z,max_x,max_y,max_z):
        np_points = np.ndarray((8,3),dtype=float)
      
        np_points[0,:] = [min_x, min_y, min_z]
        np_points[1,:] = [max_x, min_y, min_z]
        np_points[2,:] = [max_x, max_y, min_z]
        np_points[3,:] = [min_x, max_y, min_z]

        np_points[4,:] = [min_x, min_y, max_z]
        np_points[5,:] = [max_x, min_y, max_z]
        np_points[6,:] = [max_x, max_y, max_z]
        np_points[7,:] = [min_x, max_y, max_z]

        bbox_points = o3d.utility.Vector3dVector(np_points)
        self.bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(bbox_points)

        self.pcd = self.pcd.crop(self.bbox)

        print(np_points)

    def findPlane(self, distance_threshold=0.02, ransac_n=3, num_iterations=100):
        print('Starting plane detection')
        plane_model, inlier_idxs = self.pcd.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=ransac_n,
                                            num_iterations=num_iterations)
        [self.a, self.b, self.c, self.d] = plane_model

        self.inliers = self.pcd.select_by_index(inlier_idxs)
        outlier_cloud = self.pcd.select_by_index(inlier_idxs, invert=True)
        
        return outlier_cloud
    
    
    def __str__(self):
        text = 'Segmented plane from pc with '
        text += 'não sei'
        return text


