#!/usr/bin/env python3

import open3d as o3d
import numpy as np
from copy import deepcopy
from matplotlib import cm
from more_itertools import locate
import math
from point_cloud_processing import PointCloudProcessing

view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 3.0000000000000004, 3.0000000000000004, 3.83980393409729 ],
			"boundingbox_min" : [ -2.5246021747589111, -1.5300980806350708, -1.4928504228591919 ],
			"field_of_view" : 60.0,
			"front" : [ -0.71789151197863932, -0.38186310500939796, -0.58207589373002278 ],
			"lookat" : [ 0.87689053765100455, 0.27227025206511868, 1.5938394003482237 ],
			"up" : [ 0.45631767202665485, -0.88957373275513962, 0.020802792799569481 ],
			"zoom" : 0.66120000000000023
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

class Plane_detection():
    def __init__(self, point_cloud):
        self.point_cloud = point_cloud

    def colorizeInliers (self, r , g, b):
        self.inlier_cloud.paint_uniform_color([r,g,b]) # paints the plane in red

    def segment(self, distance_threshold=0.25, ransac_n=3, num_iterations=100):
        print('Starting plane detection')
        plane_model, inlier_idxs = self.point_cloud.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=ransac_n,
                                            num_iterations=num_iterations)
        [self.a, self.b, self.c, self.d] = plane_model

        self.inlier_cloud = self.point_cloud.select_by_index(inlier_idxs)
        outlier_cloud = self.point_cloud.select_by_index(inlier_idxs, invert=True)
        
        return outlier_cloud

    def __str__(self):
        text = 'Segmented plane from pc with ' + str(len(self.point_cloud.points)) + ' with ' + str(len(self.inlier_cloud.points)) + ' inliers. '
        text += '\nPlane: ' + str(self.a) +  ' x + ' + str(self.b) + ' y + ' + str(self.c) + ' z + ' + str(self.d) + ' = 0' 
        return text

def draw_registration_result(source, target, transformation):
    source_temp = deepcopy(source)
    target_temp = deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def main():
    
    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    p = PointCloudProcessing()
    p.loadPointCLoud('./scene.ply')
    
    #------------------------------------------
    # Execution
    # ------------------------------------------
    p.preProcess()
    
    p.transform(-108,8,0,0,0,0)
    p.transform(0,0,-37,0,0,0)
    p.transform(0,0,0,-0.85,-1.10,0.35)

    p.crop(-0.9,-0.9,-0.3,0.9,0.9,0.4)
   
    outliers = p.findPlane()

    cluster_idxs = list(outliers.cluster_dbscan(eps=0.3, min_points=60, print_progress=True))

    object_idxs = list(set(cluster_idxs))
    object_idxs.remove(-1)

    number_of_objects = len(object_idxs)
    colormap = cm.Pastel1(list(range(0,number_of_objects)))
    objects = []
    for object_idx in object_idxs:
        object_point_idxs = list(locate(cluster_idxs, lambda x:x == object_idx))

        object_points = outliers.select_by_index(object_point_idxs)
        # Create dictionary
        d =  {}
        d['idx'] = str(object_idx)
        d['points'] = object_points
        d['color'] = colormap[object_idx, 0:3]
        d['points'].paint_uniform_color(d['color'])
        objects.append(d)


    cereal_box_model = o3d.io.read_point_cloud('./cereal_box_2_2_40.pcd')
    for object_idx, object in enumerate(objects):
        print("Apply point-to-point ICP")
        
        trans_init = np.asarray([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0], 
                         [0.0, 0.0, 0.0, 1.0]])

        reg_p2p = o3d.pipelines.registration.registration_icp(cereal_box_model, 
                                                            object['points'], 2, trans_init,
                                                            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        print(reg_p2p.inlier_rmse)
        #object['']       
        #draw_registration_result(cereal_box_model, object['points'], reg_p2p.transformation)

    # ------------------------------------------
    # Visualization
    # ------------------------------------------

    # Create a list of entities to draw
    p.inliers.paint_uniform_color([0,1,1])

    entities = []

    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
    entities.append(frame)


    bbox_to_drawn = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(p.bbox)
    entities.append(bbox_to_drawn)

    for object in objects:
        entities.append(object['points'])

    
    o3d.visualization.draw_geometries(entities,
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'],
                                    point_show_normal = False)


if __name__ == "__main__":
    main()