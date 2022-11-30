#!/usr/bin/env python3

import open3d as o3d
import numpy as np


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    print("Load a ply point cloud, print it, and render it")
    ply_point_cloud = o3d.data.PLYPointCloud()
    point_cloud = o3d.io.read_point_cloud('data/factory.ply')
    print(point_cloud)
    print(np.asarray(point_cloud.points))
    o3d.visualization.draw_geometries([point_cloud],
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])

    # ------------------------------------------
    # Execution
    # ------------------------------------------


    # ------------------------------------------
    # Termination
    # ------------------------------------------
  
    

    
if __name__ == "__main__":
    main()