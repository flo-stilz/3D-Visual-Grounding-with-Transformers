import os
import argparse
import sys

import numpy as np
try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=str, help="scene id of scene to be visualized", default="scene0000_00")
    args = parser.parse_args()

    filename = "scans/{}/{}_vh_clean_2.ply".format(args.scene_id, args.scene_id)
    filename_output = "scans/{}/{}_vh_clean_2_aligned.ply".format(args.scene_id, args.scene_id)
    meta_file = os.path.join("scans/"+str(args.scene_id)+"/", args.scene_id + '.txt')
    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    axis_align_matrix = None
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]


    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        
        print(len(axis_align_matrix))
        print(plydata['vertex'].data['x'].shape)
        if axis_align_matrix != None:
            axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
            pts = np.ones((vertices.shape[0], 4))
            pts[:,0] = plydata['vertex'].data['x']
            pts[:,1] = plydata['vertex'].data['y']
            pts[:,2] = plydata['vertex'].data['z']
            pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
            aligned_vertices = np.copy(vertices)
            aligned_vertices[:,0:3] = pts[:,0:3]
            plydata['vertex'].data['x'] = aligned_vertices[:,0]
            plydata['vertex'].data['y'] = aligned_vertices[:,1]
            plydata['vertex'].data['z'] = aligned_vertices[:,2]
            
        plydata.write(filename_output)
