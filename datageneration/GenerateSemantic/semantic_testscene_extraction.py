import struct
import h5py
import numpy as np
from plyfile import PlyData
import json
import os
from tqdm import tqdm
import csv
import argparse

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--output_dir', type=str, required=True, help='output directory')
parser.add_argument('--target_dir', type=str, required=True, help='target sdf directory')
parser.add_argument('--mesh_dir', type=str, required=True, help='mesh directory')
args = parser.parse_args()

def create_color_palette():
    return [
        (0, 0, 0),
        (174, 199, 232),  # wall
        (152, 223, 138),  # floor
        (31, 119, 180),  # cabinet
        (255, 187, 120),  # bed
        (188, 189, 34),  # chair
        (140, 86, 75),  # sofa
        (255, 152, 150),  # table
        (214, 39, 40),  # door
        (197, 176, 213),  # window
        (148, 103, 189),  # bookshelf
        (196, 156, 148),  # picture
        (23, 190, 207),  # counter
        (178, 76, 76),
        (247, 182, 210),  # desk
        (66, 188, 102),
        (219, 219, 141),  # curtain
        (140, 57, 197),
        (202, 185, 52),
        (51, 176, 203),
        (200, 54, 131),
        (92, 193, 61),
        (78, 71, 183),
        (172, 114, 82),
        (255, 127, 14),  # refrigerator
        (91, 163, 138),
        (153, 98, 156),
        (140, 153, 101),
        (158, 218, 229),  # shower curtain
        (100, 125, 154),
        (178, 127, 135),
        (120, 185, 128),
        (146, 111, 194),
        (44, 160, 44),  # toilet
        (112, 128, 144),  # sink
        (96, 207, 209),
        (227, 119, 194),  # bathtub
        (213, 92, 176),
        (94, 106, 211),
        (82, 84, 163),  # otherfurn
        (100, 85, 144)
    ]

def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []
    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2], int(color[0]),
                                                            int(color[1]), int(color[2])))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()

def load_scene(file):
    fin = open(file, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]
    world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    # data
    num = struct.unpack('Q', fin.read(8))[0]
    locs = struct.unpack('I'*num*3, fin.read(num*3*4))
    locs = np.asarray(locs, dtype=np.int32).reshape([num, 3])
    locs = np.flip(locs,1).copy() # convert to zyx ordering
    sdf = struct.unpack('f'*num, fin.read(num*4))
    sdf = np.asarray(sdf, dtype=np.float32)
    sdf /= voxelsize
    fin.close()
    return world2grid

def visualize_semantic(semantic_label, out_name):
    zz, yy, xx = np.shape(semantic_label)
    count = 0
    points = []
    indices = []
    colors = []
    for i in range(zz):
        for j in range(yy):
            for k in range(xx):
                if (semantic_label[i][j][k] > 0) & (semantic_label[i][j][k] < 50):
                    points.append([i + 0.49, j + 0.49, k + 0.49])
                    points.append([i - 0.49, j + 0.49, k + 0.49])
                    points.append([i + 0.49, j - 0.49, k + 0.49])
                    points.append([i - 0.49, j - 0.49, k + 0.49])
                    points.append([i + 0.49, j + 0.49, k - 0.49])
                    points.append([i - 0.49, j + 0.49, k - 0.49])
                    points.append([i + 0.49, j - 0.49, k - 0.49])
                    points.append([i - 0.49, j - 0.49, k - 0.49])
                    indices.append([8 * count, 8 * count + 1, 8 * count + 2])
                    indices.append([8 * count + 1, 8 * count + 2, 8 * count + 3])
                    indices.append([8 * count + 2, 8 * count + 3, 8 * count + 6])
                    indices.append([8 * count + 3, 8 * count + 6, 8 * count + 7])
                    indices.append([8 * count, 8 * count + 2, 8 * count + 4])
                    indices.append([8 * count + 2, 8 * count + 4, 8 * count + 6])
                    indices.append([8 * count, 8 * count + 1, 8 * count + 4])
                    indices.append([8 * count + 1, 8 * count + 4, 8 * count + 5])
                    indices.append([8 * count + 1, 8 * count + 3, 8 * count + 5])
                    indices.append([8 * count + 3, 8 * count + 5, 8 * count + 7])
                    indices.append([8 * count + 4, 8 * count + 5, 8 * count + 6])
                    indices.append([8 * count + 5, 8 * count + 6, 8 * count + 7])
                    for a in range(8):
                        colors.append(list(create_color_palette()[semantic_label[i][j][k]]))
                    count += 1
    write_ply(points, colors, indices, out_name)

def read_ply(ply_file):
    with open(ply_file, 'rb') as read_file:
        ply_data = PlyData.read(read_file)
    points = []
    for x, y, z, nx, ny, nz, tx, ty, r, g, b in ply_data['vertex']:
        points.append([z, y, x])
    points = np.array(points)
    return points

def name_to_nuy40id(name, raw_name_list, name_list, id_list):
    try:
        return int(id_list[np.where(raw_name_list == name)])
    except:
        try:
            return int(id_list[np.where(name_list == name)])
        except:
            print(name)


def main():
    scenelist = open("../../filelists/scenelist_test.txt", "r")
    lines = scenelist.read().splitlines()
    scenelist.close()
    name_mapping = open('../../filelists/category_mapping.tsv')
    name_mapping = np.array(list(csv.reader(name_mapping, delimiter="\t")))
    raw_name_list = name_mapping[1:, 1]
    name_list = name_mapping[1:, 2]
    id_list = name_mapping[1:, 5]
    id_list[1239] = '40'
    output_dir = args.output_dir
    mesh_file_dir = args.mesh_dir
    target_file_dir = args.target_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for scene in tqdm(lines):
        print('processing the scene ' + scene)
        input_path = mesh_file_dir + scene
        output_path = output_dir + scene
        list_dir = os.listdir(input_path)
        ply_files = [ply_file for ply_file in list_dir if 'ply' in ply_file]
        for ply_file in ply_files:
            ply_file_index = int(ply_file.lstrip('region').rstrip('.ply'))
            target_file_path = target_file_dir + scene + '_room' + str(ply_file_index) + '__0__.sdf'
            if not os.path.exists(target_file_path):
                print(scene + '_room' + str(ply_file_index) + '__0__.sdf')
                continue
            world2grid = load_scene(target_file_path)
            print('processing the room ' + str(ply_file_index) + ' of the scene ' + scene)
            points = read_ply(input_path + '/' + ply_file)
            with open(input_path + '/region' + str(ply_file_index) + '.semseg.json', 'r') as load_f:
                load_dict = json.load(load_f)
            label_list = load_dict['segGroups']
            with open(input_path + '/region' + str(ply_file_index) + '.vsegs.json', 'r') as load_f:
                load_dict = json.load(load_f)
            categories = load_dict['segIndices']
            scale = [max(points[:, 0]) - min(points[:, 0]), max(points[:, 1]) - min(points[:, 1]),
                     max(points[:, 2]) - min(points[:, 2])]
            volumn = np.zeros([int(scale[0] * world2grid[0][0]*1.5), int(scale[1] * world2grid[1][1]*1.5),
                               int(scale[2] * world2grid[2][2]*1.5)], dtype=np.ubyte)
            new_points = np.zeros(np.shape(points))
            for i in range(len(categories)):
                new_points[i][0] = points[i][0] * world2grid[0][0] + world2grid[2][3]
                new_points[i][1] = points[i][1] * world2grid[1][1] + world2grid[1][3]
                new_points[i][2] = points[i][2] * world2grid[2][2] + world2grid[0][3]
                for label_dict in label_list:
                    if categories[i] in label_dict["segments"]:
                        label_name = label_dict["label"]
                        label_index_nyu = name_to_nuy40id(label_name, raw_name_list, name_list, id_list)
                        volumn[int(new_points[i][0])][int(new_points[i][1])][int(new_points[i][2])] = label_index_nyu
            visualize_semantic(volumn, output_path + '_room' + str(ply_file_index) + '__0__.ply')
            f = h5py.File(output_path + '_room' + str(ply_file_index) + '__0__.h5', 'w')
            f.create_dataset('label', data=volumn)
            f.create_dataset('world2grid', data=world2grid)
            f.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass