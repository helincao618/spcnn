import numpy as np
import struct
import h5py
import os
from plyfile import PlyData
from tqdm import tqdm

THRESHOLD = 1.7
padding = 1

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
        file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2], int(color[0] * 255),
                                                            int(color[1] * 255), int(color[2] * 255)))
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
    return [locs, sdf], [dimz, dimy, dimx], world2grid

def sparse_to_dense_np(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimz, dimy, dimx, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:,0], locs[:,1], locs[:,2],:] = values
    if nf_values > 1:
        return dense.reshape([dimz, dimy, dimx, nf_values])
    return dense.reshape([dimz, dimy, dimx])

def visualize_occupancy(occupancy, out_name):
    zz, yy, xx = np.shape(occupancy)
    count = 0
    points = []
    indices = []
    colors = []
    for i in range(zz):
        for j in range(yy):
            for k in range(xx):
                if occupancy[i][j][k]:
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
                        colors.append([128, 128, 128])
                    count += 1
    write_ply(points, colors, indices, out_name)

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

def padding(volumn, padding):
    zz,yy,xx =  np.shape(volumn)
    volumn_padding = np.zeros([zz+padding,yy+padding,xx+padding])
    volumn_padding[0:zz,0:yy,0:xx] = volumn
    return volumn_padding

def read_ply(ply_file):
    with open(ply_file, 'rb') as read_file:
        ply_data = PlyData.read(read_file)
    verts = []
    colors = []
    indices = []
    for x, y, z, r, g, b in ply_data['vertex']:
        verts.append([z, y, x])
        colors.append([r, g, b])
    verts = np.array(verts)
    colors = np.array(colors)
    for face in ply_data['face']:
        indices.append(face[0])
    return verts, colors, indices


def main():
    output_dir = '../dataset/h5_semantic_groundtruth_scenes/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    target_file_dir = '../dataset/target_mesh/'
    scene_list = os.listdir(target_file_dir)
    # scene_list = [scene for scene in scene_list if '' in scene]
    for scene in tqdm(scene_list[186:]):
        if scene[22] == '_':
            target_file_path = '../dataset/mp_sdf_vox_2cm_target/'+ scene[:23] +'.sdf'
            semantic_file_path = '../dataset/h5_semantic_scenes_extraction/'+ scene[:18] +'.h5'
        else:
            target_file_path = '../dataset/mp_sdf_vox_2cm_target/' + scene[:22] + '.sdf'
            semantic_file_path = '../dataset/h5_semantic_scenes_extraction/' + scene[:17] + '.h5'

        target_mesh_path = target_file_dir + scene
        [locs, sdf], [dimz, dimy, dimx], world2grid = load_scene(target_file_path)
        target_sdf_scene = sparse_to_dense_np(locs, sdf.reshape(-1,1), dimx, dimy, dimz, -100000)

        semantic_file = h5py.File(semantic_file_path, 'r')
        semantic_label = semantic_file['label'][:]
        semantic_world2grid = semantic_file['world2grid'][:]

        target_occ_scene = (target_sdf_scene < THRESHOLD) & (target_sdf_scene > -THRESHOLD)
        target_occ_scene = target_occ_scene[:128,:,:]  # Remove the ceiling for easier visualization
        zz, yy, xx = np.shape(target_occ_scene)
        target_semantic_scene = np.zeros([zz,yy,xx],dtype=np.ubyte)
        semantic_label = padding(semantic_label,1)
        for z in range(zz):
            for y in range(yy):
                for x in range(xx):
                    if target_occ_scene[z,y,x]:
                        locs = (np.array([[z,y,x]]) - world2grid[0:3, 3].T)/world2grid[0,0]*semantic_world2grid[0,0] + \
                               semantic_world2grid[1]#change the order because the trans in semantic is in order xyz, but here should be zyx
                        if locs[0][0] < np.shape(semantic_label)[0]-1 and locs[0][1] < np.shape(semantic_label)[1]-1 \
                                and locs[0][2] < np.shape(semantic_label)[2]-1:
                            if semantic_label[int(locs[0][0])][int(locs[0][1])][int(locs[0][2])]>0:
                                target_semantic_scene[z,y,x] = semantic_label[int(locs[0][0])][int(locs[0][1])][int(locs[0][2])]
                            elif semantic_label[int(locs[0][0]+1)][int(locs[0][1])][int(locs[0][2])]>0:
                                target_semantic_scene[z, y, x] = semantic_label[int(locs[0][0]+1)][int(locs[0][1])][int(locs[0][2])]
                            elif semantic_label[int(locs[0][0] - 1)][int(locs[0][1])][int(locs[0][2])] > 0:
                                target_semantic_scene[z, y, x] = semantic_label[int(locs[0][0] - 1)][int(locs[0][1])][int(locs[0][2])]
                            elif semantic_label[int(locs[0][0])][int(locs[0][1]+1)][int(locs[0][2])]>0:
                                target_semantic_scene[z, y, x] = semantic_label[int(locs[0][0])][int(locs[0][1]+1)][int(locs[0][2])]
                            elif semantic_label[int(locs[0][0])][int(locs[0][1]-1)][int(locs[0][2])] > 0:
                                target_semantic_scene[z, y, x] = semantic_label[int(locs[0][0])][int(locs[0][1]-1)][int(locs[0][2])]
                            elif semantic_label[int(locs[0][0])][int(locs[0][1])][int(locs[0][2])+1]>0:
                                target_semantic_scene[z, y, x] = semantic_label[int(locs[0][0])][int(locs[0][1])][int(locs[0][2])+1]
                            else:
                                target_semantic_scene[z, y, x] = semantic_label[int(locs[0][0])][int(locs[0][1])][int(locs[0][2])-1]
        if scene[22] == '_':
            f = h5py.File(output_dir + scene[:23] + '.h5', 'w')
            f.create_dataset('label', data=target_semantic_scene)
            f.create_dataset('world2grid', data=world2grid)
            f.close()
            # Visualize in mesh
            verts, colors, indices = read_ply(target_mesh_path)
            for i, vert in enumerate(verts):
                colors[i] = list(
                    create_color_palette()[target_semantic_scene[int(vert[0])][int(vert[1])][int(vert[2])]])
            write_ply(verts, colors, indices, output_dir + scene[:23] + '.ply')
        else:
            f = h5py.File(output_dir + scene[:22] + '.h5', 'w')
            f.create_dataset('label', data=target_semantic_scene)
            f.create_dataset('world2grid', data=world2grid)
            f.close()
            # Visualize in mesh
            verts, colors, indices = read_ply(target_mesh_path)
            for i, vert in enumerate(verts):
                colors[i] = list(
                    create_color_palette()[target_semantic_scene[int(vert[0])][int(vert[1])][int(vert[2])]])
            write_ply(verts, colors, indices, output_dir + scene[:22] + '.ply')


        # visualize_semantic(target_semantic_scene, '2t7WUuJeko7_room0__0__semantic.ply')
        # visualize_occupancy(target_occ_scene, '2t7WUuJeko7_room0__0__occ.ply')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass