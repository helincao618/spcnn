import h5py
import struct
import numpy as np
import os
from tqdm import tqdm


THRESHOLD = 2.5


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


# locs: zyx ordering
def sparse_to_dense_np(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimz, dimy, dimx, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:, 0], locs[:, 1], locs[:, 2], :] = values
    if nf_values > 1:
        return dense.reshape([dimz, dimy, dimx, nf_values])
    return dense.reshape([dimz, dimy, dimx])


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


def dense_to_sparse_np(grid, thresh):
    locs = np.where(np.abs(grid) < thresh)
    values = grid[locs[0], locs[1], locs[2]]
    locs = np.stack(locs)
    return locs, values


def load_block(file):
    fin = open(file, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]
    world2grid = struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    # input data
    num = struct.unpack('Q', fin.read(8))[0]
    input_locs = struct.unpack('I' * num * 3, fin.read(num * 3 * 4))
    input_sdfs = struct.unpack('f' * num, fin.read(num * 4))
    input_sdfs = np.asarray(input_sdfs, dtype=np.float32)
    input_sdfs /= voxelsize
    # target data
    num = struct.unpack('Q', fin.read(8))[0]
    target_locs = struct.unpack('I' * num * 3, fin.read(num * 3 * 4))
    target_locs = np.asarray(target_locs, dtype=np.int32).reshape([num, 3])
    target_locs = np.flip(target_locs, 1).copy()  # convert to zyx ordering
    target_sdfs = struct.unpack('f' * num, fin.read(num * 4))
    target_sdfs = np.asarray(target_sdfs, dtype=np.float32)
    target_sdfs /= voxelsize
    target_sdfs = sparse_to_dense_np(target_locs, target_sdfs[:, np.newaxis], dimx, dimy, dimz, -float('inf'))
    # known data
    num = struct.unpack('Q', fin.read(8))[0]
    assert (num == dimx * dimy * dimz)
    target_known = struct.unpack('B' * dimz * dimy * dimx, fin.read(dimz * dimy * dimx))
    # pre-computed hierarchy
    hierarchy = []
    factor = 2
    for h in range(3):
        num = struct.unpack('Q', fin.read(8))[0]
        hlocs = struct.unpack('I' * num * 3, fin.read(num * 3 * 4))
        hlocs = np.asarray(hlocs, dtype=np.int32).reshape([num, 3])
        hlocs = np.flip(hlocs, 1).copy()  # convert to zyx ordering
        hvals = struct.unpack('f' * num, fin.read(num * 4))
        hvals = np.asarray(hvals, dtype=np.float32)
        hvals /= voxelsize
        grid = sparse_to_dense_np(hlocs, hvals[:, np.newaxis], dimx // factor, dimy // factor, dimz // factor,
                                  -float('inf'))
        hierarchy.append(grid)
        factor *= 2
    hierarchy.reverse()
    return target_sdfs, [dimz, dimy, dimx], world2grid, hierarchy


def semnantic_downsampling_factor2(semantic_block, block):
    zz, yy, xx = np.shape(semantic_block)
    block_downsample = np.zeros([int(zz/2), int(yy/2), int(xx/2)], dtype=np.ubyte)
    for i in range(int(zz/2)):
        for j in range(int(yy/2)):
            for k in range(int(xx/2)):
                if block[i,j,k]:
                    label_list = np.array([semantic_block[2*i,2*j,2*k],semantic_block[2*i,2*j,2*k+1],
                              semantic_block[2*i,2*j+1,2*k],semantic_block[2*i,2*j+1,2*k+1],
                              semantic_block[2*i+1,2*j,2*k],semantic_block[2*i+1,2*j,2*k+1],
                              semantic_block[2*i+1,2*j+1,2*k],semantic_block[2*i+1,2*j+1,2*k+1],
                              ])
                    bool_list = np.logical_and(label_list>0, label_list<50)
                    if True in bool_list:
                        block_downsample[i,j,k]=label_list[bool_list][0]
    return block_downsample


def load_scene(file):
    fin = open(file, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]
    world2grid = struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    fin.close()
    return world2grid


def main():
    output_path = '../dataset/h5_semantic_train_blocks/'
    input_scene_path = '../dataset/mp_sdf_vox_2cm_target/'
    input_block_path = '../dataset/completion_blocks/'
    input_semantic_path = '../dataset/h5_semantic_scenes_extraction/'
    block_index_list = os.listdir(input_block_path)
    print(block_index_list[405])
    for block_index in tqdm(block_index_list):
        if block_index[17] != '_':
            room_index = block_index[0:18]
        else:
            room_index = block_index[0:17]
        world2grid_scene = load_scene(input_scene_path+room_index+'__0__.sdf')
        target_sdfs_chunk, [dimz_chunk, dimy_chunk, dimx_chunk], world2grid_chunk, hierarchy_chunk= load_block(
            input_block_path + block_index)
        semantic_label_h5 = h5py.File(input_semantic_path+room_index+'__0__.h5', 'r')
        semantic_label = semantic_label_h5['label'][:]
        world2grid_semantic = semantic_label_h5['world2grid'][:]
        scale_zyx = world2grid_scene[0][0]/world2grid_semantic[0][1]
        target_trunk_occupancy = (target_sdfs_chunk < THRESHOLD) & (target_sdfs_chunk > -THRESHOLD)
        chunk_semantic = np.zeros(np.shape(target_sdfs_chunk), dtype=np.ubyte)
        world2grid_chunk = np.array([world2grid_chunk[2][3], world2grid_chunk[1][3], world2grid_chunk[0][3]])
        for i in range(dimz_chunk):
            for j in range(dimy_chunk):
                for k in range(dimx_chunk):
                    if target_trunk_occupancy[i][j][k]:
                        locs = (np.array([i,j,k]) - world2grid_chunk)/scale_zyx + world2grid_semantic[1]
                        if locs[0] < np.shape(semantic_label)[0] and locs[1] < np.shape(semantic_label)[1] and locs[2] < np.shape(semantic_label)[2]:
                            chunk_semantic[i][j][k] = semantic_label[int(locs[0])][int(locs[1])][int(locs[2])]

        hierarchy_chunk_level0 = (hierarchy_chunk[0] < THRESHOLD*8) & (hierarchy_chunk[0] > -THRESHOLD*8)
        hierarchy_chunk_level1 = (hierarchy_chunk[1] < THRESHOLD*4) & (hierarchy_chunk[1] > -THRESHOLD*4)
        hierarchy_chunk_level2 = (hierarchy_chunk[2] < THRESHOLD*2) & (hierarchy_chunk[2] > -THRESHOLD*2)
        hierarchy_semantic_level2 = semnantic_downsampling_factor2(chunk_semantic, hierarchy_chunk_level2)
        hierarchy_semantic_level1 = semnantic_downsampling_factor2(hierarchy_semantic_level2,hierarchy_chunk_level1)
        hierarchy_semantic_level0 = semnantic_downsampling_factor2(hierarchy_semantic_level1,hierarchy_chunk_level0)
        # write to h5 file
        f = h5py.File(output_path + block_index.rstrip('.sdfs') +'.h5', 'w')
        f.create_dataset('label', data=chunk_semantic)
        f.create_dataset('hierarchy_level0', data=hierarchy_semantic_level0)
        f.create_dataset('hierarchy_level1', data=hierarchy_semantic_level1)
        f.create_dataset('hierarchy_level2', data=hierarchy_semantic_level2)
        f.close()
        # visualization
        # visualize_occupancy(target_trunk_occupancy, 'target_trunk_occupancy.ply')
        # visualize_semantic(chunk_semantic, 'chunk_semantic.ply')
        # visualize_occupancy(hierarchy_chunk_level0, 'hierarchy_chunk_level0.ply')
        # visualize_occupancy(hierarchy_chunk_level1, 'hierarchy_chunk_level1.ply')
        # visualize_occupancy(hierarchy_chunk_level2, 'hierarchy_chunk_level2.ply')
        # visualize_semantic(hierarchy_semantic_level0, 'hierarchy_semantic_level0.ply')
        # visualize_semantic(hierarchy_semantic_level1, 'hierarchy_semantic_level1.ply')
        # visualize_semantic(hierarchy_semantic_level2, 'hierarchy_semantic_level2.ply')



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
