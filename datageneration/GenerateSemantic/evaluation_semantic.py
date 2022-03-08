import numpy as np
import h5py
import os
from plyfile import PlyData
from tqdm import tqdm
import matplotlib.pyplot as plt

# EVAL_IDX_20 = [1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]
# LABEL_LIST_20 = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter',
#               'desk','curtain','refridgerator','shower curtain','toilet','sink','bathtub','otherfurniture']
# EVAL_IDX_20_NEW = [1,2,3,4,5,6,7,8,9,11,12,15,16,18,22,25,33,34,36,38]
# LABEL_LIST_20_NEW = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','picture','counter','shelves',
#                      'curtain','pillow','refridgerator','television','toilet','sink','bathtub','otherstructure']
EVAL_IDX = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
LABEL_LIST = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter', 'blinds',                                                                                                   ''
              'desk','shelves','curtain','dresser','pillow','mirror','floormat', 'clothes', 'ceiling', 'books','refridgerator',
              'television', 'paper', 'towel','shower curtain','box','whiteboard', 'person', 'nightstand','toilet','sink','lamp',
              'bathtub','bag','otherstructure','otherfurniture', 'otherprop']
EVAL_IDX_20 = [1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,38]
LABEL_LIST_20 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
                 'counter', 'desk', 'curtain', 'refridgerator', 'shower', 'toilet','sink', 'bathtub', 'otherstructure']
EVAL_IDX_16 = [1,2,3,4,5,6,7,8,9,11,12,16,33,34,36,38]
LABEL_LIST_16 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'picture',
                 'counter', 'curtain', 'toilet', 'sink', 'bathtub', 'otherstructure']

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
    verts = []
    colors = []
    indices = []
    for x, y, z, nx, ny, nz, tx, ty, r, g, b in ply_data['vertex']:
        verts.append([z, y, x])
        colors.append([r, g, b])
    verts = np.array(verts)
    colors = np.array(colors)
    for face in ply_data['face']:
        indices.append(face[0])
    return verts, colors, indices

def main():
    evaluation_idx = EVAL_IDX_16
    label_list = LABEL_LIST_16
    gt_file_dir = '../dataset/h5_semantic_groundtruth_scenes/'
    output_dir = '../dataset/semantic_prediction/'
    testlist = open("test_list.txt", "r")
    test_list = testlist.read().splitlines()
    testlist.close()
    scene_list = [scene[29:] for scene in test_list]
    iou_list = np.zeros([2,len(evaluation_idx)])
    acc_list = np.zeros([2, len(evaluation_idx)])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i, scene in tqdm(enumerate(scene_list[5:])):
        if scene[17] == '.':
            gt_file_path = gt_file_dir + scene[:17]+'__0__.h5'
            pred_file_path = '../mp/' + scene[:17] + '__0__semantic.h5'
            target_mesh_path = '../dataset/room_mesh/' + scene[:11] + '/region' + scene[16:17] + '.ply'
        else:
            gt_file_path = gt_file_dir + scene[:18]+'__0__.h5'
            pred_file_path = '../mp/' + scene[:18] + '__0__semantic.h5'
            target_mesh_path = '../dataset/room_mesh/' + scene[:11] + '/region' + scene[16:18] + '.ply'

        if not os.path.exists(pred_file_path) or not os.path.exists(gt_file_path):
            print(pred_file_path)
            continue
        f_pred = h5py.File(pred_file_path, 'r')
        sementic_pred = f_pred['label'][:]
        semantic_world2grid = f_pred['world2grid'][:]
        f_gt = h5py.File(gt_file_path, 'r')
        sementic_gt = f_gt['label'][:]
        # Chop to same size
        zz = min(np.shape(sementic_pred)[0], np.shape(sementic_gt)[0])
        yy = min(np.shape(sementic_pred)[1], np.shape(sementic_gt)[1])
        xx = min(np.shape(sementic_pred)[2], np.shape(sementic_gt)[2])
        sementic_pred = sementic_pred[:zz,:yy,:xx]
        sementic_gt = sementic_gt[:zz,:yy,:xx]
        # Only keep iso surface
        sementic_pred_visualization = sementic_pred
        sementic_pred[sementic_gt < 0.5] = 0
        # visualize_semantic(sementic_gt,'sementic_gt.ply')
        # visualize_semantic(sementic_pred,'sementic_pred.ply')
        # IOU and acc
        for j, cls in enumerate(evaluation_idx):
            gt_mask = (sementic_gt == cls).reshape(-1)
            pred_mask = (sementic_pred == cls).reshape(-1)
            if np.sum(gt_mask | pred_mask) > 0 and np.sum(gt_mask & pred_mask) > 0:
                iou_list[0][j] = iou_list[0][j] + np.sum(gt_mask & pred_mask)
                iou_list[1][j] = iou_list[1][j] + np.sum(gt_mask | pred_mask)
            pred_mask[gt_mask == 0] = 0
            acc_list[0][j] = acc_list[0][j] + np.sum(pred_mask)
            acc_list[1][j] = acc_list[1][j] + np.sum(gt_mask)
        # Visualize the prediction in mesh
        # verts, colors, indices = read_ply(target_mesh_path)
        # new_verts = np.zeros(np.shape(verts))
        # for i in range(np.shape(verts)[0]):
        #     new_verts[i][0] = verts[i][0]*semantic_world2grid[0][0][0]+semantic_world2grid[0][2][3]
        #     new_verts[i][1] = verts[i][1]*semantic_world2grid[0][0][0]+semantic_world2grid[0][1][3]
        #     new_verts[i][2] = verts[i][2]*semantic_world2grid[0][0][0]+semantic_world2grid[0][0][3]
        # new_verts = np.array(new_verts)
        # for j, vert in enumerate(new_verts):
        #     if 0<vert[0]<zz and 0<vert[1]<yy and 0<vert[2]<xx:
        #         colors[j] = list(
        #             create_color_palette()[sementic_pred_visualization[int(vert[0])][int(vert[1])][int(vert[2])]])
        #     else:
        #         colors[j] = [128,128,128]
        # if scene[17] == '.':
        #     write_ply(new_verts, colors, indices, output_dir + scene[:17] + '__0__pred-semantic.ply')
        # else:
        #     write_ply(new_verts, colors, indices, output_dir + scene[:18] + '__0__pred-semantic.ply')
    iou_list[1][iou_list[1] == 0] = 1
    iou_list[0][iou_list[1] == 0] = 0
    iou_list = iou_list[0]/iou_list[1]
    label_list_iou = ['avg iou']+label_list
    mean_iou = [np.mean(iou_list)]+ [iou for iou in iou_list]
    # Visualize the result
    plt.bar(label_list_iou, mean_iou)
    plt.xticks(fontsize=10, rotation=70)
    plt.xlabel('Class')
    plt.ylabel('Average IoU')
    plt.title("Evaluation in Matterport20 Class")
    plt.tight_layout()
    for a, b in zip(label_list_iou, mean_iou):
        plt.text(a, b + 0.003, '%.3f' % b, ha='center', va='bottom', fontsize=6)
    plt.show()

    acc_list[1][acc_list[1] == 0] = 1
    acc_list[0][acc_list[1] == 0] = 0
    acc_list = acc_list[0] / acc_list[1]
    label_list_acc = ['avg acc'] + label_list
    mean_acc = [np.mean(acc_list)] + [iou for iou in acc_list]
    # Visualize the result
    plt.bar(label_list_acc, mean_acc)
    plt.xticks(fontsize=10, rotation=70)
    plt.xlabel('Class')
    plt.ylabel('Average Accuracy')
    plt.title("Evaluation in Matterport20 Class")
    plt.tight_layout()
    for a, b in zip(label_list_acc, mean_acc):
        plt.text(a, b + 0.003, '%.3f' % b, ha='center', va='bottom', fontsize=6)
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
