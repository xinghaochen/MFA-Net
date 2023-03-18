import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from pyfeature import motion_feature as mf

np.random.seed(10000)

def normalize_sequences(seqs):
    for i in range(len(seqs)):
        n = seqs[i].shape[0]
        # normalize
        first_palm_pos = seqs[i][0, [3,4,5]]
        maxdist = 0
        maxid = 1
        for j in range(n):
            t = repmat(first_palm_pos, 1, seqs[i].shape[1]/3)
            t = t.flatten()
            seqs[i][j, :] -= t
            if j == 0:
                continue
            palm_dist = np.linalg.norm(seqs[i][0, [3,4,5]] - seqs[i][j, [3,4,5]])
            if palm_dist > maxdist:
                maxdist = palm_dist
                maxid = j
        nor_palm = 1.0 / maxdist
        offset = nor_palm * seqs[i][:, [3,4,5]] - seqs[i][:, [3,4,5]]

        t = repmat(offset, 1, seqs[i].shape[1]/3)
        seqs[i] = seqs[i] + t

    return seqs


def show_sequence(seq, label, is_show_connect=1, is_show_id=0):
    lines = [[0, 2, 3, 4, 0, 1, 6, 7, 8, 1,  10, 11, 12, 1,  14, 15, 16, 1,  18, 19, 20],
             [2, 3, 4, 5, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]

    seq = np.reshape(seq, [seq.shape[0], -1, 3])
    print seq[:, :, 0]
    print seq.dtype
    print seq.shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print seq.max(2)
    xmax = np.max(seq[:, :, 0])
    xmin = np.min(seq[:, :, 0])
    ymax = np.max(seq[:, :, 1])
    ymin = np.min(seq[:, :, 1])
    zmax = np.max(seq[:, :, 2])
    zmin = np.min(seq[:, :, 2])
    print xmax, xmin
    print ymax, ymin
    print zmax, zmin
    for frame in range(seq.shape[0]):
        color = 'b'
        plt.gca().cla()
        #print seq[frame, :, :]
        #print '\n'
        # draw points
        ax.scatter(seq[frame, :, 0], seq[frame, :, 1], seq[frame, :, 2], c=color, marker='.', s=10)
        ax.set_xlim3d(xmin, xmax)
        ax.set_ylim3d(ymin, ymax)
        ax.set_zlim3d(zmin, zmax)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        if is_show_id:
            for i in range(seq.shape[1]):
                ax.text(seq[frame, i, 0], seq[frame, i, 1], seq[frame, i, 2], str(i))
        if is_show_connect:
            # draw lines
            for i in range(len(lines[0])):
                id1 = lines[0][i]
                id2 = lines[1][i]
                ax.plot([seq[frame, id1, 0], seq[frame, id2, 0]],
                        [seq[frame, id1, 1], seq[frame, id2, 1]],
                        [seq[frame, id1, 2], seq[frame, id2, 2]], c=color)

        plt.title(str(label))
        plt.pause(0.1)
        
def show_skeleton(pose, is_show_connect=1, is_show_id=0):
    lines = [[0, 2, 3, 4, 0, 1, 6, 7, 8, 1,  10, 11, 12, 1,  14, 15, 16, 1,  18, 19, 20],
             [2, 3, 4, 5, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]

    xmax = np.max(pose[:, 0])
    xmin = np.min(pose[:, 0])
    ymax = np.max(pose[:, 1])
    ymin = np.min(pose[:, 1])
    zmax = np.max(pose[:, 2])
    zmin = np.min(pose[:, 2])
    
    color = 'b'
    plt.gca().cla()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=color, marker='.', s=10)
    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    ax.set_zlim3d(zmin, zmax)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    if is_show_id:
        for i in range(pose.shape[0]):
            ax.text(pose[i, 0], pose[i, 1], pose[i, 2], str(i))
    if is_show_connect:
        # draw lines
        for i in range(len(lines[0])):
            id1 = lines[0][i]
            id2 = lines[1][i]
            ax.plot([pose[id1, 0], pose[id2, 0]],
                    [pose[id1, 1], pose[id2, 1]],
                    [pose[id1, 2], pose[id2, 2]], c=color)

    plt.pause(0)

def show_two_skeleton(pose, pose2, is_show_connect=1, is_show_id=0):
    lines = [[0, 2, 3, 4, 0, 1, 6, 7, 8, 1,  10, 11, 12, 1,  14, 15, 16, 1,  18, 19, 20],
             [2, 3, 4, 5, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]

    xmax = np.max(pose[:, 0])
    xmin = np.min(pose[:, 0])
    ymax = np.max(pose[:, 1])
    ymin = np.min(pose[:, 1])
    zmax = np.max(pose[:, 2])
    zmin = np.min(pose[:, 2])
    xmax = xmax + (xmax - xmin)
    xmin = xmin - (xmax - xmin)
    ymax = ymax + (ymax - ymin)
    ymin = ymin - (ymax - ymin)

    color = 'b'

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=color, marker='.', s=100)
    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    ax.set_zlim3d(zmin, zmax)
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    ax.grid(False)
    if is_show_id:
        for i in range(pose.shape[0]):
            ax.text(pose[i, 0], pose[i, 1], pose[i, 2], str(i))
    if is_show_connect:
        # draw lines
        for i in range(len(lines[0])):
            id1 = lines[0][i]
            id2 = lines[1][i]
            ax.plot([pose[id1, 0], pose[id2, 0]],
                    [pose[id1, 1], pose[id2, 1]],
                    [pose[id1, 2], pose[id2, 2]], c='r', linewidth=2)
    ax.view_init(elev=-69, azim=60)
    ax.w_xaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.w_yaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.w_zaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.w_xaxis.line.set_color((1.0,1.0,1.0,0.0))
    ax.w_yaxis.line.set_color((1.0,1.0,1.0,0.0))
    ax.w_zaxis.line.set_color((1.0,1.0,1.0,0.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.set_aspect('equal')

    # pose 2
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(pose2[:, 0], pose2[:, 1], pose2[:, 2], c=color, marker='.', s=100)
    ax2.set_xlim3d(xmin, xmax)
    ax2.set_ylim3d(ymin, ymax)
    ax2.set_zlim3d(zmin, zmax)
    # ax2.set_xlabel('X axis')
    # ax2.set_ylabel('Y axis')
    # ax2.set_zlabel('Z axis')
    # ax2.grid(False)

    if is_show_id:
        for i in range(pose2.shape[0]):
            ax2.text(pose2[i, 0], pose2[i, 1], pose2[i, 2], str(i))
    if is_show_connect:
        # draw lines
        for i in range(len(lines[0])):
            id1 = lines[0][i]
            id2 = lines[1][i]
            ax2.plot([pose2[id1, 0], pose2[id2, 0]],
                    [pose2[id1, 1], pose2[id2, 1]],
                    [pose2[id1, 2], pose2[id2, 2]], c='r', linewidth=2)
    ax2.view_init(elev=-69, azim=60)
    ax2.w_xaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax2.w_yaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax2.w_zaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax2.w_xaxis.line.set_color((1.0,1.0,1.0,0.0))
    ax2.w_yaxis.line.set_color((1.0,1.0,1.0,0.0))
    ax2.w_zaxis.line.set_color((1.0,1.0,1.0,0.0))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    # ax2.set_aspect('equal')

    plt.pause(0.1)


def show_depth_sequence(data_dir, gesture, finger, subject, essai, frame):
    img=mpimg.imread('stinkbug.png')
    imgplot = plt.imshow(img)


if __name__ == '__main__':
    import gesture_dataset
    data_dir = '/home/workspace/Datasets/DHG2016'
    test_id = 2
    data = gesture_dataset.Dataset(data_dir, 0)
    (x_train, y_train), (x_test, y_test) = data.load_data(test_id, is_preprocess=False)
    '''
    x_train = normalize_sequences(x_train)
    x_train = resample_sequences(x_train)
    cluster = cluster_pose(x_train, 256)
    x_train_new = transform_sequences(x_train, cluster)
    # print x_train_new
    '''
    for i in range(40, len(x_train)):
        print "Showing sequence %d" % i
        show_sequence(x_test[i], y_test[i], 1, 0)
        plt.waitforbuttonpress()
        plt.close()
