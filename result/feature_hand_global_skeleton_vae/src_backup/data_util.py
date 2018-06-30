import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from pyfeature import motion_feature as mf

np.random.seed(10000)

def normalize_sequences(seqs):
    '''
    meanval = 0
    minval = 1000
    maxval = -1000
    '''
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

            '''
            # stats
            if minval > np.min(seqs[i]):
                minval = np.min(seqs[i])
            if maxval < np.max(seqs[i]):
                maxval = np.max(seqs[i])
            meanval += np.mean(seqs[i])/len(seqs)
            '''
        #print maxdist
        nor_palm = 1.0 / maxdist
        #print nor_palm
        #print seqs[i][:, [3,4,5]].shape
        offset = nor_palm * seqs[i][:, [3,4,5]] - seqs[i][:, [3,4,5]]
        #print offset
        #print seqs[i].shape[1]/3
        t = repmat(offset, 1, seqs[i].shape[1]/3)
        seqs[i] = seqs[i] + t

        #print offset
        #print seqs[i]
        # filter
        '''
        seq = np.copy(seqs[i]);
        for j in range(n):
            if j >= 2 and j < n - 2:
                seqs[i][j, :, :] = (-3 * seq[j - 2, :, :] + 12 * seq[j - 1, :, :]
                        + 17 * seq[j, :, :] + 12 * seq[j + 1, :, :] - 3 * seq[j + 2, :, :]) / 35;
        '''
    #print minval, maxval, meanval
    return seqs


def resample_sequences(seqs):
    target_len = 2 * 30
    res_seq = []
    for i in range(len(seqs)):
        n = seqs[i].shape[0]
        # find still frames
        dis = []
        for frame in range(n - 1):
            dis.append(np.mean(np.sum((seqs[i][frame + 1, :, :] - seqs[i][frame, :, :]) ** 2, axis = 1)))
        th = 10
        be = 0
        for frame in range(1, n - 1):
            if dis[frame] > th:
                be = frame
                break

        en = n
        for frame in range(n - 2, 0, -1):
            if dis[frame] > th:
                en = frame + 1
                break

        xp = np.arange(en - be) * 1.0 / (en - be - 1) * (target_len - 1)
        xt = np.arange(target_len)
        res_seq.append(np.zeros([target_len, seqs[i].shape[1], seqs[i].shape[2]]).copy())
        for j in range(seqs[i].shape[1]):
            for k in range(seqs[i].shape[2]):
                fp = seqs[i][be : en, j, k].ravel()
                res_seq[i][:, j, k] = np.interp(xt, xp, fp)

    return res_seq


#from pose_cluster import PoseCluster
def cluster_pose(seqs, k):
    poses = []
    for i in range(len(seqs)):
        n = seqs[i].shape[0]
        for frame in range(n):
            poses.append(seqs[i][frame, :, :3].ravel())

    cluster = PoseCluster(k)
    cluster.fit(np.array(poses))
    return cluster


def transform_sequences(seqs, cluster):
    res_seq = []
    for i in range(len(seqs)):
        res = cluster.predict(np.reshape(seqs[i][:, :, :3], [seqs[i].shape[0], -1]))
        res_seq.append(res.ravel())
    return res_seq


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
    
    color = 'b'
    plt.gca().cla()
    
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
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

    # pose 2
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(pose2[:, 0], pose2[:, 1], pose2[:, 2], c=color, marker='.', s=10)
    ax2.set_xlim3d(xmin, xmax)
    ax2.set_ylim3d(ymin, ymax)
    ax2.set_zlim3d(zmin, zmax)
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_zlabel('Z axis')
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
                    [pose2[id1, 2], pose2[id2, 2]], c=color)
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
