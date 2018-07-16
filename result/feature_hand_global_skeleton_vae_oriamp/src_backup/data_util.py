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
    xmax = xmax + (xmax - xmin)
    xmin = xmin - (xmax - xmin)
    ymax = ymax + (ymax - ymin)
    ymin = ymin - (ymax - ymin)

    color = 'b'
    # plt.gca().cla() #
    
    # fig = plt.figure(figsize=(32, 16), facecolor='white', edgecolor='white', dpi=100, frameon=True)
    # fig.patch.set_alpha(0.0)
    # plt.axis('off')

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


def random_resample_sequence(seq, res_range):
    """
    randomly interpolate or skip several frames
    :return: 
    """
    seq_len = seq.shape[0]
    rnd = np.random.randint(res_range*2+1) - res_range
    if seq_len+rnd <= 0:
        rnd = -(seq_len - 1)
    new_seq = np.zeros((seq_len+rnd, seq.shape[1]), dtype=np.float32)
    # upsample
    if rnd > 0:
        rnd_idx = np.random.permutation(seq_len+rnd)
        rnd_idx = np.sort(rnd_idx)
        new_seq[rnd_idx[:seq_len], :] = seq
        for idx in rnd_idx[seq_len:seq_len+rnd]:
            if idx > 0:
                new_seq[idx, :] = new_seq[idx-1, :]
            else:
                new_seq[idx, :] = new_seq[idx - 1, :]

    # downsample
    elif rnd < 0:
        rnd_idx = np.random.permutation(seq_len)
        rnd_idx = rnd_idx[:seq_len+rnd]
        new_seq = seq[rnd_idx, :]
    else:
        new_seq = seq
    return new_seq


def random_translate_sequence(seq, trans_range):
    """
    randomly translate gesture sequences
    :param seq: 
    :param trans_range: 
    :return: 
    """
    for frame_idx in range(seq.shape[0]):
        seq[frame_idx, :] = seq[frame_idx, :] + np.random.rand(seq.shape[1]) * trans_range
    return seq


def random_scale_sequence(seq, scale_range = 0.2):
    # Scale
    # for frame_idx in range(seq.shape[0]):
    #     seq[frame_idx, :] = seq[frame_idx, :] * (1 + (np.random.rand(seq.shape[1]) - 0.5) * scale_range)
    # return seq

    # f = np.random.uniform(0.8, 1.2)
    # Xscale = X * f
    # Yscale = Y
    # XY_train.append((Xscale, Yscale))
    s = np.random.uniform(1.0-scale_range, 1.0+scale_range)
    seq = seq * s
    return seq


def random_shift_sequence(seq, shift_range=0.1):
    # Shift
    d = [np.random.uniform(-shift_range, shift_range), np.random.uniform(-shift_range, shift_range),
         np.random.uniform(-shift_range, shift_range)]
    d = np.tile(d, 22)
    for frame_idx in range(seq.shape[0]):
        seq[frame_idx, :] = seq[frame_idx, :] + d
    return seq
    # Xshift = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    # for frame in range(0, X.shape[1]):
    #     if X[0, frame, :].all() != 0:
    #         Xshift[0, frame, :] = X[0, frame, :] + d
    # Yshift = Y
    # XY_train.append((Xshift, Yshift))


def random_interpolate_sequence(seq):
    # TimeInterpolation
    new_seq = np.zeros_like(seq)
    for frame_idx in range(seq.shape[0]-1):
        r = np.random.uniform(0, 1)
        M = seq[frame_idx + 1, :] - seq[frame_idx, :]
        new_seq[frame_idx, :] = seq[frame_idx + 1, :] - M * r
    new_seq[seq.shape[0]-1] = seq[seq.shape[0]-1]
    return new_seq

    # Xtip = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    # for frame in range(0, X.shape[1] - 1):
    #     if X[0, frame + 1, :].all() != 0:
    #         r = np.random.uniform(0, 1)
    #         M = X[0, frame + 1, :] - X[0, frame, :]
    #         Xtip[0, frame, :] = X[0, frame + 1, :] - M * r
    # Ytip = Y
    # XY_train.append((Xtip, Ytip))


def random_noise_sequence(seq, noise_range=0.1):
    # Noise
    joint_range = np.array((range(22)))
    np.random.shuffle(joint_range)
    joint_index = joint_range[:4]
    n_1 = [np.random.uniform(-noise_range, noise_range), np.random.uniform(-noise_range, noise_range), np.random.uniform(-noise_range, noise_range)]
    n_2 = [np.random.uniform(-noise_range, noise_range), np.random.uniform(-noise_range, noise_range), np.random.uniform(-noise_range, noise_range)]
    n_3 = [np.random.uniform(-noise_range, noise_range), np.random.uniform(-noise_range, noise_range), np.random.uniform(-noise_range, noise_range)]
    n_4 = [np.random.uniform(-noise_range, noise_range), np.random.uniform(-noise_range, noise_range), np.random.uniform(-noise_range, noise_range)]
    for frame_idx in range(seq.shape[0]):
        seq[frame_idx, joint_index[0]:joint_index[0] + 3] += n_1
        seq[frame_idx, joint_index[1]:joint_index[1] + 3] += n_2
        seq[frame_idx, joint_index[2]:joint_index[2] + 3] += n_3
        seq[frame_idx, joint_index[3]:joint_index[3] + 3] += n_4
    return seq
    # for frame in range(0, X.shape[1]):
    #     if X[0, frame, :].all() != 0:
    #         x = X[0, frame, :]
    #         x[joint_index[0]:joint_index[0] + 3] += n_1
    #         x[joint_index[1]:joint_index[1] + 3] += n_2
    #         x[joint_index[2]:joint_index[2] + 3] += n_3
    #         x[joint_index[3]:joint_index[3] + 3] += n_4
    #         Xnoise[0, frame, :] = x
    # Ynoise = Y
    # XY_train.append((Xnoise, Ynoise))


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