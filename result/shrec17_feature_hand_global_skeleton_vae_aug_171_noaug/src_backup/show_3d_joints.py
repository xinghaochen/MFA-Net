# display the joints in 3D
#   Xinghao Chen, 27 Mar, 2017

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def show_3d_joints(joint, show_connection, show_id):
    J = joint.shape[0]
    xmax = np.max(joint[:, 0])
    xmin = np.min(joint[:, 0])
    ymax = np.max(joint[:, 1])
    ymin = np.min(joint[:, 1])
    zmax = np.max(joint[:, 2])
    zmin = np.min(joint[:, 2])

    # display joints
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax = fig.gca(projection='3d')
    ax = Axes3D(fig)
    color = 'b'
    ax.scatter(joint[:, 0], joint[:, 1], joint[:, 2], c=color, marker='.', s=10)
    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    ax.set_zlim3d(zmin, zmax)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    if show_connection:
        lines = [[0, 2, 3, 4, 0, 1, 6, 7, 8, 1,  10, 11, 12, 1,  14, 15, 16, 1,  18, 19, 20],
                [2, 3, 4, 5, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
        # draw lines
        for i in range(len(lines[0])):
            id1 = lines[0][i]
            id2 = lines[1][i]
            ax.plot([joint[id1, 0], joint[id2, 0]],
                    [joint[id1, 1], joint[id2, 1]],
                    [joint[id1, 2], joint[id2, 2]], c=color)
    if show_id:
        for i in range(joint.shape[0]):
            ax.text(joint[i, 0], joint[i, 1], joint[i, 2], str(i))
    ax.view_init(azim=0, elev=0)
    #plt.pause(1000)
    plt.waitforbuttonpress()
    #raw_input()


