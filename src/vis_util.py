from mayavi import mlab
import numpy as np

lines = [[0, 2, 3, 4, 0, 1, 6, 7, 8, 1, 10, 11, 12, 1, 14, 15, 16, 1, 18, 19, 20],
         [2, 3, 4, 5, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]

def plot_hand_pose(pose_xyz, filename):
    # Create a mayavi window
    #mlab.close(2)
    mlab.figure(2,size=(600,600))
    mlab.clf()

    # draw hand joints
    mlab.points3d(pose_xyz[:, 0], pose_xyz[:, 1], pose_xyz[:, 2], scale_factor=0.04, color=(0, 0, 1), mode='sphere',
                  opacity=1)
    # draw bones
    for i in range(len(lines[0])):
        x = lines[0][i]
        y = lines[1][i]
        mlab.plot3d([pose_xyz[x, 0], pose_xyz[y, 0]], [pose_xyz[x, 1], pose_xyz[y, 1]],
                 [pose_xyz[x, 2], pose_xyz[y, 2]], line_width=1.0, color=(1, 0, 0), tube_radius=0.008)

    # Export the model to X3D and WRL
    # mlab.savefig('{}/png_result_{}.png'.format(result_dir, file_basename))
    mlab.savefig(filename)

    mlab.show()

if __name__ == "__main__":
    i = 18773
    pose = np.loadtxt('visualization/vae/vae_pose_{}.txt'.format(i))
    pose_gt = np.loadtxt('visualization/vae/vae_pose_{}_gt.txt'.format(i))
    plot_hand_pose(pose, 'visualization/vae/vae_pose_{}.x3d'.format(i))
    plot_hand_pose(pose_gt, 'visualization/vae/vae_pose_{}_gt.x3d'.format(i))