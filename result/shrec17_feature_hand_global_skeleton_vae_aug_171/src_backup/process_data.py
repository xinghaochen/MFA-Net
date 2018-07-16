import sys
sys.path.append('/home-2/jhou16@jhu.edu/.local/lib/python3.6/site-packages')
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import imageio

## In oder to reproduce the results
np.random.seed(seed=1234)

batch_size = 256
num_classes = 14
feat_dim = 66

def datagen(phase='train', tight_flag=1, split=1, augmentation=0):
  data_root = '/home-2/jhou16@jhu.edu/data/DHG2016'

  max_len_tight = 150
  max_len_loose = 280

  ## Generate the dict of effective biginning&end frame_index
  tight_dict = {}
  troncage_filedir = '/home-2/jhou16@jhu.edu/data/DHG2016/informations_troncage_sequences.txt'
  troncage_file = open(troncage_filedir,'r')
  for line in troncage_file:
    line_sp = line.split(' ')
    dir_code = [int(i) for i in line_sp[:4]]
    tight = [int(i) for i in line_sp[4:6]]
    tight_dict[str(dir_code)] = tight

  ## Generate Training data/Test data
  ## leave-one-subject-out cross-validation
  XY_train = []
  XY_test = []
  gestures = os.listdir(data_root)
  gestures.remove('informations_troncage_sequences.txt')
  for gst in gestures:
    dir_code = []
    dir_code.append(int(gst[8:]))
    fingers = os.listdir(os.path.join(data_root,gst))
    Y = np.zeros((1,num_classes))
    Y[0,int(gst[8:])-1] = 1
    for fg in fingers:
      dir_code.append(int(fg[7:]))
      subjects = os.listdir(os.path.join(data_root,gst,fg))
      for sb in subjects:
        if phase == 'train' and int(sb[8:]) == split:
          continue
        if phase == 'test' and int(sb[8:]) != split:
          continue
        dir_code.append(int(sb[8:]))
        essais = os.listdir(os.path.join(data_root,gst,fg,sb))
        for es in essais:
          dir_code.append(int(es[6:]))
          skeleton_filedir = os.path.join(data_root,gst,fg,sb,es,'skeleton_world.txt')

          skeleton_file = open(skeleton_filedir,'r')

          X_ = []
          palm_flag = 0
          for line in skeleton_file:
            line_sp = line.split(' ')
            x = [float(f) for f in line_sp]
            if palm_flag == 0:
              palm_pos = x[3:6]
              palm_flag = 1
            bias = palm_pos * 22
            X_.append(list(map(lambda x: x[0]-x[1], zip(x, bias))))
          Xloose_ = np.array(X_)
          Xloose = np.zeros((1,max_len_loose,feat_dim))
          Xloose[0,:Xloose_.shape[0],:] = Xloose_ 
          Xtight_ = Xloose_[tight_dict[str(dir_code)][0]:tight_dict[str(dir_code)][1]+1]
          if augmentation == 0:
            Xtight = np.zeros((1,max_len_tight,feat_dim))
          else:
            Xtight = np.zeros((1,max_len_loose,feat_dim))
          Xtight[0,:Xtight_.shape[0],:] = Xtight_ 

          if phase == 'train':
            if augmentation == 1:
              # Data Augmentation: Cropping
              XY_train.append((Xtight,Y))
              XY_train.append((Xloose,Y))
            elif tight_flag == 1:
              XY_train.append((Xtight,Y))
            else:
              XY_train.append((Xloose,Y))
          else:
            if tight_flag == 1:
              XY_test.append((Xtight,Y))
            else:
              XY_test.append((Xloose,Y))
          skeleton_file.close()

          dir_code.pop()
        dir_code.pop()
      dir_code.pop()
    dir_code.pop()

  ## Data Augmentation
  if augmentation == 1 and phase == 'train':
    XY_train_ = XY_train[:]
    indices = range(0,len(XY_train_))
    for ind in indices:
      X = XY_train_[ind][0]
      Y = XY_train_[ind][1]
      
      # Scale
      f = np.random.uniform(0.8,1.2)
      Xscale = X * f
      Yscale = Y
      XY_train.append((Xscale,Yscale))
      
      # Shift
      d = [np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1)]
      d = np.tile(d,22)
      Xshift = np.zeros((X.shape[0],X.shape[1],X.shape[2]))
      for frame in range(0,X.shape[1]):
        if X[0,frame,:].all() != 0:
          Xshift[0,frame,:] = X[0,frame,:] + d
      Yshift = Y
      XY_train.append((Xshift,Yshift))

      # TimeInterpolation
      Xtip = np.zeros((X.shape[0],X.shape[1],X.shape[2]))
      for frame in range(0,X.shape[1]-1):
        if X[0,frame+1,:].all() != 0:
          r = np.random.uniform(0,1)
          M = X[0,frame+1,:] - X[0,frame,:]
          Xtip[0,frame,:] = X[0,frame+1,:] - M*r
      Ytip = Y
      XY_train.append((Xtip,Ytip))

      # Noise
      Xnoise = np.zeros((X.shape[0],X.shape[1],X.shape[2]))
      joint_range = np.array((range(22)))
      np.random.shuffle(joint_range)
      joint_index = joint_range[:4]
      n_1 = [np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1)]
      n_2 = [np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1)]
      n_3 = [np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1)]
      n_4 = [np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1)]
      for frame in range(0,X.shape[1]):
        if X[0,frame,:].all() != 0:
          x = X[0,frame,:]
          x[joint_index[0]:joint_index[0]+3] += n_1
          x[joint_index[1]:joint_index[1]+3] += n_2
          x[joint_index[2]:joint_index[2]+3] += n_3
          x[joint_index[3]:joint_index[3]+3] += n_4
          Xnoise[0,frame,:] = x
      Ynoise = Y
      XY_train.append((Xnoise,Ynoise))
      
  if phase == 'train':
    batch_cnt = 0
    if augmentation == 0 and tight_flag == 1:
      X_train = np.zeros((batch_size,max_len_tight,feat_dim))
      Y_train = np.zeros((batch_size,num_classes))
    else:
      X_train = np.zeros((batch_size,max_len_loose,feat_dim))
      Y_train = np.zeros((batch_size,num_classes))
    while True:
      np.random.shuffle(XY_train)
      indices = range(0, len(XY_train))
      for ind in indices:
        X_train[batch_cnt] = XY_train[ind][0]
        Y_train[batch_cnt] = XY_train[ind][1]
        batch_cnt += 1
        if batch_cnt == batch_size:
          ret_X = X_train
          ret_Y = Y_train
          batch_cnt = 0
          if augmentation == 0 and tight_flag == 1:
            X_train = np.zeros((batch_size,max_len_tight,feat_dim))
            Y_train = np.zeros((batch_size,num_classes))
          else:
            X_train = np.zeros((batch_size,max_len_loose,feat_dim))
            Y_train = np.zeros((batch_size,num_classes))
          yield (ret_X,ret_Y)

  if phase == 'test':
    while True:
      np.random.shuffle(XY_test)
      indices = range(0, len(XY_test))
      for ind in indices:
        yield XY_test[ind]
  
  '''
  if phase == 'train':
    while True:
      np.random.shuffle(XY_train)
      indices = range(0, len(XY_train))
      for ind in indices:
        yield XY_train[ind]

  if phase == 'test':
    while True:
      np.random.shuffle(XY_test)
      indices = range(0, len(XY_test))
      for ind in indices:
        yield XY_test[ind]
  '''

def draw_skeleton(x,outdir,outname):
  if not os.path.exists(outdir):
    os.makedirs(outdir)

  fig = plt.figure()
  ax = Axes3D(fig)
  x = x.reshape(22,3)
  for joint_index in range(22):
    joint = x[joint_index]
    ax.scatter(joint[0],joint[1],joint[2],color='red')

  connectivity = [(0,1),(0,2),(2,3),(3,4),(4,5),(1,6),(6,7),(7,8),(8,9),(1,10),(10,11),(11,12),(12,13),(1,14),(14,15),(15,16),(16,17),(1,18),(18,19),(19,20),(20,21)]
  for bone in connectivity:
    p = bone[0]
    n = bone[1]
    ax.plot([x[p,0],x[n,0]],[x[p,1],x[n,1]],[x[p,2],x[n,2]])

  ax.set_xlabel('X')
  ax.set_xlim(-0.5,0.5)
  ax.set_ylabel('Y')
  ax.set_ylim(-0.5,0.5)
  ax.set_zlabel('Z')
  ax.set_zlim(-0.5,0.5)
  ax.view_init(elev=-90., azim=90)

  plt.savefig(outdir+outname+".png")


def draw_input(gst,fg,sb,es,tight_flag=0):
  outdir = '/home-2/jhou16@jhu.edu/scratch/DHG/visualize/input/gst{}_fg{}_sb{}_es{}/'.format(gst,fg,sb,es)
  data_dir = '/home-2/jhou16@jhu.edu/data/DHG2016/gesture_{}/finger_{}/subject_{}/essai_{}/skeleton_world.txt'.format(gst,fg,sb,es)
  skeleton_file = open(data_dir,'r')
  dir_code = [gst,fg,sb,es]

  tight_dict = {}
  if tight_flag == 1:
    troncage_filedir = '/home-2/jhou16@jhu.edu/data/DHG2016/informations_troncage_sequences.txt'
    troncage_file = open(troncage_filedir,'r')
    for line in troncage_file:
      line_sp = line.split(' ')
      dir_code = [int(i) for i in line_sp[:4]]
      tight = [int(i) for i in line_sp[4:6]]
      tight_dict[str(dir_code)] = tight

  frame_cnt = 0
  for line in skeleton_file:
    frame_cnt += 1
    if tight_flag == 1:
      if frame_cnt < tight_dict[str(dir_code)][0] or frame_cnt > tight_dict[str(dir_code)][1]:
        continue
    line_sp = line.split(' ')
    x = [float(f) for f in line_sp]
    x = np.array(x)
    draw_skeleton(x,outdir,str(frame_cnt))

  skeleton_pngs = os.listdir(outdir)
  skeleton_pngs = sorted(skeleton_pngs,key=lambda x:int(x[:x.find('.')]))
  skeleton_gif = []
  for fn in skeleton_pngs:
    skeleton_gif.append(imageio.imread(outdir+fn))
  imageio.mimsave(outdir+"gst{}_fg{}_sb{}_es{}.gif".format(gst,fg,sb,es),skeleton_gif)


if __name__ == "__main__":
  datagen()
