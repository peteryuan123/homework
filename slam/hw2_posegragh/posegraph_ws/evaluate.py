from os import error
import matplotlib
import matplotlib.pyplot as plt  
import numpy as np
import math
import cv2

plt.style.use("ggplot") 

num_poses = 12

radian30 = math.pi / 6
dead_reckoning =  np.loadtxt("output/trajectory_dead_reckoning.txt")
loop_closure = np.loadtxt("output/trajectory_loop_closure.txt")
more_constraints = np.loadtxt("output/trajectory_more_constraints.txt")
gts = np.loadtxt("output/gt.txt")

dead_reckoning_t_error = 0.0
loop_closure_t_error = 0.0
more_constraints_t_error = 0.0

dead_reckoning_r_error = 0.0
loop_closure_r_error = 0.0
more_constraints_r_error = 0.0

for i in range(1, num_poses):
    gt_theta = i * radian30 % (2 * math.pi)
    if (gt_theta > math.pi):
        gt_theta = gt_theta - 2 * math.pi
    gt = np.array([math.cos(i * radian30), math.sin(i * radian30), (gt_theta)])

    dead_reckoning_t_error += np.linalg.norm(gt[:2] -  dead_reckoning[i][:2])
    loop_closure_t_error += np.linalg.norm(gt[:2] -  loop_closure[i][:2])
    more_constraints_t_error += np.linalg.norm(gt[:2] -  more_constraints[i][:2])

    dead_reckoning_r_error += np.linalg.norm(gt[2] -  dead_reckoning[i][2])
    loop_closure_r_error += np.linalg.norm(gt[2] -  loop_closure[i][2])
    more_constraints_r_error += np.linalg.norm(gt[2] -  more_constraints[i][2])


information_matrix = np.zeros((12, 12))

for i in range (12):
    information_matrix[i][(i+1) % 12] = 255
    information_matrix[(i+1) % 12][i] = 255

cv2.imwrite("output/information_loop_closure.png", information_matrix)

for i in range (10):
    information_matrix[i][i+2] = 255
    information_matrix[i+2][i] = 255

cv2.imwrite("output/information_more_constraints.png", information_matrix)



print(dead_reckoning_t_error, loop_closure_t_error, more_constraints_t_error)
print(dead_reckoning_r_error, loop_closure_r_error, more_constraints_r_error)

# plt.scatter(dead_reckoning[...,0], dead_reckoning[...,1])
plt.plot(dead_reckoning[...,0], dead_reckoning[...,1], linestyle=':',marker='v', label='dead_reckoning')

# plt.scatter(loop_closure[...,0], loop_closure[...,1])
plt.plot(loop_closure[...,0], loop_closure[...,1], linestyle='-.',marker='D', label='loop_closure')

plt.plot(more_constraints[...,0], more_constraints[...,1],  linestyle='--',marker='+', label='more_constraints')


# plt.scatter(gts[...,0], gts[...,1])
plt.plot(gts[...,0], gts[...,1], linestyle='-',marker='o', label='gt')

# plt.scatter(more_constraints[...,0], more_constraints[...,1])
plt.title("Trajectory")
# plt.xlabel("x")
# plt.ylabel("y")

ax = plt.gca()
ax.set_aspect(1)

# 把右边和上边的边框去掉
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# 把x轴的刻度设置为‘bottom'
# 把y轴的刻度设置为 ’left‘
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# 设置bottom对应到0点
# 设置left对应到0点
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

plt.legend(loc='upper right')

plt.show()


