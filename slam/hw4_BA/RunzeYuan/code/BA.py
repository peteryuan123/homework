import numpy as np
from scipy.spatial.transform import Rotation as sciR
from scipy import linalg
import matplotlib.axes as axes
import matplotlib.pyplot as plt
import time

print("\n")
np.set_printoptions(suppress=True, linewidth=500)

K = np.array([[500, 0, 320],
              [0, 500, 240],
              [0, 0, 1]])

std_2d = 0.5
std_3d = 0.2
std_pose = 0.2

# out: list of (sciR.R, 3x1 t)
def generate_poses():
    poses = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            r = sciR.from_rotvec(np.array([0, 0, 0]))
            t = np.array([j * 0.1, i * 0.1, 0]).reshape(3, 1)
            poses.append((r, t))
    return poses


# out: 3xN array
def generate_pts3d(scale):
    pts = np.zeros((2, 25))
    for i in range(5):
        for j in range(5):
            pts[0, i * 5 + j] = 50 + j * 135
            pts[1, i * 5 + j] = 40 + i * 100
    homo_pts2d = np.concatenate([pts, np.ones((1, pts.shape[1]))])
    pts3d = np.linalg.inv(K) @ homo_pts2d
    return scale * pts3d


# out: list of 2xN array
def generate_measurements_in_frame(poses, pts3d):
    measurements = []
    for pose in poses:
        pts3d_in_frame = pose[0].as_matrix().T @ pts3d - pose[0].as_matrix().T @ pose[1]
        pts3d_normalized = pts3d_in_frame / pts3d_in_frame[2, :]
        measurements.append(pts3d_normalized[:2, :])
    return measurements


# out: list of (sciR.R, 3x1 t)
def add_noise_to_pose(poses):
    noisy_poses = []
    for pose in poses:
        noise_rt = np.random.normal(0, std_pose, (6, 1))
        noise_r = sciR.from_euler("zyx", noise_rt[:3].reshape(3))
        noise_t = noise_rt[3:]
        noisy_rt = (noise_r * pose[0], noise_r.as_matrix() @ pose[1] + noise_t)
        noisy_poses.append(noisy_rt)
    return noisy_poses


# out: list of 2xN array
def add_noise_to_measurements(measurements):
    noisy_measurements = []

    for measurement in measurements:
        image_pt = np.concatenate([measurement, np.ones((1,25))])
        image_pt = K @ image_pt
        image_pt[:2, :] += np.random.normal(0, std_2d, measurement.shape)
        noisy_measurement = (np.linalg.inv(K) @ image_pt)[:2, :]
        noisy_measurements.append(noisy_measurement)
    return noisy_measurements


# out: 3xN array
def add_noise_to_3d(pts3d):
    noisy_pts = pts3d + np.random.normal(0, std_3d, pts3d.shape)
    return noisy_pts

def get_skewed_matrix(vector3d):
    skew_matrix = np.zeros((3,3))
    skew_matrix[0, 1] = -vector3d[2]
    skew_matrix[0, 2] = vector3d[1]
    skew_matrix[1, 2] = -vector3d[0]
    skew_matrix[1, 0] = vector3d[2]
    skew_matrix[2, 0] = -vector3d[1]
    skew_matrix[2, 1] = vector3d[0]
    return skew_matrix

def f(r, t, p_global, r_change, t_change, p_change):
    p_inframe = sciR.from_rotvec(r + r_change).as_matrix().T @ (p_global + p_change - t - t_change)
    p_inframe = p_inframe / p_inframe[2, :]
    return -p_inframe[:2, :]

def numericalJacobian(pose_init, p_init, pose_change, p_change):
    jacobian = np.zeros((2,9))
    r_init, t_init = pose_init[3:].reshape(3), pose_init[:3]
    r_change, t_change = pose_change[3:].reshape(3), pose_change[:3]

    delta = 1e-5
    delta_p1 = np.array([delta, 0, 0]).reshape(3,1)
    delta_p2 = np.array([0, delta, 0]).reshape(3,1)
    delta_p3 = np.array([0, 0, delta]).reshape(3,1)
    delta_r1 = np.array([delta, 0, 0])
    delta_r2 = np.array([0, delta, 0])
    delta_r3 = np.array([0, 0, delta])
    delta_t1 = np.array([delta, 0, 0]).reshape(3,1)
    delta_t2 = np.array([0, delta, 0]).reshape(3,1)
    delta_t3 = np.array([0, 0, delta]).reshape(3,1)

    fx = f(r_init, t_init, p_init, r_change, t_change, p_change)

    fx_delta_p1 = f(r_init, t_init, p_init, r_change, t_change, p_change + delta_p1)
    fx_delta_p2 = f(r_init, t_init, p_init, r_change, t_change, p_change + delta_p2)
    fx_delta_p3 = f(r_init, t_init, p_init, r_change, t_change, p_change + delta_p3)
    fx_delta_r1 = f(r_init, t_init, p_init, r_change + delta_r1, t_change, p_change)
    fx_delta_r2 = f(r_init, t_init, p_init, r_change + delta_r2, t_change, p_change)
    fx_delta_r3 = f(r_init, t_init, p_init, r_change + delta_r3, t_change, p_change)
    fx_delta_t1 = f(r_init, t_init, p_init, r_change, t_change + delta_t1, p_change)
    fx_delta_t2 = f(r_init, t_init, p_init, r_change, t_change + delta_t2, p_change)
    fx_delta_t3 = f(r_init, t_init, p_init, r_change, t_change + delta_t3, p_change)
    jacobian[:,0] = ((fx_delta_p1 - fx) / delta).reshape(2)
    jacobian[:,1] = ((fx_delta_p2 - fx) / delta).reshape(2)
    jacobian[:,2] = ((fx_delta_p3 - fx) / delta).reshape(2)
    jacobian[:,3] = ((fx_delta_t1 - fx) / delta).reshape(2)
    jacobian[:,4] = ((fx_delta_t2 - fx) / delta).reshape(2)
    jacobian[:,5] = ((fx_delta_t3 - fx) / delta).reshape(2)
    jacobian[:,6] = ((fx_delta_r1 - fx) / delta).reshape(2)
    jacobian[:,7] = ((fx_delta_r2 - fx) / delta).reshape(2)
    jacobian[:,8] = ((fx_delta_r3 - fx) / delta).reshape(2)
    return jacobian

# [dp, dt, dr]
def calculateJacobian(pose_init, p_init, pose_change, p_change):
    jacobian = np.zeros((2, 6))

    r_init, t_init = pose_init[3:].reshape(3), pose_init[:3]
    r_change, t_change = pose_change[3:].reshape(3), pose_change[:3]
    cur_r = sciR.from_rotvec(r_init + r_change)
    cur_t = t_init + t_change
    cur_p = p_init + p_change

    # P'(X', Y', Z') = RP + t
    p_inframe = cur_r.as_matrix().T @ cur_p - cur_r.as_matrix().T @ cur_t

    # print(p_inframe)
    # (u, v) = (X' / Z', Y' / Z')
    # d(u,v) / dP'
    du_dpInframe = np.zeros((2, 3))
    du_dpInframe[0, 0] = 1 / p_inframe[2]
    du_dpInframe[0, 1] = 0
    du_dpInframe[0, 2] = -p_inframe[0] / (p_inframe[2] * p_inframe[2])
    du_dpInframe[1, 0] = 0
    du_dpInframe[1, 1] = 1 / p_inframe[2]
    du_dpInframe[1, 2] = -p_inframe[1] / (p_inframe[2] * p_inframe[2])
    # jacobian of P
    # d(u, v)/dP = d(u,v)/dP' * dP'/dP = d(u,v)/dP' * R
    du_dp = -du_dpInframe @ cur_r.as_matrix().T

    # jacobian of T
    # d（Tp）/dT
    dpinframe_dT = np.zeros((3, 6))
    dpinframe_dT[:3, :3] = -cur_r.as_matrix().T

    temp_p = sciR.from_rotvec(r_init).as_matrix().T @ (cur_p - cur_t)
    dpinframe_dT[:3, 3:] = get_skewed_matrix(temp_p)

    du_dT = -du_dpInframe @ dpinframe_dT
    # jacobian = np.concatenate([du_dp, du_dT], axis=1)
    return du_dp, du_dT

def recoverState(state_vec, view_size, pts_size):
    point_start_idx = view_size * 6
    pose = []
    pts = np.empty((3, 0))

    for pose_idx in range(view_size):
        T_state = state_vec[pose_idx * 6:(pose_idx + 1) * 6]
        r_state = sciR.from_rotvec(T_state[3:].reshape(3))
        t_state = T_state[:3].reshape(3, 1)
        pose.append((r_state, t_state))

    for point_idx in range(pts_size):
        point_state = state_vec[point_start_idx + point_idx * 3: point_start_idx + (point_idx + 1) * 3]
        pts = np.concatenate([pts, point_state], axis=1)

    return pose, pts

def computeError(init_X, X_change, view_size, pts_size):
    point_start_idx = view_size * 6
    reprojection = np.empty((0, 1))
    for pose_idx in range(view_size):
        for point_idx in range(pts_size):
            pose = init_X[pose_idx * 6:(pose_idx + 1) * 6]
            point = init_X[point_start_idx + point_idx * 3: point_start_idx + (point_idx + 1) * 3]

            pose_change = X_change[pose_idx * 6:(pose_idx + 1) * 6]
            p_change = X_change[point_start_idx + point_idx * 3: point_start_idx + (point_idx + 1) * 3]

            cur_r = pose[3:].reshape(3) + pose_change[3:].reshape(3)
            # reproject the current point to view
            cur_reprojection = sciR.from_rotvec(cur_r).as_matrix().T @ (point + p_change - pose[:3] - pose_change[:3])
            cur_reprojection = (cur_reprojection / cur_reprojection[2, :])[:2, :]
            reprojection = np.concatenate([reprojection, cur_reprojection])
    return reprojection

def gaussNewton(initialPoses, initialPoints, measurements, maxiter):
    view_size = len(initialPoses)
    pts_size = initialPoints.shape[1]
    point_start_idx = view_size * 6

    # concatenate state [pose(t,R), point]
    init_X = np.empty((0, 1))
    for (r, t) in initialPoses:
        init_X = np.concatenate(
            [init_X, t, r.as_rotvec().reshape(3, 1)])
    for col_idx in range(initialPoints.shape[1]):
        init_X = np.concatenate(
            [init_X, initialPoints[:, col_idx].reshape(3, 1)])

    # concatenate measurements [view0, view1, ...]
    objective = np.empty((0, 1))
    for measurement in measurements:
        objective = np.concatenate(
            [objective, measurement.T.reshape(pts_size*2, 1)])

    X_change = np.zeros(init_X.shape)

    lamd = 10e-3
    for i in range(maxiter):
        jacobian = np.zeros((2 * view_size * pts_size, init_X.shape[0]))
        reprojection = np.empty((0, 1))

        jtj = np.zeros((init_X.shape[0], init_X.shape[0]))
        jtr = np.zeros((init_X.shape[0], 1))

        for pose_idx in range(view_size):
            for point_idx in range(pts_size):
                cur_row = (pose_idx * pts_size + point_idx) * 2

                # get cur pose and point
                pose = init_X[pose_idx*6:(pose_idx+1)*6]
                point = init_X[point_start_idx + point_idx * 3: point_start_idx + (point_idx + 1) * 3]

                pose_change = X_change[pose_idx*6:(pose_idx+1)*6]
                p_change = X_change[point_start_idx + point_idx * 3: point_start_idx + (point_idx + 1) * 3]

                # calculate jacobian
                # dp, dT = calculateJacobian(pose, point, pose_change, p_change)
                # jacobian_analysis = np.concatenate([dp, dT], axis=1)

                jacobian_numerical = numericalJacobian(pose, point, pose_change, p_change)
                # if (np.sum(jacobian_analysis - jacobian_numerical) > 1):
                #     print("false")
                jacobian[cur_row:cur_row + 2, pose_idx * 6:(pose_idx + 1) * 6] = jacobian_numerical[:, 3:]
                jacobian[cur_row:cur_row + 2, point_start_idx + point_idx * 3: point_start_idx + (point_idx + 1) * 3] = jacobian_numerical[:, :3]


                cur_r = pose[3:].reshape(3) + pose_change[3:].reshape(3)
                # reproject the current point to view
                cur_reprojection = sciR.from_rotvec(cur_r).as_matrix().T @ (point + p_change - pose[:3] - pose_change[:3])
                cur_reprojection = (cur_reprojection / cur_reprojection[2, :])[:2, :]
                reprojection = np.concatenate([reprojection, cur_reprojection])

                J = np.zeros((2, init_X.shape[0]))
                J[:, pose_idx * 6:(pose_idx + 1) * 6] = jacobian_numerical[:, 3:]
                J[:, point_start_idx + point_idx * 3: point_start_idx + (point_idx + 1) * 3] = jacobian_numerical[:, :3]
                jtj += J.T @ J
                jtr += J.T @ (objective[pose_idx * pts_size * 2 + point_idx * 2 : pose_idx * pts_size * 2 + point_idx * 2 + 2, :] - cur_reprojection)


        # plt.matshow(jacobian)
        # plt.show()
        new_jtj = jtj + lamd * np.diag(np.diag(jtj))
        delta = linalg.solve(new_jtj, -jtr)
        temp = X_change + delta

        last_loss = np.linalg.norm(objective - reprojection, ord=2)
        cur_loss = np.linalg.norm(objective - computeError(init_X, temp, view_size, pts_size), ord=2)

        if (last_loss < cur_loss):
            print("recompute")
            lamd *= 10
            new_jtj = jtj + lamd * np.diag(np.diag(jtj))
            delta = linalg.solve(new_jtj, -jtr)
            X_change += delta
            cur_loss = np.linalg.norm(objective - computeError(init_X, X_change, view_size, pts_size), ord=2)
        else:
            lamd = max(lamd / 10, 10e-3)
            X_change += delta

        print("loss:" , last_loss)



    return recoverState(init_X + X_change, view_size, pts_size)
        # print(delta)

        # init_X = updateState(delta, init_X, view_size, pts_size)

def plotCoordinateSystem(axis, linewidth, pose, color):

    r = pose[0].as_matrix()
    axis_x = pose[1].reshape(3) + 0.5 * r[:, 0]
    axis_y = pose[1].reshape(3) + 0.5 * r[:, 1]
    axis_z = pose[1].reshape(3) + 0.5 * r[:, 2]
    # X-axis
    axis.plot([pose[1][0], axis_x[0]], [pose[1][1], axis_x[1]], zs=[pose[1][2], axis_x[2]], color = color, linewidth=linewidth)
    axis.plot([pose[1][0], axis_y[0]], [pose[1][1], axis_y[1]], zs=[pose[1][2], axis_y[2]], color = color, linewidth=linewidth)
    axis.plot([pose[1][0], axis_z[0]], [pose[1][1], axis_z[1]], zs=[pose[1][2], axis_z[2]], color = color, linewidth=linewidth)


def evaluateError(gposes, gpts3d, poses, pts3d):
    error_t = 0
    error_p = 0
    error_r = 0
    for i in range(len(gposes)):
        r, t = poses[i][0], poses[i][1]
        gr, gt = gposes[i][0], gposes[i][1]
        error_r += np.linalg.norm((r*gr.inv()).as_euler("zyx"))
        error_t += np.linalg.norm(t - gt, ord=2)

    for i in range(gpts3d.shape[1]):
        gp = gpts3d[:, i]
        p = pts3d[:, i]
        error_p += np.linalg.norm(gp - p, ord=2)

    return error_r, error_t, error_p

def main():
    # ground truth data
    poses = generate_poses()
    pts3d = generate_pts3d(2)
    measurements = generate_measurements_in_frame(poses, pts3d)

    # noisy data
    noisy_3d = add_noise_to_3d(pts3d)
    noisy_poses = add_noise_to_pose(poses)
    noisy_measurements = add_noise_to_measurements(measurements)

    # optimized data
    time_start = time.time()
    optimized_poses, optimized_pts3d = gaussNewton(noisy_poses, noisy_3d, noisy_measurements, 20)
    time_end = time.time()

    # transformed into ref frame
    ref_pose = optimized_poses[4]
    for i in range(len(optimized_poses)):
        new_r = ref_pose[0].inv() * optimized_poses[i][0]
        new_t =  ref_pose[0].as_matrix().T @ (optimized_poses[i][1] -  ref_pose[1])
        optimized_poses[i] = (new_r, new_t)
    optimized_pts3d = ref_pose[0].as_matrix().T @ (optimized_pts3d -  ref_pose[1])

    # evaluate error
    error_noise_r, error_noise_t, error_noise_p = evaluateError(poses, pts3d, noisy_poses, noisy_3d)
    error_opt_r, error_opt_t, error_opt_p = evaluateError(poses, pts3d, optimized_poses, optimized_pts3d)
    print("error_noise_r, error_noise_t, error_noise_p:", error_noise_r, error_noise_t, error_noise_p)
    print("error_opt_r, error_opt_t, error_opt_p:", error_opt_r, error_opt_t, error_opt_p)
    print("time:", time_end - time_start)
    # for pose in noisy_poses:
    #     plotCoordinateSystem(ax, 1, pose)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for pose in optimized_poses:
        plotCoordinateSystem(ax, 1, pose, "blue")
    for pose in poses:
        plotCoordinateSystem(ax, 1, pose, "green")
    for pose in noisy_poses:
        plotCoordinateSystem(ax, 1, pose, "red")

    for pt_idx in range(pts3d.shape[1]):
        ax.scatter(pts3d[0, pt_idx], pts3d[1, pt_idx], pts3d[2, pt_idx], color="green")

    for pt_idx in range(optimized_pts3d.shape[1]):
        ax.scatter(optimized_pts3d[0, pt_idx], optimized_pts3d[1, pt_idx], optimized_pts3d[2, pt_idx], color="blue")

    for pt_idx in range(noisy_3d.shape[1]):
        ax.scatter(noisy_3d[0, pt_idx], noisy_3d[1, pt_idx], noisy_3d[2, pt_idx], color="red")

    color_label = [("green", "GT"), ("red","noisy"), ("blue", "optimized")]
    for label in color_label:
        plt.scatter([], [], color=label[0],s=100, label=label[1])

    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()
