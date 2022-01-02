import numpy as np
from scipy.spatial.transform import Rotation as srot
from scipy.optimize import least_squares

def ransac(dataset, model_estimator, model_verifier, inlier_th, niter, joint_type):
    best_model = None
    best_score = -np.inf
    best_inliers = None
    for i in range(niter):
        cur_model = model_estimator(dataset, joint_type)
        cur_score, cur_inliers = model_verifier(dataset, cur_model, inlier_th)
        if cur_score > best_score:
            best_model = cur_model
            best_inliers = cur_inliers
            best_score = cur_score
    best_model = model_estimator(dataset, joint_type, best_inliers)
    return best_model, best_inliers

def joint_transformation_estimator(dataset, joint_type, best_inliers=None):
    # dataset: dict, fields include source0, target0, nsource0,
    #     source1, target1, nsource1, joint_direction
    if best_inliers is None:
        sample_idx0 = np.random.randint(dataset["nsource0"], size=3)
        sample_idx1 = np.random.randint(dataset["nsource1"], size=3)
    else:
        sample_idx0 = best_inliers[0]
        sample_idx1 = best_inliers[1]

    source0 = dataset["source0"][sample_idx0, :]
    target0 = dataset["target0"][sample_idx0, :]
    source1 = dataset["source1"][sample_idx1, :]
    target1 = dataset["target1"][sample_idx1, :]
    # prescaling and centering
    scale0 = scale_pts(source0, target0)
    scale1 = scale_pts(source1, target1)
    scale0_inv = scale_pts(target0, source0)  # check if could simply take reciprocal
    scale1_inv = scale_pts(target1, source1)

    target0_scaled_centered = scale0_inv * target0
    target0_scaled_centered -= np.mean(target0_scaled_centered, 0, keepdims=True)
    source0_centered = source0 - np.mean(source0, 0, keepdims=True)

    target1_scaled_centered = scale1_inv * target1
    target1_scaled_centered -= np.mean(target1_scaled_centered, 0, keepdims=True)
    source1_centered = source1 - np.mean(source1, 0, keepdims=True)

    joint_points0 = np.ones_like(
        np.linspace(0, 1, num=np.min((source0.shape[0], source1.shape[0])) + 1)[
            1:
        ].reshape((-1, 1))
    ) * dataset["joint_direction"].reshape((1, 3))

    R0 = rotate_pts(source0_centered, target0_scaled_centered)
    R1 = rotate_pts(source1_centered, target1_scaled_centered)

    # T0 = np.mean(target0_scaled_centered.T - np.matmul(R0, source0_centered.T), 1)
    # T1 = np.mean(target0_scaled_centered.T - np.matmul(R0, source0_centered.T), 1)

    rotvec0 = srot.from_dcm(R0).as_rotvec()
    rotvec1 = srot.from_dcm(R1).as_rotvec()
    # print('initialize rotvec0 vs rotvec1: \n', rotvec0, rotvec1)
    if joint_type == 0:
        # 0 represents primatic
        # res = least_squares(
        #     objective_eval_t,
        #     np.hstack((T0, T1)),
        #     verbose=0,
        #     ftol=1e-4,
        #     method="lm",
        #     args=(
        #         source0_centered,
        #         target0_scaled_centered,
        #         source1_centered,
        #         target1_scaled_centered,
        #         joint_points0,
        #         R0,
        #         R1,
        #         1.0,
        #         1.0,
        #         False,
        #     ),
        # )
        res = least_squares(
            objective_eval_t,
            np.hstack((rotvec0, rotvec1)),
            verbose=0,
            ftol=1e-4,
            method="lm",
            args=(
                source0_centered,
                target0_scaled_centered,
                source1_centered,
                target1_scaled_centered,
                joint_points0,
                False,
            ),
        )
    elif joint_type == 1:
        # 1 represents revolute
        res = least_squares(
            objective_eval_r,
            np.hstack((rotvec0, rotvec1)),
            verbose=0,
            ftol=1e-4,
            method="lm",
            args=(
                source0_centered,
                target0_scaled_centered,
                source1_centered,
                target1_scaled_centered,
                joint_points0,
                False,
            ),
        )
    R0 = srot.from_rotvec(res.x[:3]).as_dcm()
    R1 = srot.from_rotvec(res.x[3:]).as_dcm()

    translation0 = np.mean(target0.T - scale0 * np.matmul(R0, source0.T), 1)
    translation1 = np.mean(target1.T - scale1 * np.matmul(R1, source1.T), 1)

    jtrans = dict()
    jtrans["rotation0"] = R0
    jtrans["scale0"] = scale0
    jtrans["translation0"] = translation0
    jtrans["rotation1"] = R1
    jtrans["scale1"] = scale1
    jtrans["translation1"] = translation1
    return jtrans


def joint_transformation_verifier(dataset, model, inlier_th):
    # dataset: dict, fields include source, target, nsource, ntarget
    # model: dict, fields include rotation, scale, translation
    res0 = (
        dataset["target0"].T
        - model["scale0"] * np.matmul(model["rotation0"], dataset["source0"].T)
        - model["translation0"].reshape((3, 1))
    )
    inliers0 = np.sqrt(np.sum(res0 ** 2, 0)) < inlier_th
    res1 = (
        dataset["target1"].T
        - model["scale1"] * np.matmul(model["rotation1"], dataset["source1"].T)
        - model["translation1"].reshape((3, 1))
    )
    inliers1 = np.sqrt(np.sum(res1 ** 2, 0)) < inlier_th
    score = (np.sum(inliers0) / res0.shape[0] + np.sum(inliers1) / res1.shape[0]) / 2
    return score, [inliers0, inliers1]

def objective_eval_r(params, x0, y0, x1, y1, joints, isweight=True):
    # params: [:3] R0, [3:] R1
    # x0: N x 3, y0: N x 3, x1: M x 3, y1: M x 3, R0: 1 x 3, R1: 1 x 3, joints: K x 3
    rotvec0 = params[:3].reshape((1, 3))
    rotvec1 = params[3:].reshape((1, 3))
    res0 = y0 - rotate_points_with_rotvec(x0, rotvec0)
    res1 = y1 - rotate_points_with_rotvec(x1, rotvec1)
    res_joint = rotate_points_with_rotvec(joints, rotvec0) - rotate_points_with_rotvec(
        joints, rotvec1
    )
    if isweight:
        res0 /= x0.shape[0]
        res1 /= x1.shape[0]
        res_joint /= joints.shape[0]
    return np.concatenate((res0, res1, res_joint), 0).ravel()

# The author replies that they don't use this optimization as their final optimization
# def objective_eval_t(
#     params, x0, y0, x1, y1, joints, R0, R1, scale0, scale1, isweight=True
# ):
#     # params: [0:3] t0, [3:6] t1;
#     # joints: K * 3
#     # rotvec0, rotvec1, scale0, scale1 solved from previous steps
#     R = R0
#     transvec0 = params[0:3].reshape((1, 3))
#     transvec1 = params[3:6].reshape((1, 3))
#     res0 = y0 - scale0 * np.matmul(x0, R0.T) - transvec0
#     res1 = y1 - scale1 * np.matmul(x1, R1.T) - transvec1
#     rot_u = np.matmul(joints, R.T)[0]
#     delta_trans = transvec0 - transvec1
#     cross_mat = np.array(
#         [[0, -rot_u[2], rot_u[1]], [rot_u[2], 0, -rot_u[0]], [-rot_u[1], rot_u[0], 0]]
#     )
#     res2 = np.matmul(delta_trans, cross_mat.T).reshape(1, 3)
#     # np.linspace(0, 1, num = np.min((x0.shape[0], x1.shape[0]))+1 )[1:].reshape((-1, 1))
#     res2 = np.ones((np.min((x0.shape[0], x1.shape[0])), 1)) * res2
#     if isweight:
#         res0 /= x0.shape[0]
#         res1 /= x1.shape[0]
#         res2 /= res2.shape[0]
#     return np.concatenate((res0, res1, res2), 0).ravel()

def objective_eval_t(
    params, x0, y0, x1, y1, joints, isweight=True, joint_type="prismatic"
):
    # params: [:3] R0, [3:] R1
    # x0: N x 3, y0: N x 3, x1: M x 3, y1: M x 3, R0: 1 x 3, R1: 1 x 3, joints: K x 3
    rotvec0 = params[:3].reshape((1, 3))
    rotvec1 = params[3:].reshape((1, 3))
    res0 = y0 - rotate_points_with_rotvec(x0, rotvec0)
    res1 = y1 - rotate_points_with_rotvec(x1, rotvec1)
    res_R = rotvec0 - rotvec1
    if isweight:
        res0 /= x0.shape[0]
        res1 /= x1.shape[0]
    return np.concatenate((res0, res1, res_R), 0).ravel()


def scale_pts(source, target):
    '''
    func: compute scaling factor between source: [N x 3], target: [N x 3]
    '''
    pdist_s = source.reshape(source.shape[0], 1, 3) - source.reshape(1, source.shape[0], 3)
    A = np.sqrt(np.sum(pdist_s**2, 2)).reshape(-1)
    pdist_t = target.reshape(target.shape[0], 1, 3) - target.reshape(1, target.shape[0], 3)
    b = np.sqrt(np.sum(pdist_t**2, 2)).reshape(-1)
    scale = np.dot(A, b) / (np.dot(A, A)+1e-6)
    return scale

def rotate_pts(source, target):
    '''
    func: compute rotation between source: [N x 3], target: [N x 3]
    '''
    # pre-centering
    source = source - np.mean(source, 0, keepdims=True)
    target = target - np.mean(target, 0, keepdims=True)
    M = np.matmul(target.T, source)
    U, D, Vh = np.linalg.svd(M, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    R = np.matmul(U, Vh)
    return R

def rotate_points_with_rotvec(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def rot_diff_degree(rot1, rot2):
    return rot_diff_rad(rot1, rot2) / np.pi * 180

def rot_diff_rad(rot1, rot2):
    return np.arccos( ( np.trace(np.matmul(rot1, rot2.T)) - 1 ) / 2 ) % (2*np.pi)

def optimize_with_kinematic(ins_ancsh, ins_npcs, num_parts, niter, choose_threshold):
    camcs_per_point = ins_ancsh["camcs_per_point"]
    joint_type = ins_ancsh["joint_type"]
    # Get the predicted segmentation and npcs from the NPCS model results
    pred_npcs_per_point = ins_npcs["pred_npcs_per_point"]
    pred_seg_per_point = ins_npcs["pred_seg_per_point"]
    part_per_point = np.argmax(pred_seg_per_point, axis=1)
    # Get the joint prediction from ANCSH model results
    pred_joint_cls_per_point = ins_ancsh["pred_joint_cls_per_point"]
    pred_axis_per_point = ins_ancsh["pred_axis_per_point"]
    # Get the gt pose
    gt_rt = ins_ancsh["gt_rt"]
    gt_scale = ins_ancsh["gt_scale"]

    # Get the point mask
    partIndex = []
    for i in range(num_parts):
        partIndex.append(np.where(part_per_point == i)[0])

    jointIndex = []
    # Joint 0 means there is no joint association
    # Get the joint association mask
    for i in range(1, num_parts):
        jointIndex.append(np.where(pred_joint_cls_per_point == i)[0])

    pred_scale = []
    pred_rt = []
    err_scale = []
    err_translation = []
    err_rotation = []
    
    for i in range(1, num_parts):
        # Calculate the part pose for each moving part (from NPCS to camera)
        data = dict()
        # Get the npcs and camera coordinate of the base part
        data["source0"] = pred_npcs_per_point[partIndex[0], :3]
        data["target0"] = camcs_per_point[partIndex[0]]
        data["nsource0"] = data["source0"].shape[0]
        # Get the npcs and camera coordinate of the moving part
        data["source1"] = pred_npcs_per_point[partIndex[i], 3 * i : 3 * (i + 1)]
        data["target1"] = camcs_per_point[partIndex[i]]
        data["nsource1"] = data["source1"].shape[0]
        # Get the constrained joint info
        data["joint_direction"] = np.median(pred_axis_per_point[jointIndex[i-1]], axis=0)

        assert joint_type >= 0

        best_model, best_inliers = ransac(
            data,
            joint_transformation_estimator,
            joint_transformation_verifier,
            choose_threshold,
            niter,
            joint_type[i],
        )

        if i == 1:
            # Record the pred things and error for base part
            rdiff = rot_diff_degree(best_model["rotation0"], gt_rt[0][:3, :3])
            tdiff = np.linalg.norm(best_model["translation0"] - gt_rt[0][:3, 3])
            sdiff = np.linalg.norm(best_model["scale0"] - gt_scale[0])
            pred_scale.append(best_model["scale0"])
            rt = np.zeros((4, 4))
            rt[:3, :3] = best_model["rotation0"]
            rt[:3, 3] = best_model["translation0"]
            rt[3, 3] = 1
            pred_rt.append(rt)
            err_rotation.append(rdiff)
            err_translation.append(tdiff)
            err_scale.append(sdiff)
        # Record the pred things and error for moving parts
        rdiff = rot_diff_degree(best_model["rotation1"], gt_rt[i][:3, :3])
        tdiff = np.linalg.norm(best_model["translation1"] - gt_rt[i][:3, 3])
        sdiff = np.linalg.norm(best_model["scale1"] - gt_scale[i])
        pred_scale.append(best_model["scale1"])
        rt = np.zeros((4, 4))
        rt[:3, :3] = best_model["rotation1"]
        rt[:3, 3] = best_model["translation1"]
        rt[3, 3] = 1
        pred_rt.append(rt)
        err_rotation.append(rdiff)
        err_translation.append(tdiff)
        err_scale.append(sdiff)

    return {
        "pred_npcs_scale": pred_scale,
        "pred_npcs_rt": pred_rt,
        "err_rotation": err_rotation,
        "err_translation": err_translation,
        "err_scale": err_scale,
    }
    

    