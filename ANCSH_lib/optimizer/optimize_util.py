import numpy as np

def ransac(dataset, model_estimator, model_verifier, inlier_th, niter=10000):
    best_model = None
    best_score = -np.inf
    best_inliers = None
    for i in range(niter):
        cur_model = model_estimator(dataset)
        cur_score, cur_inliers = model_verifier(dataset, cur_model, inlier_th)
        if cur_score > best_score:
            best_model = cur_model
            best_inliers = cur_inliers
            best_score = cur_score
    best_model = model_estimator(dataset, best_inliers)
    return best_model, best_inliers


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

    # Get the point mask
    partIndex = []
    for i in range(num_parts):
        partIndex.append(np.where(part_per_point == i)[0])

    jointIndex = []
    # Joint 0 means there is no joint association
    # Get the joint association mask
    for i in range(1, num_parts):
        jointIndex.append(np.where(pred_joint_cls_per_point == i)[0])

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

        best_model, best_inliers = ransac(
            data,
            joint_transformation_estimator,
            joint_transformation_verifier,
            choose_threshold,
            niter,
            joint_type,
        )
    