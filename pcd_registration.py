import open3d as o3d
import copy
import numpy as np
import os
import trimesh
import cv2
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth

PATH_SOURCE = "produced_mesh_ninety_imgs/scaled_pcd_dataset.ply"
PATH_TARGET = "../dataset/exported_blades_v3/28_01_2024_09_55/med_scaled.ply"
BASE_DIR = "../dataset/exported_blades_v3/28_01_2024_09_55"


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        # zoom=0.4559,
        # # front=[0.6452, -0.3036, -0.7011],
        # # lookat=[1.9892, 2.0208, 1.8945],
        # # up=[-0.2779, -0.9482, 0.1556],
    )


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")

    source = o3d.io.read_point_cloud(PATH_SOURCE)
    target = o3d.io.read_point_cloud(PATH_TARGET)
    trans_init = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True
        ),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=False
        ),
    )
    return result


voxel_size = 0.005
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
    voxel_size
)
result_ransac = execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
)
print(result_ransac)
draw_registration_result(source_down, target_down, result_ransac.transformation)

result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size)
print(result_icp)
draw_registration_result(source, target, result_icp.transformation)
print(result_icp.transformation)

transformed_source = source.transform(result_icp.transformation)
transformed_source_pts = np.asarray(transformed_source.points)

mesh_path = os.path.join(BASE_DIR, "merged_blade_mapping_big.obj")
submeshes = trimesh.load_mesh(mesh_path).split()

submeshes_centers = []
submeshes_radii = []
intersections = [[] for _ in range(len(submeshes))]

submesh_points = []
for submesh in submeshes:
    mask = submesh.contains(transformed_source_pts)
    xyz = transformed_source_pts[mask]
    submesh_points.append(xyz)

for submesh in submeshes:
    center = np.mean(submesh.vertices, axis=0)
    radius = ((submesh.vertices - center) ** 2).sum(axis=1).max() ** 0.5
    submeshes_centers.append(center)
    submeshes_radii.append(radius)


for i in range(len(submeshes)):
    ci = submeshes_centers[i]
    ri = submeshes_radii[i]
    for j in range(len(submeshes)):
        if i == j:
            continue
        cj = submeshes_centers[j]
        rj = submeshes_radii[j]

        if np.linalg.norm(ci - cj) < ri + rj:
            intersections[i].append(j)


for i, submesh in enumerate(submeshes):
    mask = submesh.contains(transformed_source_pts)
    xyz_original = transformed_source_pts[mask]

    points_to_keep = np.ones(len(xyz_original), dtype=bool)

    for intersecting_idx in intersections[i]:
        intersecting_submesh = submeshes[intersecting_idx]
        intersecting_mask = intersecting_submesh.contains(xyz_original)
        points_to_keep = points_to_keep & ~intersecting_mask

    xyz_filtered = xyz_original[points_to_keep]

    if len(xyz_filtered) == 0:
        print(f"Submesh {i}: No points remaining after removing intersections")
        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(xyz_original)
        original_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
        print(f"Submesh {i}: Original points ({len(xyz_original)}) - All removed")
        o3d.visualization.draw_geometries([original_pcd])
        continue

    bounds = xyz_original.max(axis=0) - xyz_original.min(axis=0)
    x_offset = bounds[0] * 1.5

    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(xyz_original)
    original_pcd.paint_uniform_color([0.0, 0.0, 1.0])

    xyz_filtered_offset = xyz_filtered.copy()
    xyz_filtered_offset[:, 0] += x_offset

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(xyz_filtered_offset)
    filtered_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    print(
        f"Submesh {i}: Left (Blue) = Original ({len(xyz_original)}), Right (Red) = After removal ({len(xyz_filtered)})"
    )
    o3d.visualization.draw_geometries([original_pcd, filtered_pcd])

    rgb = np.asarray(transformed_source.colors, dtype=np.float32)[mask][points_to_keep]

    rgb_u8 = (rgb * 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    green_mask = (H >= 35) & (H <= 85) & (S >= 60) & (V >= 40)
    if len(xyz_filtered[green_mask]) == 0:
        # No green points probably a double cutter.
        continue

    db = DBSCAN(eps=0.008, min_samples=40).fit(xyz_filtered[green_mask])
    labels = db.labels_
    is_outlier = labels == -1

    green_inliers_mask = np.zeros(len(xyz_filtered), dtype=bool)
    green_indices = np.where(green_mask)[0]
    green_inliers_indices = green_indices[~is_outlier]
    green_inliers_mask[green_inliers_indices] = True

    green_xyz_filtered = xyz_filtered[green_inliers_mask]

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(xyz_filtered)

    colors = np.zeros((len(xyz_filtered), 3), dtype=np.float32)

    colors[green_inliers_mask] = [0.0, 1.0, 0.0]
    colors[~green_inliers_mask] = [0.7, 0.7, 0.7]
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Submesh {i}: Green detection on filtered points")
    o3d.visualization.draw_geometries([filtered_pcd])

    if len(green_xyz_filtered) == 0:
        continue
    green_pcd = o3d.geometry.PointCloud()
    green_pcd.points = o3d.utility.Vector3dVector(green_xyz_filtered)

    hull, _ = green_pcd.compute_convex_hull()

    target_pts = np.asarray(target.points)
    target_in_submesh_mask = submesh.contains(target_pts)
    target_pts_in_submesh = target_pts[target_in_submesh_mask]

    if len(target_pts_in_submesh) == 0:
        print(f"Submesh {i}: No target points found in submesh")
        continue

    hull_vertices = np.asarray(hull.vertices)
    hull_triangles = np.asarray(hull.triangles)
    hull_mesh = trimesh.Trimesh(vertices=hull_vertices, faces=hull_triangles)

    target_inside_hull_mask = hull_mesh.contains(target_pts_in_submesh)
    target_points_inside_hull = target_pts_in_submesh[target_inside_hull_mask]

    # If no target points found inside hull, perform ICP alignment
    if len(target_points_inside_hull) == 0:
        print(f"Submesh {i}: No target points inside hull, performing ICP alignment...")

        source_submesh_pcd = o3d.geometry.PointCloud()
        source_submesh_pcd.points = o3d.utility.Vector3dVector(xyz_filtered)

        target_submesh_pcd_icp = o3d.geometry.PointCloud()
        target_submesh_pcd_icp.points = o3d.utility.Vector3dVector(
            target_pts_in_submesh
        )

        distance_threshold = 0.01
        icp_result = o3d.pipelines.registration.registration_icp(
            source_submesh_pcd,
            target_submesh_pcd_icp,
            distance_threshold,
            np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        print(
            f"Submesh {i}: ICP fitness: {icp_result.fitness:.4f}, RMSE: {icp_result.inlier_rmse:.6f}"
        )

        hull_transformed = copy.deepcopy(hull)
        hull_transformed.transform(icp_result.transformation)

        green_xyz_transformed = np.asarray(green_pcd.points).copy()
        green_xyz_homogeneous = np.hstack(
            [green_xyz_transformed, np.ones((len(green_xyz_transformed), 1))]
        )
        green_xyz_transformed = (icp_result.transformation @ green_xyz_homogeneous.T).T[
            :, :3
        ]

        hull_vertices_transformed = np.asarray(hull_transformed.vertices)
        hull_triangles = np.asarray(hull_transformed.triangles)
        hull_mesh_transformed = trimesh.Trimesh(
            vertices=hull_vertices_transformed, faces=hull_triangles
        )

        target_inside_hull_mask = hull_mesh_transformed.contains(target_pts_in_submesh)
        target_points_inside_hull = target_pts_in_submesh[target_inside_hull_mask]

        print(
            f"Submesh {i}: After ICP alignment, found {len(target_points_inside_hull)} target points inside hull"
        )

        source_vis = copy.deepcopy(source_submesh_pcd)
        source_vis.transform(icp_result.transformation)
        source_vis.paint_uniform_color([0.0, 1.0, 0.0])  # Green for aligned source

        target_vis = copy.deepcopy(target_submesh_pcd_icp)
        target_vis.paint_uniform_color([0.0, 0.0, 1.0])  # Blue for target

        print(
            f"Submesh {i}: Showing ICP alignment - Green: aligned source, Blue: target"
        )
        o3d.visualization.draw_geometries([source_vis, target_vis])

    target_submesh_pcd = o3d.geometry.PointCloud()
    target_submesh_pcd.points = o3d.utility.Vector3dVector(target_pts_in_submesh)

    colors = np.zeros((len(target_pts_in_submesh), 3), dtype=np.float32)
    colors[target_inside_hull_mask] = [1.0, 0.0, 0.0]  # Red for points inside hull
    colors[~target_inside_hull_mask] = [0.7, 0.7, 0.7]  # Gray for other points

    target_submesh_pcd.colors = o3d.utility.Vector3dVector(colors)

    print(
        f"Submesh {i}: Found {len(target_points_inside_hull)} target points inside hull (colored red) out of {len(target_pts_in_submesh)} total target points in submesh"
    )
    o3d.visualization.draw_geometries([target_submesh_pcd])
