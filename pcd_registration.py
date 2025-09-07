import open3d as o3d
import copy
import numpy as np
import os
import trimesh
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist


PATH_SOURCE = "produced_mesh_ninety_imgs/scaled_pcd_dataset.ply"
PATH_TARGET = "../dataset/exported_blades_v3/28_01_2024_09_55/med_scaled.ply"
BASE_DIR = "../dataset/exported_blades_v3/28_01_2024_09_55"
# PATH_SOURCE = "produced_mesh_3/scaled_pcd_dataset.ply"
# PATH_TARGET = "../dataset/exported_blades_v3/01_02_2024_11_21/med_scaled.ply"
# BASE_DIR = "../dataset/exported_blades_v3/01_02_2024_11_21"


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
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


def refine_registration(source, target, distance_threshold):
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

distance_threshold = voxel_size * 0.4
result_icp = refine_registration(source, target, distance_threshold)
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


for i, submesh in enumerate(submeshes):
    mask = submesh.contains(transformed_source_pts)
    xyz_original = transformed_source_pts[mask]

    rgb_original = np.asarray(transformed_source.colors, dtype=np.float32)[mask]

    rgb_u8_original = (rgb_original * 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8_original.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(
        -1, 3
    )

    H, S, V = (
        hsv[:, 0],
        hsv[:, 1],
        hsv[:, 2],
    )

    green_mask_original = (H >= 35) & (H <= 85) & (S >= 60) & (V >= 40)

    if np.sum(green_mask_original) > 0:
        green_points_original = xyz_original[green_mask_original]

        print(f"Submesh {i}: Displaying original point cloud with green colored points")
        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(xyz_original)

        original_pcd_colors = np.zeros((len(xyz_original), 3), dtype=np.float32)
        original_pcd_colors[green_mask_original] = [0.0, 1.0, 0.0]
        original_pcd_colors[~green_mask_original] = [0.7, 0.7, 0.7]

        original_pcd.colors = o3d.utility.Vector3dVector(original_pcd_colors)

        o3d.visualization.draw_geometries([original_pcd])

        print(
            f"Submesh {i}: Applying DBSCAN clustering to {len(green_points_original)} green points from original"
        )

        db = DBSCAN(eps=0.008, min_samples=40).fit(green_points_original)
        cluster_labels = db.labels_

        is_outlier = cluster_labels == -1
        green_points_inliers = green_points_original[~is_outlier]
        cluster_labels_inliers = cluster_labels[~is_outlier]

        n_clusters = len(np.unique(cluster_labels_inliers))
        n_outliers = np.sum(is_outlier)

        print(f"Submesh {i}: Found {n_clusters} green clusters, {n_outliers} outliers")

        if len(green_points_inliers) > 0:
            green_colors = np.zeros((len(green_points_inliers), 3))
            unique_labels = np.unique(cluster_labels_inliers)

            cluster_info = []
            spheres = []

            for idx, label in enumerate(unique_labels):
                cluster_mask = cluster_labels_inliers == label
                cluster_points = green_points_inliers[cluster_mask]

                np.random.seed(42)
                color = np.random.rand(3)
                green_colors[cluster_mask] = color

                # Calculate cluster center
                cluster_center = np.mean(cluster_points, axis=0)

                # Calculate diameter (maximum pairwise distance)
                distances = pdist(cluster_points)
                diameter = np.max(distances)
                radius = diameter / 2.0

                cluster_info.append(
                    {
                        "center": cluster_center,
                        "radius": radius,
                    }
                )

                # Create sphere at cluster center with radius
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                sphere.translate(cluster_center)
                sphere.paint_uniform_color(color)
                spheres.append(sphere)

                print(
                    f"Submesh {i}, Cluster {label}: Center={cluster_center}, Diameter={diameter:.4f}, Radius={radius:.4f}, Points={len(cluster_points)}"
                )

            context_pcd = o3d.geometry.PointCloud()
            context_pcd.points = o3d.utility.Vector3dVector(xyz_original)

            context_colors = np.full((len(xyz_original), 3), 0.7)

            # Map cluster colors back to original green points
            green_indices = np.where(green_mask_original)[0]
            inlier_indices = green_indices[~is_outlier]

            for i_inlier, original_idx in enumerate(inlier_indices):
                context_colors[original_idx] = green_colors[i_inlier]

            context_pcd.colors = o3d.utility.Vector3dVector(context_colors)

            geometries = [context_pcd] + spheres

            o3d.visualization.draw_geometries(geometries)

        else:
            print(f"Submesh {i}: All green points were outliers")
            continue
    else:
        print(f"Submesh {i}: No green points found in original points")
        continue

    points_to_keep = np.ones(len(xyz_original), dtype=bool)

    # Find the sphere closest to the submesh center
    if len(cluster_info) > 0:
        submesh_center = np.mean(submesh.vertices, axis=0)

        # Calculate distances from submesh center to each sphere center
        distances_to_submesh = []
        for info in cluster_info:
            distance = np.linalg.norm(info["center"] - submesh_center)
            distances_to_submesh.append(distance)

        # Find the closest sphere
        closest_sphere_idx = np.argmin(distances_to_submesh)
        closest_sphere = cluster_info[closest_sphere_idx]

        print(
            f"Submesh {i}: Closest sphere to submesh center is at {closest_sphere['center']} with radius {closest_sphere['radius']:.4f}"
        )

        sphere_center = closest_sphere["center"]
        sphere_radius = closest_sphere["radius"]

        distances_to_sphere = np.linalg.norm(xyz_original - sphere_center, axis=1)
        points_to_keep = distances_to_sphere <= sphere_radius

        print(
            f"Submesh {i}: Keeping {np.sum(points_to_keep)} out of {len(xyz_original)} points inside closest sphere"
        )
    else:
        print(f"Submesh {i}: No spheres found, keeping all points")

    xyz_filtered = xyz_original[points_to_keep]

    green_pcd = o3d.geometry.PointCloud()
    green_pcd.points = o3d.utility.Vector3dVector(xyz_filtered)

    hull, _ = green_pcd.compute_convex_hull()

    target_pts = np.asarray(target.points)
    target_in_submesh_mask = submesh.contains(target_pts)
    target_pts_in_submesh = target_pts[target_in_submesh_mask]

    source_submesh_pcd_icp = o3d.geometry.PointCloud()
    source_submesh_pcd_icp.points = o3d.utility.Vector3dVector(xyz_original)

    target_submesh_pcd_icp = o3d.geometry.PointCloud()
    target_submesh_pcd_icp.points = o3d.utility.Vector3dVector(target_pts_in_submesh)

    distance_threshold = 0.01

    icp_result = o3d.pipelines.registration.registration_icp(
        source_submesh_pcd_icp,
        target_submesh_pcd_icp,
        distance_threshold,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    print(
        f"Submesh {i}: ICP fitness: {icp_result.fitness:.4f}, RMSE: {icp_result.inlier_rmse:.6f}"
    )

    hull.transform(icp_result.transformation)

    green_xyz_transformed = np.asarray(green_pcd.points).copy()
    green_xyz_homogeneous = np.hstack(
        [green_xyz_transformed, np.ones((len(green_xyz_transformed), 1))]
    )
    green_xyz_transformed = (green_xyz_homogeneous @ icp_result.transformation)[:, :3]

    hull_vertices_transformed = np.asarray(hull.vertices)
    hull_triangles = np.asarray(hull.triangles)
    hull_mesh_transformed = trimesh.Trimesh(
        vertices=hull_vertices_transformed, faces=hull_triangles
    )

    target_inside_hull_mask = hull_mesh_transformed.contains(target_pts_in_submesh)
    target_points_inside_hull = target_pts_in_submesh[target_inside_hull_mask]

    print(
        f"Submesh {i}: After ICP alignment, found {len(target_points_inside_hull)} target points inside hull"
    )

    source_vis = copy.deepcopy(source_submesh_pcd_icp)
    source_vis.transform(icp_result.transformation)
    source_vis.paint_uniform_color([1.0, 0.0, 0.0])

    target_vis = copy.deepcopy(target_submesh_pcd_icp)
    target_vis.paint_uniform_color([0.0, 0.0, 1.0])

    print(f"Submesh {i}: Showing ICP alignment - Red: aligned source, Blue: target")
    o3d.visualization.draw_geometries([source_vis, target_vis])

    target_submesh_pcd = o3d.geometry.PointCloud()
    target_submesh_pcd.points = o3d.utility.Vector3dVector(target_pts_in_submesh)

    colors = np.zeros((len(target_pts_in_submesh), 3), dtype=np.float32)
    colors[target_inside_hull_mask] = [1.0, 0.0, 0.0]
    colors[~target_inside_hull_mask] = [0.7, 0.7, 0.7]

    target_submesh_pcd.colors = o3d.utility.Vector3dVector(colors)

    print(
        f"Submesh {i}: Found {len(target_points_inside_hull)} target points inside hull (colored red) out of {len(target_pts_in_submesh)} total target points in submesh"
    )
    o3d.visualization.draw_geometries([target_submesh_pcd])
