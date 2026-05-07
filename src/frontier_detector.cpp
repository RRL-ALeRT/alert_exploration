#include "alert_exploration/frontier_detector.hpp"
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <algorithm>
#include <limits>

FrontierDetector::FrontierDetector(){
  voxel_size = 0.1;
  neighbor_radius = 0.5;
  min_neighbors = 5;
  edge_offset_ratio = 0.35;
  min_cluster_size_m = 0.3;
}

std::vector<FrontierCandidate> FrontierDetector::detect(const PointCloudT::Ptr &cloud){
  std::vector<FrontierCandidate> out;
  if (!cloud || cloud->points.empty()) return out;

  // downsample
  PointCloudT::Ptr ds(new PointCloudT());
  pcl::VoxelGrid<PointT> vg;
  vg.setInputCloud(cloud);
  vg.setLeafSize(voxel_size, voxel_size, voxel_size);
  vg.filter(*ds);

  if (ds->points.empty()) return out;

  // build kd tree
  pcl::KdTreeFLANN<PointT> tree;
  tree.setInputCloud(ds);

  std::vector<int> boundary_indices;
  boundary_indices.reserve(ds->points.size());

  for (size_t i=0;i<ds->points.size();++i){
    std::vector<int> neigh;
    std::vector<float> dists;
    tree.radiusSearch(ds->points[i], neighbor_radius, neigh, dists);
    if ((int)neigh.size() < min_neighbors) continue;
    // centroid
    Eigen::Vector4f centroid(0,0,0,0);
    for (int idx : neigh) centroid += ds->points[idx].getVector4fMap();
    centroid /= (float)neigh.size();
    double offset = (ds->points[i].getVector4fMap() - centroid).norm();
    if (offset > (edge_offset_ratio * neighbor_radius)) boundary_indices.push_back(static_cast<int>(i));
  }

  if (boundary_indices.empty()) return out;

  PointCloudT::Ptr boundary(new PointCloudT());
  pcl::copyPointCloud(*ds, boundary_indices, *boundary);

  // clustering
  pcl::search::KdTree<PointT>::Ptr btree(new pcl::search::KdTree<PointT>());
  btree->setInputCloud(boundary);
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(neighbor_radius * 1.5);
  ec.setMinClusterSize(min_neighbors);
  ec.setMaxClusterSize(250000);
  ec.setSearchMethod(btree);
  ec.setInputCloud(boundary);
  ec.extract(cluster_indices);

  for (auto &ci : cluster_indices){
    // compute bounding size (manual min/max because PCL overloads vary)
    PointT minp, maxp;
    minp.x = minp.y = minp.z = std::numeric_limits<float>::max();
    maxp.x = maxp.y = maxp.z = std::numeric_limits<float>::lowest();
    for (int idx : ci.indices){
      const auto &pt = boundary->points[idx];
      if (pt.x < minp.x) minp.x = pt.x;
      if (pt.y < minp.y) minp.y = pt.y;
      if (pt.z < minp.z) minp.z = pt.z;
      if (pt.x > maxp.x) maxp.x = pt.x;
      if (pt.y > maxp.y) maxp.y = pt.y;
      if (pt.z > maxp.z) maxp.z = pt.z;
    }
    double bbox = std::hypot(maxp.x - minp.x, maxp.y - minp.y);
    if (bbox < min_cluster_size_m) continue;
    // medoid
    Eigen::Vector4f center(0,0,0,0);
    for (int idx : ci.indices) center += boundary->points[idx].getVector4fMap();
    center /= (float)ci.indices.size();
    double bestd = 1e12; int besti = ci.indices[0];
    for (int idx : ci.indices){
      double d = (boundary->points[idx].getVector4fMap() - center).norm();
      if (d < bestd){ bestd = d; besti = idx; }
    }
    FrontierCandidate fc;
    fc.medoid = boundary->points[besti];
    fc.size = static_cast<int>(ci.indices.size());
    out.push_back(fc);
  }

  // sort by size descending
  std::sort(out.begin(), out.end(), [](const FrontierCandidate &a, const FrontierCandidate &b){ return a.size > b.size; });
  return out;
}
