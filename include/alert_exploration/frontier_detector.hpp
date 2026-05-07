#pragma once

#include <vector>
#include <memory>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;

struct FrontierCandidate {
  PointT medoid;
  int size;
};

class FrontierDetector {
public:
  FrontierDetector();
  std::vector<FrontierCandidate> detect(const PointCloudT::Ptr &cloud);

  // parameters (public for quick tuning)
  double voxel_size;
  double neighbor_radius;
  int min_neighbors;
  double edge_offset_ratio;
  double min_cluster_size_m;
};
