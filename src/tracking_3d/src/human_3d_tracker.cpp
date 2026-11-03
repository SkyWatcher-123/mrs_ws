/**
 * @file human_3d_tracker.cpp
 * @brief 3D human tracking from YOLOv7 boxes + RealSense D435i aligned depth.
 *
 * Pipeline per YOLO message:
 *   1. pick the aligned depth image whose stamp matches the YOLO image stamp
 *   2. per person box: take depth from a torso-sized window, keep only the
 *      nearest dominant depth cluster (the person, not the wall/floor behind),
 *      back-project its centroid and push it to the body centre
 *   3. transform the measurement into target_frame (e.g. "map") at the image time
 *   4. constant-velocity Kalman tracking in target_frame
 *   5. publish Detection3DArray, distances and RViz markers in target_frame
 */

#include "tracking_3d/human_3d_tracker.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/image_encodings.h>
#include <tf2/LinearMath/Quaternion.h>

namespace {

// tf2 frame ids must not start with '/' ("/map" -> "map").
std::string stripLeadingSlash(std::string frame) {
  while (!frame.empty() && frame.front() == '/') frame.erase(0, 1);
  return frame;
}

tf2::Transform toTf2(const geometry_msgs::Transform& t) {
  return tf2::Transform(
      tf2::Quaternion(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w),
      tf2::Vector3(t.translation.x, t.translation.y, t.translation.z));
}

geometry_msgs::Point toPoint(double x, double y, double z) {
  geometry_msgs::Point p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}

// Depth in metres at (row, col); 0 if invalid.
inline double readDepth(const cv::Mat& depth, int row, int col, double scale) {
  if (depth.type() == CV_16UC1) return depth.at<uint16_t>(row, col) * scale;
  const float d = depth.at<float>(row, col);  // 32FC1, already metres
  return std::isfinite(d) ? static_cast<double>(d) : 0.0;
}

cv::Rect clampedRect(double u0, double v0, double u1, double v1, const cv::Size& size) {
  const int x0 = std::max(0, static_cast<int>(std::floor(u0)));
  const int y0 = std::max(0, static_cast<int>(std::floor(v0)));
  const int x1 = std::min(size.width, static_cast<int>(std::ceil(u1)));
  const int y1 = std::min(size.height, static_cast<int>(std::ceil(v1)));
  if (x1 <= x0 || y1 <= y0) return cv::Rect();
  return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

}  // namespace

Human3DTracker::Human3DTracker(ros::NodeHandle& nh, ros::NodeHandle& pnh)
  : nh_(nh)
  , pnh_(pnh)
  , tf_listener_(tf_buffer_)
{
  std::string detections_topic, depth_topic, camera_info_topic;
  pnh_.param<std::string>("detections_topic", detections_topic, "/yolov7/yolov7");
  pnh_.param<std::string>("depth_topic", depth_topic, "/obs_camera/aligned_depth_to_color/image_raw");
  pnh_.param<std::string>("camera_info_topic", camera_info_topic, "/obs_camera/aligned_depth_to_color/camera_info");

  pnh_.param<std::string>("target_frame", target_frame_, "map");
  pnh_.param<std::string>("reference_frame", reference_frame_, "base_link");
  target_frame_ = stripLeadingSlash(target_frame_);
  reference_frame_ = stripLeadingSlash(reference_frame_);
  pnh_.param<double>("tf_timeout", tf_timeout_, 0.1);

  pnh_.param<int>("depth_buffer_size", depth_buffer_size_, 30);
  pnh_.param<double>("max_sync_offset", max_sync_offset_, 0.05);

  pnh_.param<int>("person_class_id", person_class_id_, 0);
  pnh_.param<double>("min_score", min_score_, 0.0);

  pnh_.param<double>("depth_scale", depth_scale_, 0.001);
  pnh_.param<double>("min_depth", min_depth_, 0.3);
  pnh_.param<double>("max_depth", max_depth_, 8.0);
  pnh_.param<double>("roi_width_fraction", roi_width_frac_, 0.4);
  pnh_.param<double>("roi_top_fraction", roi_top_frac_, 0.15);
  pnh_.param<double>("roi_bottom_fraction", roi_bottom_frac_, 0.55);
  pnh_.param<int>("pixel_stride", pixel_stride_, 2);
  pnh_.param<double>("depth_bin_size", depth_bin_size_, 0.05);
  pnh_.param<double>("min_cluster_fraction", min_cluster_frac_, 0.15);
  pnh_.param<double>("cluster_tolerance", cluster_tolerance_, 0.25);
  pnh_.param<int>("min_valid_points", min_valid_points_, 20);
  pnh_.param<double>("body_depth_offset", body_depth_offset_, 0.10);
  pixel_stride_ = std::max(1, pixel_stride_);
  depth_bin_size_ = std::max(0.01, depth_bin_size_);

  pnh_.param<double>("max_tracking_distance", max_tracking_distance_, 1.0);
  pnh_.param<int>("max_coast_cycles", max_coast_cycles_, 5);
  pnh_.param<int>("min_hits", min_hits_, 3);
  pnh_.param<double>("process_noise_accel", process_noise_accel_, 2.0);
  pnh_.param<double>("meas_noise_lateral", meas_noise_lateral_, 0.05);
  pnh_.param<double>("meas_noise_depth", meas_noise_depth_, 0.05);
  pnh_.param<double>("meas_noise_depth_quadratic", meas_noise_depth_quad_, 0.01);
  pnh_.param<double>("init_velocity_std", init_velocity_std_, 1.0);

  pnh_.param<double>("human_width", human_width_, 0.5);
  pnh_.param<double>("human_depth", human_depth_, 0.3);
  pnh_.param<double>("default_box_height", default_box_height_, 1.7);
  pnh_.param<double>("min_box_height", min_box_height_, 0.5);
  pnh_.param<double>("max_box_height", max_box_height_, 2.2);
  pnh_.param<bool>("planar_distance", planar_distance_, true);
  pnh_.param<double>("marker_lifetime", marker_lifetime_, 0.5);

  yolo_sub_ = nh_.subscribe(detections_topic, 5, &Human3DTracker::yoloCallback, this);
  depth_sub_ = nh_.subscribe(depth_topic, 5, &Human3DTracker::depthCallback, this);
  camera_info_sub_ = nh_.subscribe(camera_info_topic, 1, &Human3DTracker::cameraInfoCallback, this);

  detection3d_pub_ = nh_.advertise<vision_msgs::Detection3DArray>("human_detection_3d", 1);
  distance_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("human_distances", 1);
  markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("human_markers", 1);

  ROS_INFO("Human 3D Tracker initialized. Output frame: '%s', distance reference frame: '%s'",
           target_frame_.c_str(), reference_frame_.c_str());
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void Human3DTracker::cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg) {
  fx_ = msg->K[0];
  fy_ = msg->K[4];
  cx_ = msg->K[2];
  cy_ = msg->K[5];
  image_width_ = msg->width;
  image_height_ = msg->height;
  camera_info_received_ = fx_ > 0.0 && fy_ > 0.0;
  if (!camera_info_received_) {
    ROS_WARN("Camera info has an invalid K matrix, waiting for another one");
    return;
  }
  camera_info_sub_.shutdown();  // intrinsics are constant
  ROS_INFO("Camera intrinsics received: fx=%.1f fy=%.1f cx=%.1f cy=%.1f (%dx%d)",
           fx_, fy_, cx_, cy_, image_width_, image_height_);
}

void Human3DTracker::depthCallback(const sensor_msgs::Image::ConstPtr& msg) {
  // Only buffer here. Processing is driven by YOLO messages so every detection
  // is used exactly once (re-processing old boxes against every new depth frame
  // inflated hit counts and paired boxes with depth from a different instant).
  depth_buffer_.push_back(msg);
  while (static_cast<int>(depth_buffer_.size()) > std::max(1, depth_buffer_size_)) {
    depth_buffer_.pop_front();
  }
}

void Human3DTracker::yoloCallback(const vision_msgs::Detection2DArray::ConstPtr& msg) {
  if (!camera_info_received_) {
    ROS_WARN_THROTTLE(2.0, "Waiting for camera_info before processing detections");
    return;
  }
  // Capture time of the image YOLO ran on. lukazso/yolov7-ros stamps the
  // Detection2DArray header with Time::now() (publish time, i.e. after
  // inference) but copies the input image into each detection's source_img,
  // whose header carries the real camera stamp, so prefer that.
  ros::Time image_stamp = msg->header.stamp;
  for (const auto& detection : msg->detections) {
    if (!detection.source_img.header.stamp.isZero()) {
      image_stamp = detection.source_img.header.stamp;
      break;
    }
  }
  sensor_msgs::Image::ConstPtr depth = findDepthFor(image_stamp);
  if (!depth) {
    ROS_WARN_THROTTLE(2.0, "Waiting for depth images before processing detections");
    return;
  }
  processDetections(*msg, depth);
}

// The YOLO result arrives later than the depth image it was computed from
// (inference latency). Use the depth image with the matching stamp instead of
// the newest one, otherwise a walking person is measured where they *are now*
// with a box from where they *were*.
sensor_msgs::Image::ConstPtr Human3DTracker::findDepthFor(const ros::Time& stamp) const {
  if (depth_buffer_.empty()) return nullptr;
  if (stamp.isZero()) {
    ROS_WARN_ONCE("YOLO detections carry no image stamp, using the newest depth image. "
                  "Copy the input image header into the YOLO output for proper time sync.");
    return depth_buffer_.back();
  }
  sensor_msgs::Image::ConstPtr best;
  double best_dt = std::numeric_limits<double>::max();
  for (const auto& d : depth_buffer_) {
    const double dt = std::fabs((d->header.stamp - stamp).toSec());
    if (dt < best_dt) {
      best_dt = dt;
      best = d;
    }
  }
  if (best_dt > max_sync_offset_) {
    ROS_WARN_THROTTLE(2.0, "Closest depth image is %.3f s from the detection stamp (max_sync_offset %.3f s). "
                      "Increase depth_buffer_size if YOLO latency is high.", best_dt, max_sync_offset_);
  }
  return best;
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

void Human3DTracker::processDetections(const vision_msgs::Detection2DArray& detections,
                                       const sensor_msgs::Image::ConstPtr& depth_msg) {
  cv_bridge::CvImageConstPtr depth;
  try {
    depth = cv_bridge::toCvShare(depth_msg);
  } catch (const cv_bridge::Exception& e) {
    ROS_ERROR_THROTTLE(1.0, "cv_bridge exception: %s", e.what());
    return;
  }
  if (depth->image.type() != CV_16UC1 && depth->image.type() != CV_32FC1) {
    ROS_ERROR_THROTTLE(1.0, "Unsupported depth encoding '%s' (expected 16UC1 or 32FC1)",
                       depth_msg->encoding.c_str());
    return;
  }
  if (depth->image.cols != image_width_ || depth->image.rows != image_height_) {
    ROS_WARN_THROTTLE(5.0, "Depth image is %dx%d but camera_info is %dx%d. Use the aligned_depth_to_color "
                      "image and its camera_info so YOLO pixels and depth pixels line up.",
                      depth->image.cols, depth->image.rows, image_width_, image_height_);
  }

  const ros::Time stamp = depth_msg->header.stamp;
  const std::string camera_frame = stripLeadingSlash(depth_msg->header.frame_id);

  tf2::Transform T_target_cam;
  if (!lookupTransform(target_frame_, camera_frame, stamp, T_target_cam)) return;

  std::vector<Measurement> measurements;
  for (const auto& detection : detections.detections) {
    if (!isPerson(detection)) continue;
    Measurement m;
    if (measurePerson(detection, depth->image, T_target_cam, m)) {
      measurements.push_back(m);
    }
  }

  updateTracks(measurements, stamp);
  publish(stamp, T_target_cam);
}

bool Human3DTracker::isPerson(const vision_msgs::Detection2D& detection) const {
  for (const auto& result : detection.results) {
    if (result.id == person_class_id_ && result.score >= min_score_) return true;
  }
  return false;
}

// Robust "how far is the person" inside an image region.
//
// The old code averaged every pixel of the YOLO box. A standing person fills
// only part of their box; the rest is floor / wall *behind* them, so the mean
// depth (and therefore the 3D point) landed behind the person.
//
// Here: histogram the valid depths, take the NEAREST peak that holds a
// meaningful share of the pixels (the person is in front of the background),
// keep pixels within cluster_tolerance of it and return their median depth and
// pixel centroid.
bool Human3DTracker::extractForeground(const cv::Mat& depth, const cv::Rect& roi, Foreground& out) const {
  if (roi.area() <= 0) return false;

  struct Sample { int u, v; double z; };
  std::vector<Sample> samples;
  samples.reserve(roi.area() / (pixel_stride_ * pixel_stride_) + 1);
  for (int v = roi.y; v < roi.y + roi.height; v += pixel_stride_) {
    for (int u = roi.x; u < roi.x + roi.width; u += pixel_stride_) {
      const double z = readDepth(depth, v, u, depth_scale_);
      if (z >= min_depth_ && z <= max_depth_) samples.push_back({u, v, z});
    }
  }
  if (static_cast<int>(samples.size()) < min_valid_points_) return false;

  const int nbins = std::max(1, static_cast<int>(std::ceil((max_depth_ - min_depth_) / depth_bin_size_)));
  std::vector<int> hist(nbins, 0);
  for (const auto& s : samples) {
    const int b = std::min(nbins - 1, static_cast<int>((s.z - min_depth_) / depth_bin_size_));
    ++hist[b];
  }
  // 3-bin smoothing so a surface straddling a bin edge is not split in two.
  std::vector<int> smooth(nbins, 0);
  for (int i = 0; i < nbins; ++i) {
    smooth[i] = hist[i] + (i > 0 ? hist[i - 1] : 0) + (i + 1 < nbins ? hist[i + 1] : 0);
  }

  const int min_count = std::max(min_valid_points_,
                                 static_cast<int>(min_cluster_frac_ * static_cast<double>(samples.size())));
  int peak = -1;
  for (int i = 0; i < nbins; ++i) {
    const int left = i > 0 ? smooth[i - 1] : 0;
    const int right = i + 1 < nbins ? smooth[i + 1] : 0;
    if (smooth[i] >= min_count && smooth[i] >= left && smooth[i] >= right) {
      peak = i;  // nearest significant local maximum
      break;
    }
  }
  if (peak < 0) {
    peak = static_cast<int>(std::max_element(smooth.begin(), smooth.end()) - smooth.begin());
  }

  // Refine the peak position with the mean depth of the samples in its 3 bins.
  const double lo = min_depth_ + (peak - 1) * depth_bin_size_;
  const double hi = min_depth_ + (peak + 2) * depth_bin_size_;
  double sum = 0.0;
  int n = 0;
  for (const auto& s : samples) {
    if (s.z >= lo && s.z < hi) {
      sum += s.z;
      ++n;
    }
  }
  const double center = n > 0 ? sum / n : min_depth_ + (peak + 0.5) * depth_bin_size_;

  std::vector<double> zs;
  zs.reserve(samples.size());
  double su = 0.0, sv = 0.0;
  for (const auto& s : samples) {
    if (std::fabs(s.z - center) <= cluster_tolerance_) {
      su += s.u;
      sv += s.v;
      zs.push_back(s.z);
    }
  }
  if (static_cast<int>(zs.size()) < min_valid_points_) return false;

  std::nth_element(zs.begin(), zs.begin() + zs.size() / 2, zs.end());
  out.depth = zs[zs.size() / 2];
  out.u = su / zs.size();
  out.v = sv / zs.size();
  out.count = static_cast<int>(zs.size());
  return true;
}

bool Human3DTracker::measurePerson(const vision_msgs::Detection2D& detection, const cv::Mat& depth,
                                   const tf2::Transform& T_target_cam, Measurement& out) const {
  const auto& bb = detection.bbox;
  if (bb.size_x <= 1.0 || bb.size_y <= 1.0) return false;

  const double v_top = bb.center.y - 0.5 * bb.size_y;
  const double v_bottom = bb.center.y + 0.5 * bb.size_y;

  // Torso window: centred horizontally, chest/stomach vertically. Mostly
  // person pixels, few background pixels, and away from arms and legs.
  const double half_w = 0.5 * roi_width_frac_ * bb.size_x;
  const cv::Rect torso = clampedRect(bb.center.x - half_w, v_top + roi_top_frac_ * bb.size_y,
                                     bb.center.x + half_w, v_top + roi_bottom_frac_ * bb.size_y,
                                     depth.size());
  const cv::Rect full = clampedRect(bb.center.x - 0.5 * bb.size_x, v_top,
                                    bb.center.x + 0.5 * bb.size_x, v_bottom, depth.size());

  Foreground fg;
  if (!extractForeground(depth, torso, fg) && !extractForeground(depth, full, fg)) {
    ROS_DEBUG("No usable depth for person box at (%.0f, %.0f)", bb.center.x, bb.center.y);
    return false;
  }

  // Back-project the person's pixel centroid. This is a point on the front
  // surface of the body; move it along the viewing ray to the body centre.
  const tf2::Vector3 surface((fg.u - cx_) * fg.depth / fx_, (fg.v - cy_) * fg.depth / fy_, fg.depth);
  const tf2::Vector3 ray_cam = surface.normalized();
  const tf2::Vector3 center_cam = surface + ray_cam * body_depth_offset_;

  const tf2::Vector3 cam_origin = T_target_cam.getOrigin();
  const tf2::Matrix3x3& R = T_target_cam.getBasis();
  out.position = T_target_cam * center_cam;

  // Stereo depth noise grows ~quadratically with range and is much larger
  // along the viewing ray than across it: R = s_lat^2 I + (s_ray^2 - s_lat^2) r r^T
  const double range = center_cam.length();
  const double s_lat = meas_noise_lateral_;
  const double s_ray = meas_noise_depth_ + meas_noise_depth_quad_ * range * range;
  const tf2::Vector3 r = R * ray_cam;
  const double rr[3] = {r.x(), r.y(), r.z()};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out.covariance(i, j) = (i == j ? s_lat * s_lat : 0.0) + (s_ray * s_ray - s_lat * s_lat) * rr[i] * rr[j];
    }
  }

  // Vertical extent: intersect the rays through the top and bottom of the box
  // with a vertical plane through the body centre (target_frame is z-up).
  // Works for tilted cameras, unlike back-projecting at a constant depth.
  out.box_offset_z = 0.0;
  out.box_height = default_box_height_;
  tf2::Vector3 n = out.position - cam_origin;
  n.setZ(0.0);
  if (n.length() > 0.1) {
    n.normalize();
    const double plane_d = n.dot(out.position - cam_origin);
    auto zAtRow = [&](double v, double& z) {
      const tf2::Vector3 ray = R * tf2::Vector3((fg.u - cx_) / fx_, (v - cy_) / fy_, 1.0);
      const double denom = n.dot(ray);
      if (denom < 1e-6) return false;
      z = (cam_origin + ray * (plane_d / denom)).z();
      return true;
    };
    double z_top = 0.0, z_bottom = 0.0;
    const double v_top_img = std::max(0.0, v_top);
    const double v_bottom_img = std::min(static_cast<double>(depth.rows - 1), v_bottom);
    if (zAtRow(v_top_img, z_top) && zAtRow(v_bottom_img, z_bottom) && z_top > z_bottom) {
      out.box_height = std::min(max_box_height_, std::max(min_box_height_, z_top - z_bottom));
      out.box_offset_z = 0.5 * (z_top + z_bottom) - out.position.z();
    }
  }

  out.detection2d = detection;
  return true;
}

// ---------------------------------------------------------------------------
// Tracking (constant velocity Kalman filter in target_frame)
// ---------------------------------------------------------------------------

void Human3DTracker::predict(Track& track, const ros::Time& stamp) const {
  // dt is measured from the time the state refers to, so coasting tracks are
  // not over-predicted (the old code used "time since last hit" every cycle).
  const double dt = (stamp - track.stamp).toSec();
  if (dt <= 0.0) return;

  cv::Matx66d F = cv::Matx66d::eye();
  for (int i = 0; i < 3; ++i) F(i, i + 3) = dt;

  // Discrete white-noise acceleration model.
  const double q = process_noise_accel_ * process_noise_accel_;
  const double q_pp = 0.25 * dt * dt * dt * dt * q;
  const double q_pv = 0.5 * dt * dt * dt * q;
  const double q_vv = dt * dt * q;
  cv::Matx66d Q = cv::Matx66d::zeros();
  for (int i = 0; i < 3; ++i) {
    Q(i, i) = q_pp;
    Q(i, i + 3) = q_pv;
    Q(i + 3, i) = q_pv;
    Q(i + 3, i + 3) = q_vv;
  }

  track.x = F * track.x;
  track.P = F * track.P * F.t() + Q;
  track.stamp = stamp;
}

void Human3DTracker::correct(Track& track, const Measurement& m) const {
  // H = [I3 0]
  const cv::Matx31d z(m.position.x(), m.position.y(), m.position.z());
  const cv::Matx31d hx(track.x(0), track.x(1), track.x(2));
  const cv::Matx33d S = track.P.get_minor<3, 3>(0, 0) + m.covariance;
  const cv::Matx<double, 6, 3> K = track.P.get_minor<6, 3>(0, 0) * S.inv();

  track.x = track.x + K * (z - hx);

  cv::Matx66d KH = cv::Matx66d::zeros();
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 3; ++c) KH(r, c) = K(r, c);
  }
  track.P = (cv::Matx66d::eye() - KH) * track.P;
  track.P = 0.5 * (track.P + track.P.t());
}

void Human3DTracker::updateTracks(const std::vector<Measurement>& measurements, const ros::Time& stamp) {
  for (auto& track : tracks_) predict(track, stamp);

  // Global nearest-neighbour: sort all gated (track, measurement) pairs by
  // distance and assign greedily, instead of first-come per detection.
  struct Pair { double dist; size_t t; size_t m; };
  std::vector<Pair> pairs;
  for (size_t t = 0; t < tracks_.size(); ++t) {
    const tf2::Vector3 p(tracks_[t].x(0), tracks_[t].x(1), tracks_[t].x(2));
    for (size_t m = 0; m < measurements.size(); ++m) {
      const double d = (measurements[m].position - p).length();
      if (d < max_tracking_distance_) pairs.push_back({d, t, m});
    }
  }
  std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.dist < b.dist; });

  std::vector<bool> track_used(tracks_.size(), false);
  std::vector<bool> meas_used(measurements.size(), false);
  for (const auto& pr : pairs) {
    if (track_used[pr.t] || meas_used[pr.m]) continue;
    track_used[pr.t] = true;
    meas_used[pr.m] = true;
    Track& track = tracks_[pr.t];
    correct(track, measurements[pr.m]);
    track.last = measurements[pr.m];
    track.hits++;
    track.coast_cycles = 0;
  }

  for (size_t t = 0; t < tracks_.size(); ++t) {
    if (!track_used[t]) tracks_[t].coast_cycles++;
  }

  for (size_t m = 0; m < measurements.size(); ++m) {
    if (meas_used[m]) continue;
    Track track;
    track.id = next_track_id_++;
    const tf2::Vector3& p = measurements[m].position;
    track.x = cv::Matx61d(p.x(), p.y(), p.z(), 0.0, 0.0, 0.0);
    track.P = cv::Matx66d::zeros();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) track.P(i, j) = measurements[m].covariance(i, j);
      track.P(i + 3, i + 3) = init_velocity_std_ * init_velocity_std_;
    }
    track.stamp = stamp;
    track.hits = 1;
    track.coast_cycles = 0;
    track.last = measurements[m];
    tracks_.push_back(track);
  }

  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [this](const Track& t) { return t.coast_cycles > max_coast_cycles_; }),
                tracks_.end());
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

void Human3DTracker::publish(const ros::Time& stamp, const tf2::Transform& T_target_cam) {
  vision_msgs::Detection3DArray detection_array;
  detection_array.header.stamp = stamp;
  detection_array.header.frame_id = target_frame_;

  std_msgs::Float32MultiArray distance_msg;
  visualization_msgs::MarkerArray marker_array;

  tf2::Transform T_ref_target;
  const bool have_ref = lookupTransform(reference_frame_, target_frame_, stamp, T_ref_target);
  const tf2::Vector3 cam_origin = T_target_cam.getOrigin();

  for (const Track& track : tracks_) {
    if (track.hits < min_hits_) continue;

    const tf2::Vector3 p(track.x(0), track.x(1), track.x(2));

    // Upright box (target_frame is z-up) turned to face the camera.
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, std::atan2(cam_origin.y() - p.y(), cam_origin.x() - p.x()));
    geometry_msgs::Quaternion q_msg;
    q_msg.x = q.x();
    q_msg.y = q.y();
    q_msg.z = q.z();
    q_msg.w = q.w();
    const double box_z = p.z() + track.last.box_offset_z;
    const double box_h = track.last.box_height;

    double distance = -1.0;
    if (have_ref) {
      const tf2::Vector3 p_ref = T_ref_target * p;
      distance = planar_distance_ ? std::hypot(p_ref.x(), p_ref.y()) : p_ref.length();
    }

    // Detection3D
    vision_msgs::Detection3D det;
    det.header = detection_array.header;
    det.results = track.last.detection2d.results;
    for (auto& result : det.results) {
      result.pose.pose.position = toPoint(p.x(), p.y(), p.z());
      result.pose.pose.orientation = q_msg;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) result.pose.covariance[i * 6 + j] = track.P(i, j);
      }
    }
    det.bbox.center.position = toPoint(p.x(), p.y(), box_z);
    det.bbox.center.orientation = q_msg;
    det.bbox.size.x = human_depth_;   // along the viewing direction
    det.bbox.size.y = human_width_;
    det.bbox.size.z = box_h;          // vertical
    detection_array.detections.push_back(det);

    distance_msg.data.push_back(static_cast<float>(track.id));
    distance_msg.data.push_back(static_cast<float>(distance));

    // Markers
    visualization_msgs::Marker marker;
    marker.header = detection_array.header;
    marker.action = visualization_msgs::Marker::ADD;
    marker.id = track.id;
    marker.lifetime = ros::Duration(marker_lifetime_);

    visualization_msgs::Marker center = marker;
    center.ns = "human_centers";
    center.type = visualization_msgs::Marker::SPHERE;
    center.pose.position = toPoint(p.x(), p.y(), p.z());
    center.pose.orientation.w = 1.0;
    center.scale.x = center.scale.y = center.scale.z = 0.2;
    center.color.g = 1.0;
    center.color.a = 0.8;
    marker_array.markers.push_back(center);

    visualization_msgs::Marker box = marker;
    box.ns = "human_boxes";
    box.type = visualization_msgs::Marker::CUBE;
    box.pose = det.bbox.center;
    box.scale = det.bbox.size;
    box.color.b = 1.0;
    box.color.a = 0.3;
    marker_array.markers.push_back(box);

    visualization_msgs::Marker text = marker;
    text.ns = "human_ids";
    text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text.pose.position = toPoint(p.x(), p.y(), box_z + 0.5 * box_h + 0.3);  // above the head
    text.pose.orientation.w = 1.0;
    text.scale.z = 0.3;
    text.color.r = text.color.g = text.color.b = 1.0;
    text.color.a = 1.0;
    std::stringstream ss;
    ss << "ID: " << track.id << "\nDist: ";
    if (distance >= 0.0) {
      ss << std::fixed << std::setprecision(2) << distance << "m";
    } else {
      ss << "n/a";
    }
    text.text = ss.str();
    marker_array.markers.push_back(text);
  }

  detection3d_pub_.publish(detection_array);
  distance_pub_.publish(distance_msg);
  markers_pub_.publish(marker_array);
}

bool Human3DTracker::lookupTransform(const std::string& target, const std::string& source,
                                     const ros::Time& stamp, tf2::Transform& out) {
  if (target == source) {
    out.setIdentity();
    return true;
  }
  geometry_msgs::TransformStamped tf;
  try {
    // Transform at the image time: required when the camera or robot moves.
    tf = tf_buffer_.lookupTransform(target, source, stamp, ros::Duration(tf_timeout_));
  } catch (const tf2::TransformException& ex) {
    try {
      tf = tf_buffer_.lookupTransform(target, source, ros::Time(0));
      ROS_WARN_THROTTLE(5.0, "TF %s -> %s not available at image time, using latest (%s)",
                        source.c_str(), target.c_str(), ex.what());
    } catch (const tf2::TransformException& ex2) {
      ROS_WARN_THROTTLE(2.0, "TF lookup %s -> %s failed: %s", source.c_str(), target.c_str(), ex2.what());
      return false;
    }
  }
  out = toTf2(tf.transform);
  return true;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "human_3d_tracker");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  Human3DTracker tracker(nh, pnh);
  ros::spin();
  return 0;
}
