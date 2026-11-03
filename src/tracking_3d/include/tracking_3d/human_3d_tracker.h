#ifndef HUMAN_3D_TRACKER_H
#define HUMAN_3D_TRACKER_H

#include <deque>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32MultiArray.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <vision_msgs/Detection2D.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/Detection3DArray.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv2/core.hpp>

class Human3DTracker {
public:
    Human3DTracker(ros::NodeHandle& nh, ros::NodeHandle& pnh);

private:
    // Robust depth of the person inside an image region (see extractForeground).
    struct Foreground {
        double u = 0.0;      // mean pixel column of the person's pixels
        double v = 0.0;      // mean pixel row of the person's pixels
        double depth = 0.0;  // median depth of the person's pixels [m]
        int count = 0;       // number of pixels that belong to the person
    };

    // One person measured from a YOLO box + aligned depth, expressed in target_frame_.
    struct Measurement {
        tf2::Vector3 position;        // body centre (torso height)
        cv::Matx33d covariance;       // measurement noise, larger along the viewing ray
        double box_offset_z = 0.0;    // (vertical centre of the person) - position.z
        double box_height = 0.0;      // head-to-foot extent
        vision_msgs::Detection2D detection2d;
    };

    // Constant-velocity Kalman track. State [x y z vx vy vz] in target_frame_.
    // cv::Matx has value semantics, so tracks can be copied/erased safely
    // (cv::KalmanFilter shares its cv::Mat buffers between copies).
    struct Track {
        int id = 0;
        cv::Matx61d x;
        cv::Matx66d P;
        ros::Time stamp;              // time the state refers to
        int hits = 0;
        int coast_cycles = 0;
        Measurement last;
    };

    // ROS
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber yolo_sub_;
    ros::Subscriber depth_sub_;
    ros::Subscriber camera_info_sub_;
    ros::Publisher detection3d_pub_;
    ros::Publisher distance_pub_;
    ros::Publisher markers_pub_;

    // TF
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::string target_frame_;       // frame everything is tracked/published in (e.g. "map")
    std::string reference_frame_;    // frame the published distance is measured from
    double tf_timeout_;

    // Camera intrinsics (of the aligned depth == colour image)
    bool camera_info_received_ = false;
    double fx_ = 0.0, fy_ = 0.0, cx_ = 0.0, cy_ = 0.0;
    int image_width_ = 0, image_height_ = 0;

    // Depth images kept for matching against (late) YOLO results by timestamp
    std::deque<sensor_msgs::Image::ConstPtr> depth_buffer_;
    int depth_buffer_size_;
    double max_sync_offset_;

    // Detection filtering
    int person_class_id_;
    double min_score_;

    // Depth extraction
    double depth_scale_;             // 16UC1 units -> metres
    double min_depth_, max_depth_;
    double roi_width_frac_, roi_top_frac_, roi_bottom_frac_;
    int pixel_stride_;
    double depth_bin_size_;
    double min_cluster_frac_;
    double cluster_tolerance_;
    int min_valid_points_;
    double body_depth_offset_;

    // Tracking
    std::vector<Track> tracks_;
    int next_track_id_ = 0;
    double max_tracking_distance_;
    int max_coast_cycles_;
    int min_hits_;
    double process_noise_accel_;
    double meas_noise_lateral_;
    double meas_noise_depth_;
    double meas_noise_depth_quad_;
    double init_velocity_std_;

    // Output
    double human_width_, human_depth_;
    double default_box_height_, min_box_height_, max_box_height_;
    bool planar_distance_;
    double marker_lifetime_;

    // Callbacks
    void yoloCallback(const vision_msgs::Detection2DArray::ConstPtr& msg);
    void depthCallback(const sensor_msgs::Image::ConstPtr& msg);
    void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg);

    // Pipeline
    sensor_msgs::Image::ConstPtr findDepthFor(const ros::Time& stamp) const;
    void processDetections(const vision_msgs::Detection2DArray& detections,
                           const sensor_msgs::Image::ConstPtr& depth_msg);
    bool isPerson(const vision_msgs::Detection2D& detection) const;
    bool measurePerson(const vision_msgs::Detection2D& detection, const cv::Mat& depth,
                       const tf2::Transform& T_target_cam, Measurement& out) const;
    bool extractForeground(const cv::Mat& depth, const cv::Rect& roi, Foreground& out) const;
    void publish(const ros::Time& stamp, const tf2::Transform& T_target_cam);
    bool lookupTransform(const std::string& target, const std::string& source,
                         const ros::Time& stamp, tf2::Transform& out);

    // Tracking
    void updateTracks(const std::vector<Measurement>& measurements, const ros::Time& stamp);
    void predict(Track& track, const ros::Time& stamp) const;
    void correct(Track& track, const Measurement& m) const;
};

#endif // HUMAN_3D_TRACKER_H
