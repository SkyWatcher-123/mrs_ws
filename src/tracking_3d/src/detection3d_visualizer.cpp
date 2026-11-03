#include <ros/ros.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <vision_msgs/Detection3DArray.h>
#include <visualization_msgs/MarkerArray.h>

// Draws vision_msgs/Detection3DArray as RViz boxes.
//
// ~frame_id: frame to draw in (e.g. "map"). Empty (default) = draw in the
// frame of the incoming detections. human_3d_tracker already publishes in its
// ~target_frame, so this only needs setting to re-express them elsewhere.
class Detection3DVisualizer {
private:
    ros::NodeHandle nh_;
    ros::Subscriber detection_sub_;
    ros::Publisher marker_pub_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::string frame_id_;
    double marker_lifetime_;

    static std::string stripLeadingSlash(std::string frame) {
        while (!frame.empty() && frame.front() == '/') frame.erase(0, 1);
        return frame;
    }

    bool lookup(const std::string& source, const ros::Time& stamp, tf2::Transform& out) {
        geometry_msgs::TransformStamped tf;
        try {
            tf = tf_buffer_.lookupTransform(frame_id_, source, stamp, ros::Duration(0.1));
        } catch (const tf2::TransformException&) {
            try {
                tf = tf_buffer_.lookupTransform(frame_id_, source, ros::Time(0));
            } catch (const tf2::TransformException& ex) {
                ROS_WARN_THROTTLE(2.0, "TF %s -> %s failed: %s", source.c_str(), frame_id_.c_str(), ex.what());
                return false;
            }
        }
        const auto& t = tf.transform;
        out = tf2::Transform(tf2::Quaternion(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w),
                             tf2::Vector3(t.translation.x, t.translation.y, t.translation.z));
        return true;
    }

public:
    Detection3DVisualizer(ros::NodeHandle& nh) : nh_(nh), tf_listener_(tf_buffer_) {
        ros::NodeHandle pnh("~");
        pnh.param<std::string>("frame_id", frame_id_, "");
        pnh.param<double>("marker_lifetime", marker_lifetime_, 0.5);
        frame_id_ = stripLeadingSlash(frame_id_);

        detection_sub_ = nh_.subscribe("human_detection_3d", 1, &Detection3DVisualizer::detectionCallback, this);
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("detection_markers", 1);

        ROS_INFO("Detection3D visualizer initialized, drawing in %s",
                 frame_id_.empty() ? "the detections' own frame" : ("frame '" + frame_id_ + "'").c_str());
    }

    void detectionCallback(const vision_msgs::Detection3DArray::ConstPtr& msg) {
        visualization_msgs::MarkerArray marker_array;

        for (size_t i = 0; i < msg->detections.size(); ++i) {
            const auto& detection = msg->detections[i];

            std_msgs::Header header = detection.header;
            if (header.frame_id.empty()) header = msg->header;
            header.frame_id = stripLeadingSlash(header.frame_id);

            geometry_msgs::Pose pose = detection.bbox.center;
            if (!frame_id_.empty() && frame_id_ != header.frame_id) {
                tf2::Transform T;
                if (!lookup(header.frame_id, header.stamp, T)) continue;
                const auto& p = pose.position;
                const auto& o = pose.orientation;
                const tf2::Transform in(tf2::Quaternion(o.x, o.y, o.z, o.w), tf2::Vector3(p.x, p.y, p.z));
                const tf2::Transform res = T * in;
                pose.position.x = res.getOrigin().x();
                pose.position.y = res.getOrigin().y();
                pose.position.z = res.getOrigin().z();
                const tf2::Quaternion q = res.getRotation();
                pose.orientation.x = q.x();
                pose.orientation.y = q.y();
                pose.orientation.z = q.z();
                pose.orientation.w = q.w();
                header.frame_id = frame_id_;
            }

            visualization_msgs::Marker box_marker;
            box_marker.header = header;
            box_marker.ns = "detection_boxes";
            box_marker.id = i;
            box_marker.type = visualization_msgs::Marker::CUBE;
            box_marker.action = visualization_msgs::Marker::ADD;
            box_marker.pose = pose;
            box_marker.scale = detection.bbox.size;
            box_marker.color.r = 0.0;
            box_marker.color.g = 0.5;
            box_marker.color.b = 1.0;
            box_marker.color.a = 0.6;
            box_marker.lifetime = ros::Duration(marker_lifetime_);
            marker_array.markers.push_back(box_marker);

            visualization_msgs::Marker text_marker;
            text_marker.header = header;
            text_marker.ns = "detection_ids";
            text_marker.id = i;
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::Marker::ADD;
            text_marker.pose.position = pose.position;
            text_marker.pose.position.z += detection.bbox.size.z / 2 + 0.2;  // above the box (z-up frame)
            text_marker.pose.orientation.w = 1.0;
            text_marker.scale.z = 0.2;
            text_marker.color.r = 1.0;
            text_marker.color.g = 1.0;
            text_marker.color.b = 1.0;
            text_marker.color.a = 1.0;
            text_marker.text = "Human " + std::to_string(i);
            text_marker.lifetime = ros::Duration(marker_lifetime_);
            marker_array.markers.push_back(text_marker);
        }

        marker_pub_.publish(marker_array);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "detection3d_visualizer");
    ros::NodeHandle nh;

    Detection3DVisualizer visualizer(nh);

    ros::spin();

    return 0;
}
