#include <openvslam_ros.h>

#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <openvslam/publish/map_publisher.h>
#include <Eigen/Geometry>

#include <tf2_ros/create_timer_ros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace openvslam_ros {
system::system(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path, const std::string& mask_img_path)
    : SLAM_(cfg, vocab_file_path), cfg_(cfg), node_(std::make_shared<rclcpp::Node>("run_slam")), custom_qos_(rmw_qos_profile_default),
      mask_(mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE)),
      pose_pub_(node_->create_publisher<nav_msgs::msg::Odometry>("~/camera_pose", 1)) {

    // transformation from OpenCV coordinate frame to ROS
    cv_to_ros_ << 0, 0, 1,
        -1, 0, 0,
        0, -1, 0;

    custom_qos_.depth = 1;
    custom_qos_.reliability = RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;

    map_frame_id_ = node_->declare_parameter<std::string>("map_frame_id", "map");
    camera_frame_id_ = node_->declare_parameter<std::string>("camera_frame_id", "camera_link");
    odom_frame_id_ = node_->declare_parameter<std::string>("odom_frame_id", "odom");
    base_frame_id_ = node_->declare_parameter<std::string>("base_frame_id", "base_link");
    use_odometry_ = node_->declare_parameter<bool>("use_odometry", true);
    transform_tolerance_ = tf2::durationFromSec(node_->declare_parameter<float>("transform_tolerance", 0.5));

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
        node_->get_node_base_interface(),
        node_->get_node_timers_interface());
    tf_buffer_->setCreateTimerInterface(timer_interface);
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);

    exec_.add_node(node_);
}

void system::publish_pose() {
    if (SLAM_.get_frame_state().tracker_state != openvslam::tracker_state_t::Tracking) {
        // don't publish if we are not properly tracking
        return;
    }

    const auto state_timestamp = node_->now();

    // SLAM get the motion matrix publisher
    auto cam_pose_wc = SLAM_.get_map_publisher()->get_current_cam_pose_wc();

    // Extract rotation matrix and translation vector from
    Eigen::Matrix3d rot = cam_pose_wc.block<3, 3>(0, 0);
    Eigen::Vector3d trans = cam_pose_wc.block<3, 1>(0, 3);

    // Transform from CV coordinate system to ROS coordinate system on camera coordinates
    Eigen::Quaterniond quat(cv_to_ros_ * rot * cv_to_ros_.transpose());
    trans = cv_to_ros_ * trans;

    // Create odometry message and update it with current camera pose
    nav_msgs::msg::Odometry pose_msg;
    pose_msg.header.stamp = state_timestamp;
    pose_msg.header.frame_id = map_frame_id_;
    pose_msg.child_frame_id = camera_frame_id_;
    pose_msg.pose.pose.orientation.x = quat.x();
    pose_msg.pose.pose.orientation.y = quat.y();
    pose_msg.pose.pose.orientation.z = quat.z();
    pose_msg.pose.pose.orientation.w = quat.w();
    pose_msg.pose.pose.position.x = trans(0);
    pose_msg.pose.pose.position.y = trans(1);
    pose_msg.pose.pose.position.z = trans(2);
    pose_pub_->publish(pose_msg);

    tf2::Stamped<tf2::Transform> odom_to_map;
    tf2::Quaternion q(quat.x(), quat.y(), quat.z(), quat.w());
    tf2::Vector3 p(trans(0), trans(1), trans(2));
    tf2::Vector3 p_rot = tf2::quatRotate(q, p);
    tf2::Stamped<tf2::Transform> cam_to_map(
        tf2::Transform(q, -p_rot), tf2_ros::fromMsg(state_timestamp), camera_frame_id_);

    try {
        geometry_msgs::msg::TransformStamped cam_to_map_msg, base_to_map_msg, odom_to_map_msg;

        // https://github.com/ros2/geometry2/issues/176
        // not working for some reason...
        // base_to_map_msg = tf2::toMsg(base_to_map);
        cam_to_map_msg.header.stamp = state_timestamp;
        cam_to_map_msg.header.frame_id = cam_to_map.frame_id_;
        cam_to_map_msg.transform.translation.x = cam_to_map.getOrigin().getX();
        cam_to_map_msg.transform.translation.y = cam_to_map.getOrigin().getY();
        cam_to_map_msg.transform.translation.z = cam_to_map.getOrigin().getZ();
        cam_to_map_msg.transform.rotation = tf2::toMsg(cam_to_map.getRotation());

        tf_buffer_->transform(cam_to_map_msg, base_to_map_msg, base_frame_id_);

        odom_to_map_msg = tf_buffer_->transform(base_to_map_msg, odom_frame_id_);
        tf2::fromMsg(odom_to_map_msg, odom_to_map);
    } catch (tf2::TransformException & e) {
        RCLCPP_ERROR(node_->get_logger(), "Transform from base_link to odom failed: %s",
        e.what());
        return;
    }

    // set map to odom for our transformation thread to publish
    tf2::Transform map_to_odom = tf2::Transform(tf2::Quaternion(odom_to_map.getRotation() ),
        tf2::Vector3(odom_to_map.getOrigin())).inverse();

    geometry_msgs::msg::TransformStamped msg;
    msg.transform = tf2::toMsg(map_to_odom);
    msg.child_frame_id = odom_frame_id_;
    msg.header.frame_id = map_frame_id_;
    msg.header.stamp = state_timestamp + transform_tolerance_;

    RCLCPP_DEBUG(node_->get_logger(), "Sending out new transform %f %f %f rot %f %f %f %f",
                msg.transform.translation.x,
                msg.transform.translation.y,
                msg.transform.translation.z,
                msg.transform.rotation.w,
                msg.transform.rotation.x,
                msg.transform.rotation.y,
                msg.transform.rotation.z);

    tf_broadcaster_->sendTransform(msg);
}

std::pair<bool, geometry_msgs::msg::TransformStamped> system::get_transform(std::string const& from_frame,
    std::string const& to_frame, rclcpp::Time const& at_time) {

    // wait up to 100 ms for transform to arrive, parameter is nanoseconds
    auto transformTimeout = rclcpp::Duration::from_nanoseconds(0.1 * std::pow(10.0, 9.0));

    try {
        // first: target frame
        // second: source frame
        // lookup the full transformation all the way to the camera
        // this will include the odometry transformation which is the relevant part here !
        // because the map -> odom transformation is quite stable over time this
        // allows us to provide global position input, even if the tracking is lost
        auto tf_stamped = tf_buffer_->lookupTransform(odom_frame_id_, camera_frame_id_, at_time,
            transformTimeout);
        return {true, tf_stamped};
    } catch (tf2::TransformException const& ex) {
        RCLCPP_ERROR(node_->get_logger(), "Cannot get transformation from %s to %s: %s",
            from_frame, to_frame, ex.what());
        return {false, geometry_msgs::msg::TransformStamped()};
    }
}

openvslam::navigation_state system::tf_to_navigation_state( geometry_msgs::msg::Transform const& tf) const {
        Eigen::Vector3d trans;
        trans << tf.translation.x,
            tf.translation.y,
            tf.translation.z;

        const Eigen::Quaterniond rot(
            tf.rotation.w,
            tf.rotation.x,
            tf.rotation.y,
            tf.rotation.z);

        // openvslam Convention needs an inverted quaternion
        const Eigen::Quaterniond invRot = rot.inverse();
        // convert to OpenCV coordinate frame
        const Eigen::Quaterniond ovr_quat(cv_to_ros_.transpose() * invRot * cv_to_ros_);
        const Eigen::Vector3d ovr_trans = cv_to_ros_.transpose() * trans;

        openvslam::navigation_state nav_state;
        nav_state.valid = true;
        nav_state.cam_rotation = ovr_quat.normalized().toRotationMatrix();
        nav_state.cam_translation = ovr_trans;

        RCLCPP_DEBUG(node_->get_logger(), "Translation in Navigation state computed to: x=%f y=%f z=%f ",
            nav_state.cam_translation.x(),
            nav_state.cam_translation.y(),
            nav_state.cam_translation.z());

        return nav_state;
}

std::pair<openvslam::navigation_state, openvslam::navigation_state> system::get_odom_map_state(rclcpp::Time const& at_time) {

    openvslam::navigation_state odom_nav_state;
    odom_nav_state.valid = false;
    openvslam::navigation_state map_nav_state;
    map_nav_state.valid = false;

    if (!use_odometry_) {
        return {odom_nav_state, map_nav_state};
    }

    const auto odom_transform = get_transform(odom_frame_id_, camera_frame_id_, at_time);

    if (odom_transform.first) {
        odom_nav_state = tf_to_navigation_state(odom_transform.second.transform);
    }

    const auto map_transform = get_transform(map_frame_id_, camera_frame_id_, at_time);
    if (map_transform.first) {
        map_nav_state = tf_to_navigation_state(map_transform.second.transform);
    }

    return {odom_nav_state, map_nav_state};
}

mono::mono(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path, const std::string& mask_img_path)
    : system(cfg, vocab_file_path, mask_img_path) {
    sub_ = image_transport::create_subscription(
        node_.get(), "camera/image_raw", [this](const sensor_msgs::msg::Image::ConstSharedPtr& msg) { callback(msg); }, "raw", custom_qos_);
}
void mono::callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    const rclcpp::Time tp_1 = node_->now();
    const double timestamp = tp_1.seconds();

    // get odometry pose for this frame
    const auto odom_map_nav_state = get_odom_map_state(msg->header.stamp);

    RCLCPP_DEBUG(node_->get_logger(), "Forwarding with nav state: odom: %s map: %s",
        odom_map_nav_state.first.valid ? "valid"  : "invalid",
        odom_map_nav_state.second.valid ? "valid"  : "invalid");

    // input the current frame and estimate the camera pose
    SLAM_.feed_monocular_frame(cv_bridge::toCvShare(msg)->image, timestamp, mask_,
        odom_map_nav_state.first, odom_map_nav_state.second);

    const rclcpp::Time tp_2 = node_->now();
    const double track_time = (tp_2 - tp_1).seconds();

    //track times in seconds
    track_times_.push_back(track_time);
}

stereo::stereo(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path, const std::string& mask_img_path,
               const bool rectify)
    : system(cfg, vocab_file_path, mask_img_path),
      rectifier_(rectify ? std::make_shared<openvslam::util::stereo_rectifier>(cfg) : nullptr),
      left_sf_(node_, "camera/left/image_raw", custom_qos_),
      right_sf_(node_, "camera/right/image_raw", custom_qos_),
      sync_(left_sf_, right_sf_, 10) {
    sync_.registerCallback(&stereo::callback, this);
}

void stereo::callback(const sensor_msgs::msg::Image::ConstSharedPtr& left, const sensor_msgs::msg::Image::ConstSharedPtr& right) {
    auto leftcv = cv_bridge::toCvShare(left)->image;
    auto rightcv = cv_bridge::toCvShare(right)->image;
    if (leftcv.empty() || rightcv.empty()) {
        return;
    }

    if (rectifier_) {
        rectifier_->rectify(leftcv, rightcv, leftcv, rightcv);
    }

    const rclcpp::Time tp_1 = node_->now();
    const double timestamp = tp_1.seconds();

    // get odometry pose for this frame
    const auto odom_map_nav_state = get_odom_map_state(left->header.stamp);

    RCLCPP_DEBUG(node_->get_logger(), "Forwarding with nav state: odom: %s map: %s",
        odom_map_nav_state.first.valid ? "valid"  : "invalid",
        odom_map_nav_state.second.valid ? "valid"  : "invalid");

    // input the current frame and estimate the camera pose
    SLAM_.feed_stereo_frame(leftcv, rightcv, timestamp, mask_,
        odom_map_nav_state.first, odom_map_nav_state.second);

    const rclcpp::Time tp_2 = node_->now();
    const double track_time = (tp_2 - tp_1).seconds();

    //track times in seconds
    track_times_.push_back(track_time);
}

rgbd::rgbd(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path, const std::string& mask_img_path)
    : system(cfg, vocab_file_path, mask_img_path),
      color_sf_(node_, "camera/color/image_raw", custom_qos_),
      depth_sf_(node_, "camera/depth/image_raw", custom_qos_),
      sync_(color_sf_, depth_sf_, 10) {
    sync_.registerCallback(&rgbd::callback, this);
}

void rgbd::callback(const sensor_msgs::msg::Image::ConstSharedPtr& color, const sensor_msgs::msg::Image::ConstSharedPtr& depth) {
    auto colorcv = cv_bridge::toCvShare(color)->image;
    auto depthcv = cv_bridge::toCvShare(depth)->image;
    if (colorcv.empty() || depthcv.empty()) {
        return;
    }

    const rclcpp::Time tp_1 = node_->now();
    const double timestamp = tp_1.seconds();

    // get odometry pose for this frame
    const auto odom_map_nav_state = get_odom_map_state(color->header.stamp);

    RCLCPP_DEBUG(node_->get_logger(), "Forwarding with nav state: odom: %s map: %s",
        odom_map_nav_state.first.valid ? "valid"  : "invalid",
        odom_map_nav_state.second.valid ? "valid"  : "invalid");

    // input the current frame and estimate the camera pose
    SLAM_.feed_RGBD_frame(colorcv, depthcv, timestamp, mask_,
        odom_map_nav_state.first, odom_map_nav_state.second);

    const rclcpp::Time tp_2 = node_->now();
    const double track_time = (tp_2 - tp_1).seconds();

    // track time in seconds
    track_times_.push_back(track_time);
}

} // namespace openvslam_ros
