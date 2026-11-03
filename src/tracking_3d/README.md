# tracking_3d

Tracks people in 3D from YOLOv7 person boxes and the aligned depth image of a
RealSense D435i. The node publishes tracked people in a configurable frame,
which is `map` by default.

## Nodes

### `human_3d_tracker`

| | Topic | Type |
|---|---|---|
| in | `/yolov7/yolov7` | `vision_msgs/Detection2DArray` |
| in | `/obs_camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` (16UC1 or 32FC1) |
| in | `/obs_camera/aligned_depth_to_color/camera_info` | `sensor_msgs/CameraInfo` |
| out | `human_detection_3d` | `vision_msgs/Detection3DArray` in `~target_frame` |
| out | `human_distances` | `std_msgs/Float32MultiArray`, `[id, dist, id, dist, ...]` |
| out | `human_markers` | `visualization_msgs/MarkerArray` in `~target_frame` |

Each YOLO message is processed as follows:

1. Pick the buffered depth image taken at the same time as the image YOLO ran on.
2. Inside a chest-sized window of each person box, find the nearest large
   cluster of depth values, which is the person rather than the wall behind
   them. Take its median depth and the centre of its pixels.
3. Back-project that point and move it `body_depth_offset` along the viewing
   ray, from the front of the body to its centre.
4. Transform the point into `~target_frame` using TF at the image timestamp.
5. Track people with a constant-velocity Kalman filter in `~target_frame`.

`~target_frame` must have z pointing up (for example `map`, `odom`, `world`
or `base_link`). The box height is measured along that axis. A leading `/`
(`/map`) is accepted and removed.

All parameters, with comments, are in
[`config/human_3d_tracker.yaml`](config/human_3d_tracker.yaml).

### `detection3d_visualizer`

Draws `human_detection_3d` as boxes on `detection_markers`. The `~frame_id`
parameter is empty by default, which keeps each detection's own frame. Set it
to re-express the boxes in another frame through TF.

## Launch

```bash
roslaunch tracking_3d human_3d_tracker.launch target_frame:=map
```

## Time sync with yolov7-ros

`lukazso/yolov7-ros` stamps its `Detection2DArray` header with
`rospy.Time.now()`, which is the time it publishes, after inference. The time
the camera captured the image is only in each detection's `source_img.header`,
and the tracker reads it from there. A cleaner fix is to change
`create_detection_msg()` in `src/utils/ros.py` to use `header = img_msg.header`.
With that change you can also stop copying the full image into every
detection.
