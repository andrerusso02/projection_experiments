import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

import rospy
from sensor_msgs.msg import CameraInfo
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped, PointStamped


class BirdView:

    def __init__(self, K, T_cam_to_work_plane, pos_start_work_plane, pos_end_work_plane, resolution=20):
        self._K = K
        self._T_cam_to_work_plane = T_cam_to_work_plane
        self._resolution = resolution # 1m = <resolution> px

        self._pos_start_work_plane = np.array(pos_start_work_plane)
        self._pos_end_work_plane = np.array(pos_end_work_plane)

        self._img_size_px = (self._pos_end_work_plane - self._pos_start_work_plane) * self._resolution
        self._img_size_px = self._img_size_px.astype(np.int32)

        self._H = None
        self._init_projection_parameters()

    
    def _init_projection_parameters(self):

        pts_work_plane = np.array([[0., 0., 0., 1.],
                       [1., 0., 0., 1.],
                       [0., 1., 0., 1.],
                       [1., 1., 0., 1.]])
        pts_work_plane = pts_work_plane.T

        pts_cam = np.linalg.inv(self._T_cam_to_work_plane) @ pts_work_plane

        pts_sensor = pts_cam[:2] / pts_cam[2]
        pts_sensor = np.vstack((pts_sensor, np.ones((1,4))))

        pts_img = K @ pts_cam[:3]
        pts_img = pts_img / pts_img[2]

        pts_work_plane_px = pts_work_plane[:2]
        pts_work_plane_px *= self._resolution
        pts_work_plane_px = pts_work_plane_px.T
        pts_work_plane_px -= self._pos_start_work_plane * self._resolution
        pts_work_plane_px = pts_work_plane_px.astype(np.float32)

        self._H = cv2.getPerspectiveTransform(pts_img[:2].T.astype(np.float32), pts_work_plane_px)
    
    def project_img_to_bird(self, img):
        return cv2.warpPerspective(img, self._H, self._img_size_px)
    
    def project_work_plane_pt_to_source_img(self, pt):
        pt = pt - self._pos_start_work_plane
        pt = pt * self._resolution
        pt = np.hstack((pt, 1.))
        pt = np.linalg.inv(self._H) @ pt
        if pt[2] == 0:
            return None
        pt = pt / pt[2]
        return pt[:2]
    
    def project_source_img_pt_to_work_plane(self, pt):
        pt = np.hstack((pt, 1.))
        pt = self._H @ pt
        if pt[2] == 0:
            return None
        pt = pt / pt[2]
        pt = pt[:2]
        pt = pt / self._resolution
        pt = pt + self._pos_start_work_plane
        return pt
    
    def get_work_plane_pt_in_bird_img(self, pt):
        pt = pt - self._pos_start_work_plane
        pt = pt * self._resolution
        return pt
    
    def get_bird_img_pt_in_work_plane(self, pt):
        pt = pt / self._resolution
        pt = pt + self._pos_start_work_plane
        return pt


class BirdViewRos(BirdView):
    def __init__(self, frame_work_plane, frame_camera, camera_info_topic_name, pos_start_work_plane, pos_end_work_plane, resolution=20):

        self.frame_work_plane = frame_work_plane
        self.frame_camera = frame_camera

        self.ready = False
        self._K = None
        self._T_cam_to_work_plane = None
        self._init_from_ros_topics(camera_info_topic_name)

        super().__init__(self._K, self._T_cam_to_work_plane, pos_start_work_plane, pos_end_work_plane, resolution)
    
    def _init_from_ros_topics(self, topic_name):

        rospy.init_node('bird_view_initializer', anonymous=True)
        self.sub_camera_info = rospy.Subscriber(topic_name, CameraInfo, self._camera_info_callback)
        self.timer_tf = rospy.Timer(rospy.Duration(0.1), self._timer_callback)

        while not rospy.is_shutdown() and not self.ready:
            rospy.sleep(0.1)
        
        rospy.loginfo("bird view initialized from ros topics")

        # shutdown node
        rospy.signal_shutdown("bird view initialized from ros topics")




    def _camera_info_callback(self, msg):
        print("camera info received")
        self._K = np.array(msg.K).reshape((3,3))
        # unregister subscriber after receiving the first message
        self.sub_camera_info.unregister()
    
    def _timer_callback(self, event):

        if self._T_cam_to_work_plane is None:

            # try to get tf from camera to work plane
            try:
                tf_buffer = tf2_ros.Buffer()
                tf_listener = tf2_ros.TransformListener(tf_buffer)
                tf_stamped = tf_buffer.lookup_transform(self.frame_work_plane, self.frame_camera, rospy.Time(0))
                self._T_cam_to_work_plane = tf2_geometry_msgs.transform_to_kdl(tf_stamped.transform)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn_once("Failed to get tf from {} to {}, retrying".format(self.frame_camera, self.frame_work_plane))
                return
        print(self._T_cam_to_work_plane)
        print(self._K)
        if self._T_cam_to_work_plane is not None and self._K is not None: # init done!
            self.timer_tf.shutdown()
            self.ready = True
            rospy.loginfo("bird view init done!")

        


if __name__ == '__main__':
    # test
    img = cv2.imread('image.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.imshow(img, cmap='gray')
    plt.show()

    K = np.array([  [1124.66943359375, 0.0, 505.781982421875],
                [0.0, 1124.6165771484375, 387.8110046386719],
                [0.0, 0.0, 1.0]])

    # bv = BirdView(K, T_cam_to_work_plane, (-20, 5), (20, 50))
    bv = BirdViewRos('base_link', 'camera_forward_optical_frame', '/forwardCamera/camera_infoq', (-20, 5), (20, 50))
    
    time_start = time.time()

    img_bird = bv.project_img_to_bird(img)

    time_end = time.time()
    print("time: ", time_end - time_start)

    pt_px_cam = bv.project_work_plane_pt_to_source_img(np.array([20., 50.]))
    print(pt_px_cam)

    pt_px_bird = bv.get_work_plane_pt_in_bird_img(np.array([-20., 5.]))
    print(pt_px_bird)

    pt_work_plane = bv.get_bird_img_pt_in_work_plane(np.array([800., 900.]))
    print(pt_work_plane)

    pt_work_plane = bv.project_source_img_pt_to_work_plane(np.array([979.47613383, 451.36251473]))
    print(pt_work_plane)

    plt.imshow(img_bird, cmap='gray')
    plt.show()

