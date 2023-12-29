import cv2
import numpy as np
import matplotlib.pyplot as plt
import time



class bird_view:

    def __init__(self, K, T_cam_to_ground, pos_start_area, pos_end_area):
        self.K = K
        self.T_cam_to_ground = T_cam_to_ground
        self.scale = 20 # 1m = <scale> px

        self.pos_start_area = np.array(pos_start_area)
        self.pos_end_area = np.array(pos_end_area)
        
        self._init_projection_parameters()

    
    def _init_projection_parameters(self):

        pts_ground = np.array([[0., 0., 0., 1.],
                       [1., 0., 0., 1.],
                       [0., 1., 0., 1.],
                       [1., 1., 0., 1.]])
        pts_ground = pts_ground.T

        pts_cam = np.linalg.inv(T_cam_to_ground) @ pts_ground

        pts_sensor = pts_cam[:2] / pts_cam[2]
        pts_sensor = np.vstack((pts_sensor, np.ones((1,4))))

        pts_img = K @ pts_cam[:3]
        pts_img = pts_img / pts_img[2]

        pts_ground_img = pts_ground[:2].copy()
        pts_ground_img *= self.scale
        pts_ground_img = pts_ground_img.T
        pts_ground_img -= self.pos_start_area * self.scale
        pts_ground_img = pts_ground_img.astype(np.float32)

        self.H = cv2.getPerspectiveTransform(pts_img[:2].T.astype(np.float32), pts_ground_img)
    
    def project(self, img):
        size = (self.pos_end_area - self.pos_start_area) * self.scale
        print(size)
        return cv2.warpPerspective(img, self.H, (int(size[0]), int(size[1])))


if __name__ == '__main__':
    img = cv2.imread('image.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # plt.imshow(img, cmap='gray')
    # plt.show()

    K = np.array([  [1124.66943359375, 0.0, 505.781982421875],
                [0.0, 1124.6165771484375, 387.8110046386719],
                [0.0, 0.0, 1.0]])

    T_cam_to_ground = np.array([[ 0.99991969 , 0.01149682 ,-0.00533206 , 0.05419943],
    [ 0.00510958 , 0.01929436 , 0.99980079 , 1.96159697],
    [ 0.0115974  ,-0.99974774  ,0.01923406  ,1.55057154],
        [ 0.          ,0.          ,0.          ,1.        ]])
    

    bird_view = bird_view(K, T_cam_to_ground, (-20, 5), (20, 50))

    time_start = time.time()

    img_bird = bird_view.project(img)

    time_end = time.time()
    print("time: ", time_end - time_start)

    plt.imshow(img_bird, cmap='gray')
    plt.show()

