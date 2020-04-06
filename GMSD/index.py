import cv2
import numpy as np
from scipy import signal

ori_img = cv2.imread('ori.bmp', cv2.IMREAD_COLOR)
dist_img = cv2.imread('dist.bmp', cv2.IMREAD_COLOR)

ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
dist_img = cv2.cvtColor(dist_img, cv2.COLOR_BGR2GRAY)

ori_img = np.float64(ori_img)
dist_img = np.float64(dist_img)

ave_kernal = np.array([[0.25, 0.25],
                       [0.25, 0.25]])

ave_ori = signal.convolve2d(ori_img, ave_kernal, boundary='fill', mode='full')
(M, N) = ave_ori.shape
ave_ori = ave_ori[1:M, 1:N]

ave_dist = signal.convolve2d(dist_img, ave_kernal, boundary='fill', mode='full')
(M, N) = ave_dist.shape
ave_dist = ave_dist[1:M, 1:N]

ori_img = ave_ori[::2, ::2]
dist_img = ave_dist[::2, ::2] 

Prewitt_x = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0, -1]])/3
Prewitt_x = np.float64(Prewitt_x)
Prewitt_y = np.transpose(Prewitt_x)

ori_GM_x = signal.convolve2d(ori_img, Prewitt_x, boundary='fill', mode='full')
(M, N) = ori_GM_x.shape
ori_GM_x = ori_GM_x[1:M-1, 1:N-1]

ori_GM_y = signal.convolve2d(ori_img, Prewitt_y, boundary='fill', mode='full')
(M, N) = ori_GM_y.shape
ori_GM_y = ori_GM_y[1:M-1, 1:N-1]

ori_GM = np.sqrt(np.square(ori_GM_x) + np.square(ori_GM_y))

dist_GM_x = signal.convolve2d(dist_img, Prewitt_x, boundary='fill', mode='full')
(M, N) = dist_GM_x.shape
dist_GM_x = dist_GM_x[1:M-1, 1:N-1]

dist_GM_y = signal.convolve2d(dist_img, Prewitt_y, boundary='fill', mode='full')
(M, N) = dist_GM_y.shape
dist_GM_y = dist_GM_y[1:M-1, 1:N-1]

dist_GM = np.sqrt(np.square(dist_GM_x) + np.square(dist_GM_y))

T = 170
quality_map = (2*ori_GM*dist_GM + 170)/(np.square(ori_GM) + np.square(dist_GM) + T)

score = np.std(quality_map)
print("score:", score)
