# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import deque
#deque 双端队列，用于存储特征历史记录
import numpy as np
from .basetrack import TrackState
#表示跟踪的状态，Tracked已跟踪、Lost丢失
from .byte_tracker import BYTETracker, STrack
from .utils import matching
#匹配算法模块，用于目标与检测结果的匹配
from .utils.gmc import GMC
#全局运动补偿模块，考虑背景运动影响
#from .utils.kalman_filter0 import KalmanFilterXYWH
from .utils.kalman_filter2 import KalmanFilterXYWH
#卡尔曼滤波器模块，用于目标状态预测和更新

#对STrack的扩张，用于跟踪目标，增加了平滑特征的功能，采用卡尔曼滤波器实现状态预测
class BOTrack(STrack):
    """
    An extended version of the STrack class for YOLOv8, adding object tracking features.

    Attributes:
        shared_kalman (KalmanFilterXYWH): A shared Kalman filter for all instances of BOTrack.
        smooth_feat (np.ndarray): Smoothed feature vector.
        curr_feat (np.ndarray): Current feature vector.
        features (deque): A deque to store feature vectors with a maximum length defined by `feat_history`.
        alpha (float): Smoothing factor for the exponential moving average of features.
        mean (np.ndarray): The mean state of the Kalman filter.
        covariance (np.ndarray): The covariance matrix of the Kalman filter.

    Methods:
        update_features(feat): Update features vector and smooth it using exponential moving average.
        predict(): Predicts the mean and covariance using Kalman filter.
        re_activate(new_track, frame_id, new_id): Reactivates a track with updated features and optionally new ID.
        update(new_track, frame_id): Update the YOLOv8 instance with new track and frame ID.
        tlwh: Property that gets the current position in tlwh format `(top left x, top left y, width, height)`.
        multi_predict(stracks): Predicts the mean and covariance of multiple object tracks using shared Kalman filter.
        convert_coords(tlwh): Converts tlwh bounding box coordinates to xywh format.
        tlwh_to_xywh(tlwh): Convert bounding box to xywh format `(center x, center y, width, height)`.

    Usage:
        bo_track = BOTrack(tlwh, score, cls, feat)
        bo_track.predict()
        bo_track.update(new_track, frame_id)
    """

    shared_kalman = KalmanFilterXYWH()

    #初始化
    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):
        """Initialize YOLOv8 object with temporal parameters, such as feature history, alpha and current features."""
        super().__init__(tlwh, score, cls)
        #初始化目标框：（topleftx，toplefty，w,h）
        #score:目标置信度
        #cls：种类
        self.smooth_feat = None  #平滑特征
        self.curr_feat = None   #当前特征
        if feat is not None:
            self.update_features(feat) #更新特征
        self.features = deque([], maxlen=feat_history) #保存特征历史记录
        self.alpha = 0.9  #平滑因子
        #平滑因子alpha：数学上为指数加权移动平均（EMA）
        #smooth_feat=alpha*smooth_feat+(1-alpha)*curr_feat
    #更新特征
    def update_features(self, feat):
        """Update features vector and smooth it using exponential moving average."""
        feat /= np.linalg.norm(feat) #特征归一化，保证特征向量的长度为1
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat) #平滑特征归一化
    #预测
    def predict(self):
        """Predicts the mean and covariance using Kalman filter."""
        #使用卡尔曼滤波器预测的状态 x，p
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivates a track with updated features and optionally assigns a new ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)

    def update(self, new_track, frame_id):
        """Update the YOLOv8 instance with new track and frame ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y, width, height)`."""
        if self.mean is None:   #若没有预测，直接返回初始框
            return self._tlwh.copy()
        ret = self.mean[:4].copy() #（x,y,w,h)
        ret[:2] -= ret[2:] / 2   #x=x-w/2,y=y-h/2
        return ret
    #利用共享卡尔曼滤波器对所有目标进行统一预测，计算效率高
    @staticmethod
    def multi_predict(stracks):
        """Predicts the mean and covariance of multiple object tracks using shared Kalman filter."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh):
        """Converts Top-Left-Width-Height bounding box coordinates to X-Y-Width-Height format."""
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width, height)`."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


class BOTSORT(BYTETracker):
    """
    An extended version of the BYTETracker class for YOLOv8, designed for object tracking with ReID and GMC algorithm.

    Attributes:
        proximity_thresh (float): Threshold for spatial proximity (IoU) between tracks and detections.
        appearance_thresh (float): Threshold for appearance similarity (ReID embeddings) between tracks and detections.
        encoder (object): Object to handle ReID embeddings, set to None if ReID is not enabled.
        gmc (GMC): An instance of the GMC algorithm for data association.
        args (object): Parsed command-line arguments containing tracking parameters.

    Methods:
        get_kalmanfilter(): Returns an instance of KalmanFilterXYWH for object tracking.
        init_track(dets, scores, cls, img): Initialize track with detections, scores, and classes.
        get_dists(tracks, detections): Get distances between tracks and detections using IoU and (optionally) ReID.
        multi_predict(tracks): Predict and track multiple objects with YOLOv8 model.

    Usage:
        bot_sort = BOTSORT(args, frame_rate)
        bot_sort.init_track(dets, scores, cls, img)
        bot_sort.multi_predict(tracks)

    Note:
        The class is designed to work with the YOLOv8 object detection model and supports ReID only if enabled via args.
    """

    def __init__(self, args, frame_rate=30):
        """Initialize YOLOv8 object with ReID module and GMC algorithm."""
        super().__init__(args, frame_rate)
        # ReID module
        self.proximity_thresh = args.proximity_thresh   #空间接近阈值：基于IOU交并比计算
        self.appearance_thresh = args.appearance_thresh  #外观相似度阈值：使用Reid特征计算，衡量外观的匹配程度

        if args.with_reid:
            # Haven't supported BoT-SORT(reid) yet
            self.encoder = None   #ReID编码器
        self.gmc = GMC(method=args.gmc_method)  #GMC模块

    def get_kalmanfilter(self):
        """Returns an instance of KalmanFilterXYWH for object tracking."""
        return KalmanFilterXYWH()

    def init_track(self, dets, scores, cls, img=None):
        """Initialize track with detections, scores, and classes."""
        if len(dets) == 0:
            return []
        if self.args.with_reid and self.encoder is not None:
            features_keep = self.encoder.inference(img, dets)
            return [BOTrack(xyxy, s, c, f) for (xyxy, s, c, f) in zip(dets, scores, cls, features_keep)]  # detections
        else:
            return [BOTrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)]  # detections
    #关联距离
    def get_dists(self, tracks, detections):
        """Get distances between tracks and detections using IoU and (optionally) ReID embeddings."""
        #使用IOU和REID计算轨迹与检测距离
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > self.proximity_thresh

        # TODO: mot20
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)

        if self.args.with_reid and self.encoder is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
        return dists

    def multi_predict(self, tracks):
        """Predict and track multiple objects with YOLOv8 model."""
        BOTrack.multi_predict(tracks)

    def reset(self):
        """Reset tracker."""
        super().reset()
        self.gmc.reset_params()
