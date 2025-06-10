# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import scipy.linalg


class KalmanFilterXYAH:
    """
    For bytetrack. A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space (x, y, a, h, vx, vy, va, vh) contains the bounding box center position (x, y), aspect
    ratio a, height h, and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location (x, y, a, h) is taken as direct
    observation of the state space (linear observation model).
    """
#这段代码是初始化一个卡尔曼滤波器模型，其中包含了运动和观测不确定性相关的权重
    def __init__(self):
        """Initialize Kalman filter model matrices with motion and observation uncertainty weights."""
        #ndim是维度4 dt=时间步长 即每次状态更新之间的时间间隔，表示每次状态更新的时间间隔为1秒
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices
        #创建了一个2*dim大小的单位矩阵，2*4=8，8x8单位矩阵  状态变量的维度
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        #使得运动矩阵的结构符合常见的线性运动模型  x(t+1) = x(t) + v(t) * dt
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        #self._motion_mat：运动矩阵，用于描述从一个状态到下一个状态的转移过程（包括位置和速度的关系）。
        #self._update_mat：更新矩阵，用于从状态估计中提取位置部分进行观测更新

        #self._update_mat=4x8的矩阵 更新矩阵
        #它用于表示如何从高维的状态空间（包括位置和速度）中选择出需要的部分来进行观测更新。
        # 由于在状态向量中有位置和速度，更新矩阵控制了如何从完整的状态向量中提取出位置的部分
        # 观测量的维度
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current state estimate. These weights control
        # the amount of uncertainty in the model.
        #位置和速度的不确定性权重
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        #kalman 的不确定权重控制了对每个测量数据的不确定性的权重， 位置的标准差相对较大，表示位置的估计可能不太准确或不确定，位置的不确定性越大，卡尔曼滤波器越依赖于测量数据而非预测模型

    def initiate(self, measurement: np.ndarray) -> tuple:
        """
        Create track from unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, a, h) with center position (x, y), aspect ratio a,
                and height h.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector (8 dimensional) and covariance matrix (8x8 dimensional) of
                the new track. Unobserved velocities are initialized to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8 dimensional mean vector of the object state at the previous time step.
            covariance (ndarray): The 8x8 dimensional covariance matrix of the object state at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            (tuple[ndarray, ndarray]): Returns the projected mean and covariance matrix of the given state estimate.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple:
        """
        Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4 dimensional measurement vector (x, y, a, h), where (x, y) is the center
                position, a the aspect ratio, and h the height of the bounding box.

        Returns:
            (tuple[ndarray, ndarray]): Returns the measurement-corrected state distribution.
        """
        #投影步骤
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        #卡尔曼增益
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        # 观测值和预测值之间的差异，修正的核心 当前时刻的状态变量 -x 差异量
        #后验估计
        innovation = measurement - projected_mean
       # z-x


        new_mean = mean + np.dot(innovation, kalman_gain.T)

        #后验估计协方差
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """
        Compute gating distance between state distribution and measurements. A suitable distance threshold can be
        obtained from `chi2inv95`. If `only_position` is False, the chi-square distribution has 4 degrees of freedom,
        otherwise 2.

        Args:
            mean (ndarray): Mean vector over the state distribution (8 dimensional).
            covariance (ndarray): Covariance of the state distribution (8x8 dimensional).
            measurements (ndarray): An Nx4 matrix of N measurements, each in format (x, y, a, h) where (x, y)
                is the bounding box center position, a the aspect ratio, and h the height.
            only_position (bool, optional): If True, distance computation is done with respect to the bounding box
                center position only. Defaults to False.
            metric (str, optional): The metric to use for calculating the distance. Options are 'gaussian' for the
                squared Euclidean distance and 'maha' for the squared Mahalanobis distance. Defaults to 'maha'.

        Returns:
            (np.ndarray): Returns an array of length N, where the i-th element contains the squared distance between
                (mean, covariance) and `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)  # square maha
        else:
            raise ValueError("Invalid distance metric")


class KalmanFilterXYWH(KalmanFilterXYAH):
    """
    For BoT-SORT. A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space (x, y, w, h, vx, vy, vw, vh) contains the bounding box center position (x, y), width
    w, height h, and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location (x, y, w, h) is taken as direct
    observation of the state space (linear observation model).
    """
#一、初始化
    def initiate(self, measurement: np.ndarray) -> tuple:
        """
        Create track from unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, w, h) with center position (x, y), width, and height.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector (8 dimensional) and covariance matrix (8x8 dimensional) of
                the new track. Unobserved velocities are initialized to 0 mean.
        """
#一、初始化状态矩阵和协方差矩阵——————————————————————————————————————————————————————————————————————————————————————
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel] #1.初始状态矩阵 mean=[x,y,w,h,0,0,0,0]
#对目标位置和速度估计的标准差  _std_weight_position是权重系数 控制位置相关的标准差 通常小于1 控制速度相关的标准差，通常大于1
# 标准差是位置测量值的两倍，这个因子说明了位置测量不确定性的大小  速度：标准差是速度测量值的10倍，大是因为速度受到噪声的影响可能更严重 2,3 只是对于w h 嘛 这些标准差决定了目标状态估计中的不确定性
        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
# 2.初始协方差矩阵，通常使用测量的标准差来初始化，w和h的标准差基于一个权重 _std_weight_position ，v速度的标准差使用一个更大的权重进行初始化
        covariance = np.diag(np.square(std))
#这种操作通常用于构造协方差矩阵的简化版本。协方差矩阵表示数据中各个变量之间的关系，通常是一个对称矩阵。
# 如果变量之间是独立的且没有协方差，那么协方差矩阵就是一个对角矩阵，其中每个对角线上的元素就是每个变量的方差（标准差的平方）。
# 所以，这行代码实际是在构建一个标准差的平方（方差）组成的对角矩阵，通常用于某些统计学或机器学习算法中，比如生成一个简单的协方差矩阵。
        return mean, covariance

#二、卡尔曼滤波的预测步骤，基于当前状态均值和协方差预测下一时刻的状态——————————————————————————————————————————————————————————————————————————————————————————
    def predict(self, mean, covariance) -> tuple:
        """
        Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8 dimensional mean vector of the object state at the previous time step.
            covariance (ndarray): The 8x8 dimensional covariance matrix of the object state at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        # ._motion_mat 状态转移矩阵 在父类上面  np.square 是平方 协方差矩阵？
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        #np.dot 矩阵的乘法 mean 状态矩阵  self._motion_mat 是一个矩阵，通常表示某种“运动模型”或转换矩阵，可能用于描述从一个时间步到下一个时间步的状态转移。.T 是对该矩阵进行转置操作。
        #前验估计 在预测步骤 对x进行状态估计
        mean = np.dot(mean, self._motion_mat.T) #mena'=mean * _motion_mat.T
        #预测协方差 covariance' = _motion_mat * covariance * _motion_mat.T + motion_cov
        #前验估计协方差  #w是
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance
#三、将状态预测到观测空间，计算在观测空间中的均值和协方差 （是要预测为我们的新的xywh？）
    # 更新步骤
    def project(self, mean, covariance) -> tuple:
        """
        Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            (tuple[ndarray, ndarray]): Returns the projected mean and covariance matrix of the given state estimate.
        """
        #观测噪声协方差
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std)) #这是观测噪声的协方差
#_update_mat 将状态空间中的状态（8维）————>观测空间中的观测（4维） 卡尔曼滤波在此模型中直接使用了边界框的位置（x，y，w，h）作为观测矩阵
        # 投影后的状态 新的状态矩阵=状态转换矩阵和原来的状态矩阵相乘
        mean = np.dot(self._update_mat, mean)
        #投影后的协方差 这几个矩阵相乘
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov
#四、这是一个向量化版本的预测步骤，可以处理多个目标的状态
    def multi_predict(self, mean, covariance) -> tuple:
        """
        Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        #通过批量操作
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance
#五、卡尔曼滤波器的修正步骤，根据实际观测来调整预测的状态
    def update(self, mean, covariance, measurement) -> tuple:
        """
        Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4 dimensional measurement vector (x, y, w, h), where (x, y) is the center
                position, w the width, and h the height of the bounding box.

        Returns:
            (tuple[ndarray, ndarray]): Returns the measurement-corrected state distribution.
        """
        return super().update(mean, covariance, measurement)

