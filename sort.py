"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    # 尝试导入 lap 模块，使用 lapjv 函数解决线性分配问题
    import lap
    # 调用 lapjv 函数，返回任务与代理之间的最佳匹配索引 (y, x)
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    # 将结果处理成数组形式，其中 y[i] 表示任务 i 分配给的代理，返回这个数组
    return np.array([[y[i], i] for i in x if i >= 0])
  except ImportError:
    # lap 模块导入失败，使用 scipy 库的 linear_sum_assignment 函数解决线性分配问题
    from scipy.optimize import linear_sum_assignment
    # 调用 linear_sum_assignment 函数，返回任务与代理之间的最佳匹配索引 (x, y)
    x, y = linear_sum_assignment(cost_matrix)
    # 将结果处理成数组形式，返回这个数组
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  计算两个边界框之间的 Intersection over Union (IoU)。

  参数:
      bb_test: 测试边界框，格式为 [x1, y1, x2, y2]。
      bb_gt: 真实边界框，格式为 [x1, y1, x2, y2]。

  返回:
      IoU 值。
  """
  # 将真实边界框和测试边界框的形状调整，使其适应后续运算
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)

  # 计算交集的坐标
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

  # 计算交集和并集的宽度和高度
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h

  # 计算 IoU 值
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) +
            (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)

  return o


def convert_bbox_to_z(bbox):
  """
  将边界框从 [x1, y1, x2, y2] 格式转换为 [x, y, s, r] 格式，
  其中 x, y 为边界框中心坐标，s 为面积/尺度，r 为宽高比。

  参数:
      bbox: 边界框，格式为 [x1, y1, x2, y2]。

  返回:
      转换后的 z 值，格式为 [x, y, s, r]。
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w / 2.
  y = bbox[1] + h / 2.
  s = w * h  # 面积即尺度
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
  """
  将中心形式的边界框 [x, y, s, r] 转换为标准形式 [x1, y1, x2, y2] 的函数。

  参数:
  - x (numpy.ndarray): 输入的中心形式边界框，其中 `x` 表示中心坐标，`s` 是尺度/面积，`r` 是宽高比。
  - score (float, 可选): 与边界框关联的置信度分数。

  返回:
  numpy.ndarray: 转换后的标准形式边界框 [x1, y1, x2, y2] 或带有分数的形式 [x1, y1, x2, y2, score]（如果提供了分数）。
  """
  # 从尺度和宽高比计算宽度和高度
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w

  # 计算边界框的左上角和右下角坐标
  if score is None:
    return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
  else:
    return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    这个类表示以bbox形式观察到的单个被跟踪对象的内部状态。
    """
    count = 0

    def __init__(self, bbox):
      """
      使用初始边界框初始化跟踪器。
      """
      # 定义恒定速度模型
      self.kf = KalmanFilter(dim_x=7, dim_z=4)
      self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                            [0, 1, 0, 0, 0, 1, 0],
                            [0, 0, 1, 0, 0, 0, 1],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1]])
      self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0]])

      self.kf.R[2:, 2:] *= 10.
      self.kf.P[4:, 4:] *= 1000.  # 给不可观测的初始速度提供高不确定性
      self.kf.P *= 10.
      self.kf.Q[-1, -1] *= 0.01
      self.kf.Q[4:, 4:] *= 0.01

      self.kf.x[:4] = convert_bbox_to_z(bbox)
      self.time_since_update = 0
      self.id = KalmanBoxTracker.count
      KalmanBoxTracker.count += 1
      self.history = []
      self.hits = 0
      self.hit_streak = 0
      self.age = 0

    def update(self, bbox):
      """
      使用观察到的bbox更新状态向量。
      """
      self.time_since_update = 0
      self.history = []
      self.hits += 1
      self.hit_streak += 1
      self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
      """
      推进状态向量并返回预测的边界框估计。
      """
      if (self.kf.x[6] + self.kf.x[2]) <= 0:
        self.kf.x[6] *= 0
      self.kf.predict()
      self.age += 1
      if (self.time_since_update > 0):
        self.hit_streak = 0
      self.time_since_update += 1
      self.history.append(convert_x_to_bbox(self.kf.x))
      return self.history[-1]

    def get_state(self):
      """
      返回当前边界框估计。
      """
      return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    初始化SORT算法的关键参数。

    Parameters:
    - max_age (int): 最大帧数，跟踪器在超过此帧数后将被删除。
    - min_hits (int): 最小命中数，仅在命中数达到此值或总帧数小于等于此值时返回目标。
    - iou_threshold (float): IOU（交并比）阈值，用于关联检测和跟踪器。

    Attributes:
    - max_age (int): 最大帧数。
    - min_hits (int): 最小命中数。
    - iou_threshold (float): IOU阈值。
    - trackers (list): 存储跟踪器的列表。
    - frame_count (int): 帧计数。
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    """
    更新跟踪器状态。

    Parameters:
    - dets (numpy.ndarray): 一个包含检测结果的numpy数组，格式为[[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...]。

    Returns:
    numpy.ndarray: 包含更新后的目标状态，最后一列是对象ID。

    Note:
    返回的目标数量可能与提供的检测数量不同。
    """
    self.frame_count += 1

    # 获取现有跟踪器的预测位置
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)

    # 删除无效的跟踪器
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    # 关联检测与跟踪器
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

    # 更新匹配的跟踪器状态
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # 创建新的跟踪器
    for i in unmatched_dets:
      trk = KalmanBoxTracker(dets[i, :])
      self.trackers.append(trk)

    i = len(self.trackers)
    # 移除失效的跟踪器
    for trk in reversed(self.trackers):
      d = trk.get_state()[0]
      if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
        ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
      i -= 1
      if (trk.time_since_update > self.max_age):
        self.trackers.pop(i)

    if len(ret) > 0:
      return np.concatenate(ret)
    return np.empty((0, 5))

