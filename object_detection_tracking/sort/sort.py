import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.H = np.zeros((4, 7))
        self.kf.H[:4, :4] = np.eye(4)
        self.kf.P *= 10.
        self.kf.R *= 1.
        self.kf.Q *= 0.01
        self.kf.x[:4] = bbox.reshape((4, 1))

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.kf.update(bbox.reshape((4, 1)))

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x[:4].reshape((4,))

class Sort:
    def __init__(self):
        self.trackers = []

    def update(self, dets):
        updated = []
        for det in dets:
            if len(self.trackers) == 0:
                self.trackers.append(KalmanBoxTracker(det))
            else:
                self.trackers[0].update(det)
        for t in self.trackers:
            updated.append(np.append(t.predict(), t.id))
        return np.array(updated)
