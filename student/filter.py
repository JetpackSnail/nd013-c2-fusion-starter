# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        # [px, py, pz, vx, vy, vz]
        self.dim_state = params.dim_state
        self.dt = params.dt
        self.q = params.q

    def F(self):
        # Constant velocity process matrix
        F = [[1,   0,   0,  self.dt,   0,     0],
             [0,   1,   0,    0,    self.dt,  0],
             [0,   0,   1,    0,       0,  self.dt],
             [0,   0,   0,    1,       0,     0],
             [0,   0,   0,    0,       1,     0],
             [0,   0,   0,    0,       0,     1]]

        return np.matrix(F)

    def Q(self):
        Q = np.identity(self.dim_state) * self.q * self.dt

        return np.matrix(Q)

    def predict(self, track):
        # predict state and estimation error covariance to next timestep
        F = self.F()
        Q = self.Q()

        x = F * track.x # state prediction
        P = F * track.P * F.transpose() + Q # covariance prediction
        
        track.set_x(x)
        track.set_P(P)

    def update(self, track, meas):
        H = meas.sensor.get_H(track.x) # measurement matrix
        gamma = self.gamma(track, meas) # residual
        
        S = self.S(track, meas, H) # covariance of residual
        K = track.P * H.transpose() * np.linalg.inv(S) # Kalman gain
        x = track.x + K * gamma # state update
        I = np.identity(self.dim_state)
        P = (I - K * H) * track.P # covariance update

        track.set_x(x)
        track.set_P(P)
        track.update_attributes(meas)

    def gamma(self, track, meas):
        return meas.z - meas.sensor.get_hx(track.x)

    def S(self, track, meas, H):
        return H * track.P * H.transpose() + meas.R
