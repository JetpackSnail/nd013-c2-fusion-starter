# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
  
    def associate(self, track_list, meas_list, KF):
        N = len(track_list) # N tracks
        M = len(meas_list) # M measurements

        # initialize association matrix
        self.association_matrix = np.inf * np.ones((N,M)) 

        # loop over all tracks and all measurements to set up association matrix
        for i in range(N): 
            track = track_list[i]
            for j in range(M):
                meas = meas_list[j]
                dist = self.MHD(track, meas)

                if self.gating(MHD=dist, sensor=meas.sensor):
                    self.association_matrix[i,j] = dist

        self.unassigned_tracks = np.arange(len(track_list)).tolist()
        self.unassigned_meas = np.arange(len(meas_list)).tolist()

        return

    def get_closest_track_and_meas(self):
        if self.association_matrix.min == np.inf:
            return np.nan, np.nan

        min_row, min_col = np.unravel_index(np.argmin(self.association_matrix, axis=None), self.association_matrix.shape)
        self.association_matrix = np.delete(self.association_matrix, min_row, 0)
        self.association_matrix = np.delete(self.association_matrix, min_col, 1)

        update_track = self.unassigned_tracks[min_row] 
        update_meas = self.unassigned_meas[min_col]

        # remove from list
        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)

        return update_track, update_meas     

    def gating(self, MHD, sensor):
        limit = chi2.ppf(params.gating_threshold, df=sensor.dim_meas)
        if MHD < limit:
            return True
        else:
            return False
 
    def MHD(self, track, meas):
        z = np.matrix(meas.z)
        z_pred = meas.sensor.get_hx(track.x)
        hx = z - z_pred 
        S = meas.R
        MHD = hx.T * S.I * hx

        return MHD

    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)

        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)
