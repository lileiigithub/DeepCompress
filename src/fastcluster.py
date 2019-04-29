__author__ = 'LiLei'

import numpy as np

'''a method for faster cluster camparing with KMeans'''
class Fastcluster(object):
    def __init__(self,ndarray,cluster_nums):
        self.ndarray = ndarray
        self.cluster_nums = cluster_nums
        if self.cluster_nums>self.ndarray.size:
            assert 0
        self.cluster_centers_ = []
        self.labels_ = []
        self.fastCluster()

    def fastCluster(self):
        array_1d = self.ndarray.flatten()
        array_1d_sorted = -np.sort(-array_1d) # descending sort
        step =  self.ndarray.size// self.cluster_nums
        index = 0
        for i in range( self.cluster_nums):
            centrid = array_1d_sorted[index]
            index=index+step
            self.cluster_centers_.append(centrid)
        for item in (array_1d):
            distance = np.absolute(item- self.cluster_centers_)   # L1 distance
            #distance = np.absolute((item- self.cluster_centers_)*(item- self.cluster_centers_))  # # L2 distance
            self.labels_.append(np.where(distance ==(np.min(distance)))[0][0])

        self.cluster_centers_ = np.array(self.cluster_centers_,dtype=np.float32)
        self.labels_ = np.array(self.labels_)
# cluster = Fastcluster(np.random.rand(1000),50)
# print(type(cluster.cluster_centers_),type(cluster.labels_))
