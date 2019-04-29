# -*- coding:utf-8 -*-
import numpy as np
import math
from sklearn.cluster import KMeans
# from tensorflow.python import pywrap_tensorflow
import os
import struct
from fastcluster import Fastcluster
from HuffmanCoding import huffman_encode, huffman_decode, huffman_store, huffman_restore


class Compress(object):
    def __init__(self):
        pass

    def prune_edges_with_small_weight(self,ndarray,percent):
        weights = ndarray.flatten()
        abso = np.absolute(weights)
        print(weights.size)
        threshold = np.sort(abso)[int(math.ceil(weights.size * percent))]
        weights[abso < threshold] = 0
        print(threshold)
        return  weights.reshape(ndarray.shape)

    # Turn absolute index to relative index(difference index), and if the diff index exceed the bound(max_index),padding zero
    def relative_index(self,absolute_index, ndarray, max_index):
        first = absolute_index[0]
        relative = np.insert(np.diff(absolute_index), 0, first)
        dense = ndarray.tolist()
        max_index_or_less = relative.tolist()
        shift = 0

        for i in np.where(relative > max_index)[0].tolist():
            # minus max_index every iteration until max_index_or_less[i] less than max_index
            while max_index_or_less[i+shift] > max_index:
                max_index_or_less.insert(i+shift,max_index)
                dense.insert(i+shift,0)
                shift+=1
                max_index_or_less[i+shift] -= max_index
        return (np.array(max_index_or_less), np.array(dense))


    def store_compressed_network(self, path, layers):
        with open(path, 'wb') as f:
            for layer in layers:
                huffman_store(f, layer[0])   # Q:8B
                f.write(struct.pack('Q', layer[1].size))
                f.write(layer[1].tobytes())
                f.write(struct.pack('Q',layer[2].size))
                f.write(layer[2].tobytes())
                f.write(struct.pack('Q', layer[3].size))
                f.write(layer[3].tobytes())
                print("layer info: ",len(layer[0]), layer[1].size, layer[2].size, layer[3].size)
        return path

    '''
    relative_index_in_4bits: indicate the position of non-element;
    pair_of_4bits_in_1byte： combine the adjacent two elements of relative_index_in_4bits into one byte;
    cluster_labels: the non-element's label;
    clusters.cluster_centers_: the codebook;
    store pair_of_4bits_in_1byte,cluster_labels,clusters.cluster_centers_;
    input:
    ndarray: the ndarray to compress
    pruning_percent: the percent of pruning;
    cluster_num: the number of clusters in KMeans
    '''
    def compress_a_ndarray(self,ndarray,pruning_percent,cluster_num):
        sparse_1d = self.prune_edges_with_small_weight(ndarray,pruning_percent).flatten()
        nonzero = sparse_1d[sparse_1d != 0]  # remove the non-zero elements of array
        #clusters = KMeans(n_clusters=cluster_num).fit(nonzero.reshape(-1,1).astype(np.float32))
        clusters = Fastcluster(nonzero,cluster_num)  # ues my faster cluster methods
        print("clusters: ",clusters.cluster_centers_.shape,clusters.labels_.shape)
        print(clusters.cluster_centers_,clusters.labels_)
        relative_index_in_4bits, cluster_labels = self.relative_index(np.where(sparse_1d !=0)[0],clusters.labels_+1,max_index=16-1)
        print(relative_index_in_4bits)

        huffman_info = huffman_encode(relative_index_in_4bits)
        cluster_centers_ = clusters.cluster_centers_.astype(np.float32)
        cluster_labels = cluster_labels.astype(np.uint8)  # convert int32 to uint8

        return huffman_info, cluster_labels, cluster_centers_  #clusters.cluster_centers_

    def compress_a_layer(self,weights,biases,pruning_percent,cluster_num):
        # compress the weights
        huffman_info, cluster_labels,cluster_centers_ = self.compress_a_ndarray(weights,pruning_percent,cluster_num)
        compent = (huffman_info, cluster_labels, cluster_centers_, biases)  # 偏置没有压缩
        print("the compressing data type: ", type(huffman_info), cluster_labels.dtype, cluster_centers_.dtype, biases.dtype)
        return compent


    ''' target_layers is a list like [(weight1,bias1),(weight2,bias2),...]'''
    def compress_network(self,target_layers, pruning_percent, cluster_num, store_path_prefix):
        components = []
        for layer in target_layers:
            weights = layer[0]
            biases = layer[1]
            component = self.compress_a_layer(weights,biases,pruning_percent,cluster_num)
            components.append(component)
            print("the layer information:")
            print("compressed size: (except component[0])", component[1].size, component[2].size, component[3].size)
            print("a layer compressed")
        return self.store_compressed_network(store_path_prefix+'_'+str(pruning_percent)+'_'+str(cluster_num)+'_clusters.npy', components)

    '''
    decode the compressed ndarray
    shape:the shape of ndarray wanted to get
    '''
    def decode_a_ndarray(self,shape, _huffman_info,cluster_labels,cluster_centers):
        weights = np.zeros(shape, np.float32).flatten()
        diff_indices = huffman_decode(_huffman_info)
        index = np.cumsum(diff_indices)
        weights[index] = np.insert(cluster_centers, 0, 0)[cluster_labels]
        return weights.reshape(shape)

    def decode_a_layer(self, w_shape, _huffman_info,cluster_labels,sharing_weights,biases):
        weights = self.decode_a_ndarray(w_shape, _huffman_info, cluster_labels, sharing_weights)
        return weights, biases


    '''
    path: the file path
    target_layers:[w1.shape,w2.shape,...]
    '''
    def decode_network(self,path,target_layers):
        f = open(path, 'rb')
        components = []
        for w_shape in target_layers:
            huffman_info = huffman_restore(f)
            num1 = np.fromfile(f, dtype=np.int64, count=1)
            cluster_labels = np.fromfile(f, dtype=np.uint8, count=num1[0])
            num2 = np.fromfile(f, dtype=np.int64, count=1)
            sharing_weights = np.fromfile(f, dtype=np.float32, count=num2[0])
            num3 = np.fromfile(f, dtype=np.int64, count=1)
            biases = np.fromfile(f, dtype=np.float32, count=num3[0])
            print("decode shape: ", cluster_labels.shape, sharing_weights.shape, biases.shape)
            weights, biases = self.decode_a_layer(w_shape, huffman_info, cluster_labels, sharing_weights, biases)
            components.append((weights, biases))

        return components



if __name__ == '__main__':
    arr = np.random.randint(0,255,(1000))
    cm = Compress()
    huffman_info, cluster_labels, cluster_centers_ = cm.compress_a_ndarray(arr,0.2,10)
    print("huffman_info:\n",huffman_info )
    print("cluster_labels:\n",cluster_labels )
    print("cluster_centers_:\n",cluster_centers_)






