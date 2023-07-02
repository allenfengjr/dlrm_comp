# This is a class to compress embedding vector outputs
# Note this is not a real workflow, it just for meature compress ratio and accuracy after compress
from re import S
from unittest import case
import zfpy
import numpy as np
from scipy import sparse
from pathlib import Path
from pysz import SZ
import platform
import torch
import os

def reformat(self,format,data):
    """
    The input data should be a python array, the length of it should be #table as the shape in `dlrm_s_pytorch.py`
    """
    data = np.array(data)
    num_rank = os.environ.get("WORLD_SIZE")
    processed_data = []
    if format == "table_wise":
        for i in range(len(data)):
            processed_data.append(data[i])
    elif format == "column_wise":
        for rank in range(num_rank)
    elif format == "3d":
        processed_data.append(data)
    elif format == "2d":
        """
        There are two approach, the first one is that concatenate in 
        """
    elif format == "1d":
        processed_data.append(data.flatten())
    return processed_data

def getSize(self,module,data):
    mem_usage = 0
    if module == "compressed":
        for i in data:
            mem_usage += len(i)
    elif module == "uncompressed":
        for i in data:
            mem_usage += i.size * i.itemsize
    return mem_usage

class zfp_compressor():
    """
    zfp_compressor use zfp as its compression algorithm
    compress needs tolerance(error_bound)
    decompress needs nothing
    """
    def compress(self, data, layout, error_bound):
        # error bound should be always a reletive value
        data = reformat(format, data)
        for s in data:
                res.append(zfpy.compress_numpy(s,tolerance))
            ratio = self.getSize("uncompressed",data=data) / self.getSize("compressed",data=res)
        return None
    def decompress(self, data, layout, data_shape, data_type):
        return None

class sz_compressor():
    """
    sz_compressor use SZ3 as its compression algorithm
    """
    def compress(self, data, layout, error_bound):
        data = reformat(format, data)
        return None
    def decompress(self, data, layout, data_shape, data_type):
        return None

class sparse_compressor():
    """
    sparse_compressor is a self-implement compression algorithm, it has a trival implementation,
    it apply a threshold, usually a reletive value of all data range, no matter negetive or possitive value,
    once their absolute value is smaller than that threshold, this element will be considered as zero. 
    Then the whole dense numpy array will be transformed to a sparse matrix.
    """
    def compress(self, data, layout, error_bound):
        data = reformat(format, data)
        res = []
        for s in data:
            a_error_bound = error_bound * (s.max()-s.min())
            with np.nditer(s, op_flags=['readwrite']) as it:
                for x in it:
                    if np.absolute(x) < a_error_bound:
                        x[...] = 0
            # CSR or CSR may depend on the largest dimension
            res.append(sparse.csr_matrix(s))
        return res
    def decompress(self, data, layout, data_shape, data_type):
        res = []
        for s in data:
            res.append(sparse.csr_matrix.todense(s).dtype(data_type))

class quantization_compressor():
    """
    quantization_compressor is the base line compression algorithm.
    """
    def compress(self, data, layout, bits):
        data = reformat(format,data)
        for s in data:
            s_compressed = torch.quantize_per_tensor(s, bits, 0)

    def decompress(self, data, layout, bits):
        res = []
        for s in data:
            s_decompressed = s.dequantize()


class emb_compressor():
    def __init__(self):
        self.compressor = None
        self.ratioLog = {}
        lib_extention = "so" if platform.system() == 'Linux' else "dylib"
        sz = SZ("/N/u/haofeng/BigRed200/SZ3_build/lib64/libSZ3c.{}".format(lib_extention))
        return 


    def compress(self, compressor, format,data, tolerance = -1, rate = -1, precision = -1):
        # we can apply different compressor: sz copressor, zfp compressor, and so on ...
        # different 4 kinds of format: sample-wise-seperate, table-wise-seperate, sample-wise-one, table-wise-one
        # the data transform from a 3D tensor to a 3D numpy array
        # this function will return a list, which contains the result
        
        # We just compress 1D numpy array, so after split, we will flatten the 2D array.
        # some compressor may support compress 2D array. However, 
        data = self.reformat(format,data)
        res = []
        ratio = 0.0
        if compressor == "ZFP_compressor":
            for s in data:
                res.append(zfpy.compress_numpy(s,tolerance))
            ratio = self.getSize("uncompressed",data=data) / self.getSize("compressed",data=res)
        elif compressor == "SZ_compressor":
            lib_extention = "so" if platform.system() == 'Linux' else "dylib"
            sz = SZ("/N/u/haofeng/BigRed200/SZ3_build/lib64/libSZ3c.{}".format(lib_extention))
            for s in data:
                data_cmpr, data_ratio = sz.compress(s, 0, tolerance, 0, 0)
                res.append(data_cmpr)
                ratio += data_ratio
            ratio /= len(data)
        return res, ratio

    def decompress(self, compressor, format, data, data_shape, data_type):
        # Input: An array, size is 1 or #table
        # Output: An array, size is #table
        res = []
        if compressor == "ZFP_compressor":
            if format == "table_wise_seperate":
                for i,s in enumerate(data):
                    data[i]= zfpy.decompress_numpy(s)
                    res.append(data[i])
            elif format == "table_wise_one":
                res.append(zfpy.decompress_numpy(data[0]))
            elif format == "sample_wise_one":
                res.append(zfpy.decompress_numpy(data[0]).transpose([1,0,2]))

        elif compressor == "SZ_compressor":
            lib_extention = "so" if platform.system() == 'Linux' else "dylib"
            sz = SZ("/N/u/haofeng/BigRed200/SZ3_build/lib64/libSZ3c.{}".format(lib_extention))
            if format == "table_wise_one":
                res.append(sz.decompress(data[0], data_shape, data_type))
            elif format == "table_wise_seperate":
                for i,s in enumerate(data):
                    data[i]= sz.decompress(s, data_shape, data_type)
                    res.append(data[i])
            elif format == "sample_wise_one":
                res.append(sz.decompress(data[0].transpose([1,0,2]), data_shape, data_type))

        return res

    def recordRatio(self, name, ratio):
        if name in self.ratioLog.keys():
            self.ratioLog[name].append(ratio)
        else:
            self.ratioLog[name] = [ratio]

 
