# This is a class to compress embedding vector outputs
# Note this is not a real workflow, it just for meature compress ratio and accuracy after compress
from re import S
from unittest import case
import zfpy
import numpy as np
from pathlib import Path
from pysz import SZ
import platform
import torch
import ctypes
from ctypes import *
from random import random

class zfp_compressor():
    def compress(self,data,layout,error_bound):
        # error bound should be always a reletive value
        data = self.reformat(format,)
        
        return None
class emb_compressor():
    def __init__(self):
        self.compressor = None
        self.ratioLog = {}
        lib_extention = "so" if platform.system() == 'Linux' else "dylib"
        sz = SZ("/N/u/haofeng/BigRed200/SZ3_build/lib64/libSZ3c.{}".format(lib_extention))
        return 

    def reformat(self,format,data):
        processed_data = []
        if format == "table_wise_seperate":
            for s in data:
                processed_data.append(s)
        elif format == "table_wise_one":
            processed_data.append(np.stack(data))
        elif format == "sample_wise_one":
            # this should not be use for very low compression ratio
            processed_data.append(data.transpose([1,0,2]))
        elif format == "flatten":
            processed_data.append(np.stack(data).flatten())
        return processed_data

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
                a_tolerance = (s.max()-s.min()) * tolerance
                res.append(zfpy.compress_numpy(s,tolerance = a_tolerance))
            ratio = self.getSize("uncompressed",data=data) / self.getSize("compressed",data=res)
        elif compressor == "SZ_compressor":
            lib_extention = "so" if platform.system() == 'Linux' else "dylib"
            sz = SZ("/N/u/haofeng/BigRed200/SZ3_build/lib64/libSZ3c.{}".format(lib_extention))
            for s in data:
                #print("data is, " , data)
                #print("s is, ", s)
                a_tolerance = (s.max()-s.min()) * tolerance
                data_cmpr, data_ratio = sz.compress(s, 0, a_tolerance, 0, 0)
                res.append(data_cmpr)
                ratio += data_ratio
            ratio /= len(data)
        elif compressor == "Fake_compressor":
            ratio = 1.0
            for s in data:
                res.append(s)
        return res, ratio

    def decompress(self, compressor, format, data, data_shape, data_type):
        # Input: An array, size is 1 or #table
        # Output: An array, size is #table
        res = []
        if compressor == "ZFP_compressor":
            if format == "table_wise_seperate":
                for i,s in enumerate(data):
                    data[i]= zfpy.decompress_numpy(s)
                    res.append(data[i].astype(data_type))
            elif format == "table_wise_one":
                res.append(zfpy.decompress_numpy(data[0]))
            elif format == "sample_wise_one":
                res.append(zfpy.decompress_numpy(data[0]).transpose([1,0,2]))

        elif compressor == "SZ_compressor":
            lib_extention = "so" if platform.system() == 'Linux' else "dylib"
            sz = SZ("/N/u/haofeng/BigRed200/SZ3_build/lib64/libSZ3c.{}".format(lib_extention))
            if format == "table_wise_one":
                res.append(sz.decompress(data[0], data_shape, data_type).astype(data_type))
            elif format == "table_wise_seperate":
                for i,s in enumerate(data):
                    tmp = sz.decompress(s, data_shape, data_type)
                    tmp = tmp.astype(data_type)
                    res.append(tmp)
            elif format == "sample_wise_one":
                res.append(sz.decompress(data[0], data_shape, data_type).transpose([1,0,2]))
            elif format == "flatten":
                res.append(sz.decompress[data[0], data_shape, data_type).reshape()
        elif compressor == "Fake_compressor":
            res = data
        elif compressor == "Noise_generator":
            for i,s in enumerate(data):
                noise = np.random.uniform(low=0.5, high=13.3, size=(50,))
                res.append(s+noise)
        return res


    # I ignore getRatio funciton because i can check it via getSize
    def getSize(self,module,data):
        mem_usage = 0
        if module == "compressed":
            for i in data:
                mem_usage += len(i)
        elif module == "uncompressed":
            for i in data:
                mem_usage += i.size * i.itemsize
        return mem_usage

    def recordRatio(self, name, ratio):
        if name in self.ratioLog.keys():
            self.ratioLog[name].append(ratio)
        else:
            self.ratioLog[name] = [ratio]

 

def split_tensor(tensor, num_splits, dimension):
    size = tensor.size(dimension)
    split_size = size // num_splits
    split_sizes = [split_size] * num_splits

    # Adjust the last split size if the division is not exact
    remainder = size % num_splits
    if remainder > 0:
        split_sizes[-1] += remainder
    split_tensors = torch.split(tensor, split_sizes, dim=dimension)
    return split_tensors


def bytes2int(data):
    # change Byte to uint8
    return None


def concatAndOffset(listOfData):
    n = len(listOfData)
    # 3d <--> 1d, using flatten() and reshape()
    offset = np.zero(n)
    for i in range(n-1):
        offset[i+1] = offset[i] + listOfData[i].size
    concatedData = np.concatenate(listOfData)
    return concatedData, offset


# compression and decompression round trip
def pfz_round_trip():
    so_path = '/N/u/haofeng/BigRed200/fz/fz-gpu.so'
    dll = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
    func = dll.fzRoundTrip
    func.argtypes = [POINTER(c_float), POINTER(c_float), c_int, c_int, c_int, c_int, c_double]
    return func

def run_pfz(input, output, x, y, z, error_bound):
    # get input GPU pointer
    input_gpu_ptr = input.data_ptr()
    input_gpu_ptr = cast(input_gpu_ptr, ctypes.POINTER(c_float))

    # get output GPU pointer
    output_gpu_ptr = output.data_ptr()
    output_gpu_ptr = cast(output_gpu_ptr, ctypes.POINTER(c_float))

    __pfz = pfz_round_trip()
    __pfz(input_gpu_ptr, output_gpu_ptr, c_int(x * y * z * 4), c_int(x), c_int(y), c_int(z), c_double(error_bound))
