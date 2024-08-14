import onnxruntime as ort

import os,sys
import numpy as np



if __name__ == "__main__":

    args = sys.argv

    ## Infer ONNX 
    # sess = onnxruntime.InferenceSession(args[1], 
    #                                 providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    mmdep_path = "/home/k-morioka/anaconda3/envs/internimage/lib/python3.7/site-packages/mmdeploy/lib/libmmdeploy_onnxruntime_ops.so"
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(mmdep_path)
    sess = ort.InferenceSession(args[1], providers=['CPUExecutionProvider'], sess_options=session_options)

    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    input_type = sess.get_inputs()[0].type
    print("Input name  :", input_name)
    print("Input shape :", input_shape)
    print("Input type  :", input_type)
