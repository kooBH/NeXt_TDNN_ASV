import sys, argparse, os
import torch
import importlib
from collections import OrderedDict
from SpeakerNet import SpeakerNet

import librosa as rs

import numpy as np
import torch.nn as nn
import torch.onnx
import onnx
import onnxruntime

parser = argparse.ArgumentParser(description = "Speaker Verification")
parser.add_argument('--config',         type=str,   default='configs.constant',   help='Config file')
args = parser.parse_args()

args.config = args.config.replace('/', '.')
args.config = args.config.replace('.py', '')
print(args.config)

config = importlib.import_module(args.config)

# set feature extractor
feature_extractor = importlib.import_module('preprocessing.' + config.FEATURE_EXTRACTOR).__getattribute__("feature_extractor")
feature_extractor = feature_extractor(**config.FEATURE_EXTRACTOR_CONFIG)

# set speaker embedding extractor
model = importlib.import_module('models.' + config.MODEL).__getattribute__("MainModel")
model =  model(**config.MODEL_CONFIG)

# set aggregation
aggregation = importlib.import_module('aggregation.' + config.AGGREGATION).__getattribute__("Aggregation")
aggregation = aggregation(**config.AGGREGATION_CONFIG)


speaker_net = SpeakerNet(feature_extractor = feature_extractor,
                    spec_aug = None, 
                    model = model,
                    aggregation=aggregation,
                    loss_function = None)
speaker_net.eval()

# check model param, mmac
#get_model_param_mmac(speaker_net, int(160*config.EVAL_FRAMES + 240))

# =============================================
# 😀😀 3. load checkpoint
# =============================================
"""
if config has 'TEST_CHECKPOINT', load it.
else, find min eer ckpt from test result(e.g. train/lightning_logs/version_0/checkpoints/###.ckpt) and load it.
"""
if hasattr(config, 'TEST_CHECKPOINT'):
    min_eer_epoch = config.TEST_CHECKPOINT.split('epoch=')[-1].split('-')[0] # epoch
    engine_state_dict = torch.load(config.TEST_CHECKPOINT)        
else: # find min eer ckpt
    min_eer_ckpt_path = get_min_eer_ckpt(config)
    min_eer_epoch = min_eer_ckpt_path.split('epoch=')[-1].split('-')[0] # epoch
    engine_state_dict = torch.load(min_eer_ckpt_path)

# restore model_state_dict
model_statd_dict = OrderedDict()

# remove 'model.' from keys in the engine_state_dict
for k in engine_state_dict['state_dict'].keys():
    model_statd_dict[k.replace('speaker_net.', '')] = engine_state_dict['state_dict'][k]

load_msg = speaker_net.load_state_dict(model_statd_dict, strict=False)
print(load_msg)

x,_ = rs.load('female_1.wav', sr=16000)
x = torch.tensor(x).float().unsqueeze(0)
feat = speaker_net(x)
feat = feat.detach().numpy()
feat.tofile('x.npy')
print(f"feat : {feat.shape}")




# feature_extractor is not compatible
speaker_net.feature_extractor=nn.Identity()

# =============================================
# ONNX export
# =============================================

print("ONXX Export")
#dummy_input = torch.randn(1,16000)
dummy_input = torch.randn(1,80,157).float()

path_out = f"{args.config}.onnx"  

torch.onnx.export(
    speaker_net,
    (dummy_input),
    path_out,
    verbose=False,
    opset_version=17,
    input_names=["inputs"],
    output_names=["outputs"],
    dynamic_axes={'inputs' : {2 : 'time_stamp'}}
)
print(path_out)

#### Test ####

# Check
onnx_model = onnx.load(path_out)
onnx.checker.check_model(onnx_model)

# Infer
ort_session = onnxruntime.InferenceSession(path_out,
providers=["CUDAExecutionProvider"]
)

ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
ort_outs = ort_session.run(None, ort_inputs)[0]

torch_out = speaker_net(dummy_input).detach().numpy()


print(f"{ort_outs.shape} {torch_out.shape}")

error_level_rtol = 1e-05
error_level_atol = 1e-05

print("TESTING torch vs ONNX")
try :
    np.testing.assert_allclose(torch_out, ort_outs, rtol=error_level_rtol, atol=error_level_atol,verbose=True)
except AssertionError :
    print(f"ERROR is larger than rtol {error_level_rtol} | atol {error_level_atol}")

print(f"{np.sum(np.abs(ort_outs-torch_out))} : {np.mean(np.abs(ort_outs)),np.mean(np.abs(torch_out))}")