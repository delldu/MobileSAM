# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 16 Aug 2023 04:17:55 PM CST
# ***
# ************************************************************************************/
#
# from .debug import debug_var
import torch
import numpy as np

def __output_kv__(prefix, k, v):
	if isinstance(v, torch.Tensor):
		print(f"{prefix}tensor [{k}] size:", list(v.size()), ", min:", v.min(), ", max:", v.max())
	elif isinstance(v, np.ndarray):
		print(f"{prefix}array [{k}] shape:", v.shape, ", min:", v.min(), ", max:", v.max())
	elif isinstance(v, list):
		print(f"{prefix}list [{k}] len:", len(v), ",", v)
	elif isinstance(v, tuple):
		print(f"{prefix}tuple [{k}] len:", len(v), ",", v)
	elif isinstance(v, str):
		print(f"{prefix}[{k}] value:", "'" + v + "'")
	else:
		print(f"{prefix}[{k}] value:", v)

def debug_var(v_name, v_value):
	if isinstance(v_value, dict):
		prefix = "    "
		print(f"{v_name} is dict:")
		for k, v in v_value.items():
			__output_kv__(prefix, k, v)
	else:
		prefix = ""
		__output_kv__(prefix, v_name, v_value)
