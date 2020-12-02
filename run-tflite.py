#!/usr/bin/env python3

import tensorflow.lite as lite
import time
import numpy as np
import PIL
from coco import image_classes

ms = lambda: int(round(time.time() * 1000))

#model_path = "ssd_mobilenet_v1_coco_2018_01_28/foo.tflite"
model_path = "ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tflite"

is_quant = "quant" in model_path.lower()

def get_mobilenet_input(f, out_size=(300, 300), is_quant=True):
    img = np.array(PIL.Image.open(f).resize(out_size))
    if not(is_quant):
        img = img.astype(np.float32) / 128 - 1
    return np.array([img]) 


def print_coco_label(cl_id, t):
    print("class: {}, label: {}, time: {:,} ms".format(cl_id, image_classes[cl_id], t))


def print_output(inp_files, res):
    boxes, classes, scores, num_det=res
    
    for i, fname in enumerate(inp_files):
      n_obj = int(num_det[i])

      print("{} - found objects:".format(fname))
      for j in range(n_obj):
        cl_id = int(classes[i][j]) + 1
        label = image_classes[cl_id]
        score = scores[i][j]
        if score < 0.5:
            continue
        box = boxes[i][j]
        print("  ", cl_id, label, score, box)


ip = lite.Interpreter(model_path=model_path)
ip.allocate_tensors()
inp_id = ip.get_input_details()[0]["index"]
out_det = ip.get_output_details()
out_id0 = out_det[0]["index"]
out_id1 = out_det[1]["index"]
out_id2 = out_det[2]["index"]
out_id3 = out_det[3]["index"]

image_f = 'dog.jpg'

img = get_mobilenet_input(image_f, is_quant=is_quant)
for i in range(1,100):
  t0 = ms()
  ip.set_tensor(inp_id, img)
  ip.invoke()
  tt = ms() - t0
  print("Time:", tt, "ms")
  boxes = ip.get_tensor(out_id0)
  classes = ip.get_tensor(out_id1)
  scores = ip.get_tensor(out_id2)
  num_det = ip.get_tensor(out_id3)
  print_output([f], [boxes, classes, scores, num_det])

