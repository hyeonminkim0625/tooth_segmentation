import numpy as np
import argparse
import csv
import math
import torchvision
import torch

# box consist of x0, y0, x1, y1
def iou(pred_box, gt_box):
   # if the box consists of x, y, w, h, uncomment belows
   #pred_box[:, 2] = pred_box[:, 0] + pred_box[:, 2]
   #pred_box[:, 3] = pred_box[:, 1] + pred_box[:, 3]
   #gt_box[:, 2] = gt_box[:, 0] + gt_box[:, 2]
   #gt_box[:, 3] = gt_box[:, 1] + gt_box[:, 3]
   pred_box = torch.tensor([pred_box])
   gt_box = torch.tensor([gt_box])

   return torchvision.ops.box_iou(pred_box, gt_box)

# calculate mAP for each prediction
def metrics_each_sample(pred, gt, iou_threshold=0.8):
   pred_boxes = np.array(pred['boxes'])
   pred_labels = np.array(pred['labels'])
   pred_confs = np.array(pred['scores'])

   sort_idx = np.argsort(pred_confs)
   sort_idx = sort_idx[::-1]

   pred_boxes = pred_boxes[sort_idx].tolist()
   pred_labels = pred_labels[sort_idx].tolist()
   pred_confs = pred_confs[sort_idx].tolist()

   gt_boxes = gt['boxes']
   gt_labels = gt['labels']

   tps = np.zeros(len(pred_labels), dtype=np.int8)
   fps = np.zeros(len(pred_labels), dtype=np.int8)
   fns = np.ones(len(gt_labels), dtype=np.int8)

   for i, pred_box in enumerate(pred_boxes):
      iou_max = -1
      idx_max = -1
      for j, gt_box in enumerate(gt_boxes):
         if fns[j] == 0:
            continue

         iou_j = iou(pred_box, gt_box)
         if iou_j > iou_max:
            iou_max = iou_j
            idx_max = j

      if iou_max < iou_threshold:
         fps[i] = 1
      else:
         if pred_labels[i] == gt_labels[idx_max]:
            tps[i] = 1
            fns[idx_max] = 0
         else:
            fps[i] = 1

   acc_tp = np.cumsum(tps)
   acc_fp = np.cumsum(fps)

   recall = np.divide(acc_tp, len(gt_labels))
   precision = np.divide(acc_tp, (acc_tp + acc_fp))

   class_metrics = {}
   for i, c in enumerate(pred_labels):
      if c in class_metrics:
         metrics = class_metrics[c]

         class_metrics[c] = [metrics[0] + [tps[i]], 
                             metrics[1] + [fps[i]],
                             metrics[2] + [pred_confs[i]]]
      else:
         class_metrics[c] = [[tps[i]], [fps[i]], [pred_confs[i]]]

   return precision, recall, class_metrics

def get_n_gts_each_class(gts):
   n_gts_each_class = {}
   for gt in gts:
      labels = gt['labels']
      for label in labels:
         if label in n_gts_each_class:
            n_gts_each_class[label] += 1
         else:
            n_gts_each_class[label] = 1
   return n_gts_each_class

# Calcumate mAP
## @ preds: predictions
##          = [ pred_1, pred_2, ... , pred_k ]
##          where pred_i = { 'boxes': [N_i, 4], 'labels': [N_i], 'scores': [N_i] }
## @ gts: ground truths
##        = [ gt_1, gt_2, ... , gt_k ]
##        where gt_i = {'boxes': [M_i, 4], 'labels': [M_i]}
def map(preds, gts, iou_threshold=0.8):
   if len(preds) != len(gts):
      raise Exception("Sample counts does not match", len(preds), len(gts))

   len_samples = len(preds)

   class_metrics = {}
   for i in range(len_samples):
      if preds[i]['id'] == gts[i]['id']:
         prec_i, rec_i, class_metrics_i = metrics_each_sample(preds[i], gts[i], iou_threshold)
      else:
         image_id = preds[i]['id']
         found = False
         for j in range(len(gts)):
            if image_id == gts[j]['id']:
               prec_i, rec_i, class_metrics_i = metrics_each_sample(preds[i], gts[j], iou_threshold)
               found = True
               break
         if not found:
            return -2

      for k, v in class_metrics_i.items():
         if k in class_metrics:
            class_metrics[k] = [class_metrics[k][0] + v[0], class_metrics[k][1] + v[1], class_metrics[k][2] + v[2]]
         else:
            class_metrics[k] = v

   n_gts_each_class = get_n_gts_each_class(gts)

   # calculate ap for each class
   epsilon = 1e-9
   aps = []
   for k, v in class_metrics.items():
      if not k in n_gts_each_class:
         continue

      tps = np.array(v[0])
      fps = np.array(v[1])
      scores = v[2]

      sort_idx = np.argsort(scores)
      sort_idx[::-1]

      tps = tps[sort_idx]
      fps = fps[sort_idx]

      tp_cumsum = np.cumsum(tps)
      fp_cumsum = np.cumsum(fps)

      recalls = tp_cumsum / n_gts_each_class[k]
      precisions = tp_cumsum / (tp_cumsum + fp_cumsum + epsilon)

      precisions = np.concatenate(([1], precisions))
      recalls = np.concatenate(([0], recalls))

      aps.append(np.trapz(precisions, recalls))

   if len(aps) == 0:
      return 0

   return (sum(aps) / len(aps))

# evaluation.py
def read_prediction(prediction_file):
   # NEED TO IMPLEMENT #1
   # function that loads prediction
   # pred_array = np.loadtxt(prediction_file, dtype=np.int16)
   csvfile = open(prediction_file, "r")
   csvread = csv.reader(csvfile, delimiter=',')
   pred_array = list(csvread)
   csvfile.close()
   
   return pred_array

def read_ground_truth(ground_truth_file):
   # NEED TO IMPLEMENT #2
   # function that loads test_data
   # gt_array = np.loadtxt(ground_truth_file, dtype=np.int16)
   csvfile = open(ground_truth_file, "r")
   csvread = csv.reader(csvfile, delimiter=',')
   gt_arr = list(csvread)
   csvfile.close()

   return gt_arr

def is_ascending(arr):
   if len(arr) <= 1 or len(arr[0]) <= 0 or len(arr[1]) <= 0:
      return True

   for i in range(1, len(arr)):
      if arr[i-1][0] >= arr[i][0]:
         return False
   return True
      

def bin_search(arr, key, left=0, right=None):
   if right == None:
      right = len(arr)
   while left < right:
      cur = (left + right) // 2
      if arr[cur][0] == key:
         return arr[cur], cur
      if arr[cur][0] < key:
         left = cur + 1
      else:
         right = cur
   return None, -1

def bf_search(arr, key):
   for val in arr:
      if key == val[0]:
         return val
   return None

#
## @ prediction: a row of prediction is consists of below fields
## >> image_id | 0th_class_id | 0th_scores | 0th_bbox_x0 | 0th_bbox_y0 | 0th_bbox_x1 | 0th_bbox_y1 | ,,, | kth_bbox_y1
## @ ground_truth: a row of ground truth is consists of below fields
## >> image_id | class_id | bbox_x0 | bbox_y0 | bbox_x1 | bbox_y1 | image_file_name
def evaluate(prediction, ground_truth):
   # NEET TO IMPLEMENT #3

   # Convert prediction to y_pred like below
   # [ { 'boxes': [N_i, 4], 'labels': [N_i], 'scores': [N_i] }, ... ]

   y_pred = []
   y_id = []
   for pred in prediction:
      if len(pred) <= 0:
         continue
      image_id = pred[0]
      y_id.append(image_id)
      pred = pred[1:]
      if not len(pred) % 6 == 0:
         raise Exception("Invalid prediction row")
      
      boxes = []
      scores = []
      labels = []
      for i in range(0, len(pred), 6):
         x0 = float(pred[i+2])
         y0 = float(pred[i+3])
         x1 = float(pred[i+4])
         y1 = float(pred[i+5])
         labels.append(int(pred[i]))
         scores.append(float(pred[i+1]))
         boxes.append([min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1)])

      y_pred.append({'id': image_id,'boxes': boxes, 'labels': labels, 'scores': scores})

   # Convert ground_truth to y_true like below
   # [ {'boxes': [M_i, 4], 'labels': [M_i]}, ... ]

   #ground_truth = np.array(ground_truth)
   #y_true = [{'boxes': [[min(float(x0), float(x1), 
   #                      min(float(y0), float(y1), 
   #                      max(float(x0), float(x1), 
   #                      max(float(y0), float(y1))]], 
   #           'labels':[int(c)]} for _,c,x0,y0,x1,y1 in ground_truth[:, :6]]

   y_true = []
   is_sored = is_ascending(ground_truth)

   for i in range(len(y_id)):
      key = y_id[i]

      if is_sored:
         row, _ = bin_search(ground_truth, key)
      else:
         row = bf_search(ground_truth, key)

      if row == None:
         return -3

      c = int(row[1])
      x0 = float(row[2])
      y0 = float(row[3])
      x1 = float(row[4])
      y1 = float(row[5])

      y_true.append({'id': key, 'boxes': [[min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]], 'labels':[c]})

   if not len(y_true) == len(y_pred):
      raise Exception('prediction rows does not match with ground truth rows, predictions: %d, ground truth: %d' % (len(y_pred), len(y_true)))

   metric_result = map(y_pred, y_true, 0.8)

   return metric_result

# user-defined function for evaluation metrics
def evaluation_metrics(prediction_file: str, ground_truth_file: str):
   # read prediction and ground truth from file
   prediction = read_prediction(prediction_file)  # NOTE: prediction is text
   ground_truth = read_ground_truth(ground_truth_file)

   # if not len(prediction) == len(ground_truth):
   #    return -1

   return evaluate(prediction, ground_truth)


if __name__ == '__main__':
   args = argparse.ArgumentParser()
   # --prediction is set by file's name that contains the result of inference. (nsml internally sets)
   # prediction file requires type casting because '\n' character can be contained.
   args.add_argument('--prediction', type=str, default='pred.txt')
   args.add_argument('--test_label_path', type=str)
   config = args.parse_args()
   # print the evaluation result
   # evaluation prints only int or float value.
   print(evaluation_metrics(config.prediction, config.test_label_path))
   