#!/usr/bin/env python3

import argparse
import contextlib
import io

import cv2
import numpy as np

from superpoint_pytorch import SuperPointFrontend


def main() -> int:
  parser = argparse.ArgumentParser(description="Run SuperPoint (PyTorch) and export features for C++ frontend.")
  parser.add_argument("--weights", required=True, help="Path to SuperPoint PyTorch weights file.")
  parser.add_argument("--image", required=True, help="Path to grayscale image.")
  parser.add_argument("--output", required=True, help="Path to output yaml file.")
  parser.add_argument("--conf_thresh", type=float, required=True, help="Confidence threshold.")
  parser.add_argument("--nms_dist", type=int, required=True, help="NMS distance in pixels.")
  parser.add_argument("--cuda", type=int, default=0, help="Use CUDA if 1, otherwise CPU.")
  args = parser.parse_args()

  gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
  if gray is None:
    return 2

  img = gray.astype(np.float32) / 255.0
  frontend = SuperPointFrontend(
      weights_path=args.weights,
      nms_dist=args.nms_dist,
      conf_thresh=args.conf_thresh,
      nn_thresh=0.7,
      cuda=bool(args.cuda),
  )

  # The original script prints debug point counts; suppress to keep C++ logs clean.
  with contextlib.redirect_stdout(io.StringIO()):
    pts, desc, _ = frontend.run(img)

  if pts is None or pts.shape[1] == 0 or desc is None or desc.shape[1] == 0:
    points_out = np.zeros((0, 1, 2), dtype=np.float32)
    descriptors_out = np.zeros((0, 256), dtype=np.float32)
  else:
    n = min(pts.shape[1], desc.shape[1])
    points_out = np.zeros((n, 1, 2), dtype=np.float32)
    points_out[:, 0, 0] = pts[0, :n]
    points_out[:, 0, 1] = pts[1, :n]
    descriptors_out = desc[:, :n].T.astype(np.float32)

  fs = cv2.FileStorage(args.output, cv2.FILE_STORAGE_WRITE)
  fs.write("points", points_out)
  fs.write("descriptors", descriptors_out)
  fs.release()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
