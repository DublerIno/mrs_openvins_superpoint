#!/usr/bin/env python3

import argparse
import contextlib
import io
import struct
import sys

import numpy as np

from superpoint_pytorch import SuperPointFrontend


def read_exact(stream, nbytes):
  buf = bytearray()
  while len(buf) < nbytes:
    chunk = stream.read(nbytes - len(buf))
    if not chunk:
      return None
    buf.extend(chunk)
  return bytes(buf)


def write_all(stream, data):
  stream.write(data)
  stream.flush()

def apply_anms(points_conf, desc, num_features):
  # points_conf: Nx3 [x, y, response], desc: NxD
  if num_features <= 0 or points_conf.shape[0] <= num_features:
    return points_conf, desc

  # Sort by response descending.
  order = np.argsort(-points_conf[:, 2], kind="mergesort")
  points_conf = points_conf[order]
  desc = desc[order]

  n = points_conf.shape[0]
  radii = np.full(n, np.finfo(np.float32).max, dtype=np.float32)

  # Compute suppression radii (O(N^2)), same style as requested implementation.
  for i in range(n):
    xi = points_conf[i, 0]
    yi = points_conf[i, 1]
    for j in range(i):
      dx = xi - points_conf[j, 0]
      dy = yi - points_conf[j, 1]
      dist = dx * dx + dy * dy
      if dist < radii[i]:
        radii[i] = dist
      if dist < radii[j]:
        radii[j] = dist

  idx = np.argsort(-radii, kind="mergesort")
  idx = idx[:num_features]
  return points_conf[idx], desc[idx]


def main() -> int:
  parser = argparse.ArgumentParser(description="Persistent SuperPoint worker for OpenVINS.")
  parser.add_argument("--weights", required=True, help="Path to SuperPoint PyTorch weights file.")
  parser.add_argument("--conf_thresh", type=float, required=True, help="Confidence threshold.")
  parser.add_argument("--nms_dist", type=int, required=True, help="NMS distance in pixels.")
  parser.add_argument("--cuda", type=int, default=0, help="Use CUDA if 1, otherwise CPU.")
  parser.add_argument("--num_features", type=int, default=-1, help="ANMS target count. <=0 disables ANMS.")
  args = parser.parse_args()

  frontend = SuperPointFrontend(
      weights_path=args.weights,
      nms_dist=args.nms_dist,
      conf_thresh=args.conf_thresh,
      nn_thresh=0.7,
      cuda=bool(args.cuda),
  )

  stdin = sys.stdin.buffer
  stdout = sys.stdout.buffer

  while True:
    req_header = read_exact(stdin, 8)
    if req_header is None:
      return 0
    width, height = struct.unpack("<II", req_header)
    if width == 0 or height == 0:
      write_all(stdout, struct.pack("<III", 1, 0, 0))
      continue

    image_bytes = int(width) * int(height)
    payload = read_exact(stdin, image_bytes)
    if payload is None:
      return 0

    try:
      gray = np.frombuffer(payload, dtype=np.uint8).reshape((height, width))
      img = gray.astype(np.float32) / 255.0

      with contextlib.redirect_stdout(io.StringIO()):
        pts, desc, _ = frontend.run(img)

      if pts is None or desc is None or pts.shape[1] == 0 or desc.shape[1] == 0:
        points_out = np.zeros((0, 2), dtype=np.float32)
        desc_out = np.zeros((0, 256), dtype=np.float32)
      else:
        n = min(pts.shape[1], desc.shape[1])
        points_conf = np.empty((n, 3), dtype=np.float32)
        points_conf[:, 0] = pts[0, :n]
        points_conf[:, 1] = pts[1, :n]
        points_conf[:, 2] = pts[2, :n]
        desc_n = desc[:, :n].T.astype(np.float32, copy=False)

        points_conf, desc_n = apply_anms(points_conf, desc_n, args.num_features)
        points_out = points_conf[:, :2].astype(np.float32, copy=False)
        desc_out = desc_n.astype(np.float32, copy=False)

      write_all(stdout, struct.pack("<III", 0, int(points_out.shape[0]), int(desc_out.shape[1] if desc_out.ndim == 2 else 0)))
      if points_out.size > 0:
        write_all(stdout, points_out.tobytes(order="C"))
      if desc_out.size > 0:
        write_all(stdout, desc_out.tobytes(order="C"))
    except Exception:
      write_all(stdout, struct.pack("<III", 1, 0, 0))


if __name__ == "__main__":
  raise SystemExit(main())
