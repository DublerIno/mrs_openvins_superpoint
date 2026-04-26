#!/usr/bin/env python3

import argparse
import contextlib
import io
import struct
import sys

import numpy as np
import torch

from superpoint_pytorch import SuperPointFrontend
from models.superglue import SuperGlue


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

  order = np.argsort(-points_conf[:, 2], kind="mergesort")
  points_conf = points_conf[order]
  desc = desc[order]

  n = points_conf.shape[0]
  radii = np.full(n, np.finfo(np.float32).max, dtype=np.float32)
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


def run_superpoint(frontend, gray, num_features):
  img = gray.astype(np.float32) / 255.0
  with contextlib.redirect_stdout(io.StringIO()):
    pts, desc, _ = frontend.run(img)

  if pts is None or desc is None or pts.shape[1] == 0 or desc.shape[1] == 0:
    points_out = np.zeros((0, 2), dtype=np.float32)
    desc_out = np.zeros((0, 256), dtype=np.float32)
    scores_out = np.zeros((0,), dtype=np.float32)
    return points_out, desc_out, scores_out

  n = min(pts.shape[1], desc.shape[1])
  points_conf = np.empty((n, 3), dtype=np.float32)
  points_conf[:, 0] = pts[0, :n]
  points_conf[:, 1] = pts[1, :n]
  points_conf[:, 2] = pts[2, :n]
  desc_n = desc[:, :n].T.astype(np.float32, copy=False)

  points_conf, desc_n = apply_anms(points_conf, desc_n, num_features)
  points_out = points_conf[:, :2].astype(np.float32, copy=False)
  scores_out = points_conf[:, 2].astype(np.float32, copy=False)
  desc_out = desc_n.astype(np.float32, copy=False)
  return points_out, desc_out, scores_out


def run_superglue(superglue, device, req, stdin, stdout):
  w0, h0, w1, h1, n0, n1, d = req
  if w0 == 0 or h0 == 0 or w1 == 0 or h1 == 0 or d == 0:
    write_all(stdout, struct.pack("<II", 1, 0))
    return

  img0_bytes = int(w0) * int(h0)
  img1_bytes = int(w1) * int(h1)

  payload_img0 = read_exact(stdin, img0_bytes)
  payload_img1 = read_exact(stdin, img1_bytes)
  if payload_img0 is None or payload_img1 is None:
    raise EOFError

  payload_kpts0 = read_exact(stdin, int(n0) * 2 * 4)
  payload_kpts1 = read_exact(stdin, int(n1) * 2 * 4)
  payload_scores0 = read_exact(stdin, int(n0) * 4)
  payload_scores1 = read_exact(stdin, int(n1) * 4)
  payload_desc0 = read_exact(stdin, int(n0) * int(d) * 4)
  payload_desc1 = read_exact(stdin, int(n1) * int(d) * 4)
  if (payload_kpts0 is None or payload_kpts1 is None or payload_scores0 is None or payload_scores1 is None or
      payload_desc0 is None or payload_desc1 is None):
    raise EOFError

  try:
    gray0 = np.frombuffer(payload_img0, dtype=np.uint8).reshape((h0, w0)).astype(np.float32) / 255.0
    gray1 = np.frombuffer(payload_img1, dtype=np.uint8).reshape((h1, w1)).astype(np.float32) / 255.0

    kpts0 = np.frombuffer(payload_kpts0, dtype=np.float32).reshape((n0, 2))
    kpts1 = np.frombuffer(payload_kpts1, dtype=np.float32).reshape((n1, 2))
    scores0 = np.frombuffer(payload_scores0, dtype=np.float32).reshape((n0,))
    scores1 = np.frombuffer(payload_scores1, dtype=np.float32).reshape((n1,))
    desc0 = np.frombuffer(payload_desc0, dtype=np.float32).reshape((n0, d))
    desc1 = np.frombuffer(payload_desc1, dtype=np.float32).reshape((n1, d))

    data = {
      'image0': torch.from_numpy(gray0).unsqueeze(0).unsqueeze(0).to(device),
      'image1': torch.from_numpy(gray1).unsqueeze(0).unsqueeze(0).to(device),
      'keypoints0': torch.from_numpy(kpts0).unsqueeze(0).to(device),
      'keypoints1': torch.from_numpy(kpts1).unsqueeze(0).to(device),
      'scores0': torch.from_numpy(scores0).unsqueeze(0).to(device),
      'scores1': torch.from_numpy(scores1).unsqueeze(0).to(device),
      'descriptors0': torch.from_numpy(desc0.T).unsqueeze(0).to(device),
      'descriptors1': torch.from_numpy(desc1.T).unsqueeze(0).to(device),
    }

    with torch.no_grad():
      pred = superglue(data)

    matches0 = pred['matches0'][0].detach().cpu().numpy().astype(np.int32, copy=False)
    mscore0 = pred['matching_scores0'][0].detach().cpu().numpy().astype(np.float32, copy=False)

    write_all(stdout, struct.pack("<II", 0, int(matches0.shape[0])))
    if matches0.size > 0:
      write_all(stdout, matches0.tobytes(order="C"))
      write_all(stdout, mscore0.tobytes(order="C"))
  except Exception:
    write_all(stdout, struct.pack("<II", 1, 0))


def main() -> int:
  parser = argparse.ArgumentParser(description="Persistent SuperPoint/SuperGlue worker for OpenVINS.")
  parser.add_argument("--weights", required=True, help="Path to SuperPoint PyTorch weights file.")
  parser.add_argument("--conf_thresh", type=float, required=True, help="Confidence threshold.")
  parser.add_argument("--nms_dist", type=int, required=True, help="NMS distance in pixels.")
  parser.add_argument("--cuda", type=int, default=0, help="Use CUDA if 1, otherwise CPU.")
  parser.add_argument("--num_features", type=int, default=-1, help="ANMS target count. <=0 disables ANMS.")
  parser.add_argument("--superglue", choices=['indoor', 'outdoor'], default='outdoor', help="SuperGlue weights.")
  parser.add_argument("--sinkhorn_iterations", type=int, default=20, help="SuperGlue Sinkhorn iterations.")
  parser.add_argument("--match_threshold", type=float, default=0.2, help="SuperGlue match threshold.")
  args = parser.parse_args()

  use_cuda = bool(args.cuda) and torch.cuda.is_available()
  device = 'cuda' if use_cuda else 'cpu'

  frontend = SuperPointFrontend(
      weights_path=args.weights,
      nms_dist=args.nms_dist,
      conf_thresh=args.conf_thresh,
      nn_thresh=0.7,
      cuda=use_cuda,
  )

  with contextlib.redirect_stdout(io.StringIO()):
    superglue = SuperGlue({
        'weights': args.superglue,
        'sinkhorn_iterations': args.sinkhorn_iterations,
        'match_threshold': args.match_threshold,
    }).eval().to(device)

  stdin = sys.stdin.buffer
  stdout = sys.stdout.buffer

  while True:
    req_cmd_raw = read_exact(stdin, 4)
    if req_cmd_raw is None:
      return 0
    cmd = struct.unpack("<I", req_cmd_raw)[0]

    if cmd == 0:
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
        points_out, desc_out, scores_out = run_superpoint(frontend, gray, args.num_features)

        write_all(stdout, struct.pack("<III", 0, int(points_out.shape[0]), int(desc_out.shape[1] if desc_out.ndim == 2 else 0)))
        if points_out.size > 0:
          write_all(stdout, points_out.tobytes(order="C"))
        if desc_out.size > 0:
          write_all(stdout, desc_out.tobytes(order="C"))
        if scores_out.size > 0:
          write_all(stdout, scores_out.tobytes(order="C"))
      except Exception:
        write_all(stdout, struct.pack("<III", 1, 0, 0))
    elif cmd == 1:
      req = read_exact(stdin, 28)
      if req is None:
        return 0
      run_superglue(superglue, device, struct.unpack("<IIIIIII", req), stdin, stdout)
    else:
      # Unknown command, respond with generic failure for forward compatibility.
      write_all(stdout, struct.pack("<III", 1, 0, 0))


if __name__ == "__main__":
  raise SystemExit(main())
