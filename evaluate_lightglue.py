import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from tqdm import tqdm
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
from lightglue import DISK, SIFT, LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

#Data structures

@dataclass
class ImagePair:
    img0_path: Path
    img1_path: Path
    K0: np.ndarray       # 3×3 intrinsic matrix, image 0
    K1: np.ndarray       # 3×3 intrinsic matrix, image 1
    T_0to1: np.ndarray   # 4×4 — transforms points from cam0 frame to cam1 frame
    name: str = ""


@dataclass
class PairResult:
    num_matches: int = 0
    precision: float = 0.0
    pose_error_deg: float = float("inf")
    R_error_deg: float = float("inf")
    t_error_deg: float = float("inf")
    num_inliers: int = 0
    match_time_ms: float = 0.0

#Geometry helpers

def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]], dtype=float)


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    R_rel = R_est @ R_gt.T
    cos_a = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def translation_error_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    #Angular error between translation directions.
    t_e = t_est.ravel() / (np.linalg.norm(t_est) + 1e-8)
    t_g = t_gt.ravel() / (np.linalg.norm(t_gt) + 1e-8)
    cos_t = np.clip(np.dot(t_e, t_g), -1.0, 1.0)
    return float(np.degrees(np.arccos(abs(cos_t))))


def epipolar_precision(pts0: np.ndarray, pts1: np.ndarray,
                       K0: np.ndarray, K1: np.ndarray,
                       R_gt: np.ndarray, t_gt: np.ndarray,
                       threshold: float = 3.0) -> float:
    #Fraction of matched pairs with Sampson distance < threshold (px).
    if len(pts0) == 0:
        return 0.0
    E = skew(t_gt) @ R_gt
    F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)
    p0h = np.c_[pts0, np.ones(len(pts0))].T
    p1h = np.c_[pts1, np.ones(len(pts1))].T
    Fp0  = F   @ p0h
    FTp1 = F.T @ p1h
    num   = np.abs((p1h * Fp0).sum(axis=0))
    denom = np.sqrt(Fp0[0]**2 + Fp0[1]**2 + FTp1[0]**2 + FTp1[1]**2 + 1e-8)
    return float((num / denom < threshold).mean())


def estimate_pose(pts0: np.ndarray, pts1: np.ndarray,
                  K: np.ndarray, threshold: float = 1.0
                  ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    if len(pts0) < 5:
        return None

    pts0 = np.ascontiguousarray(pts0, dtype=np.float32)
    pts1 = np.ascontiguousarray(pts1, dtype=np.float32)
    K = np.asarray(K, dtype=np.float64)

    def recover_best(E: np.ndarray, mask: np.ndarray,
                     p0: np.ndarray, p1: np.ndarray,
                     camera_matrix: Optional[np.ndarray] = None,
                     focal: float = 1.0,
                     pp: Tuple[float, float] = (0.0, 0.0)
                     ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        best = None
        E_candidates = [E[i:i + 3] for i in range(0, E.shape[0], 3)]
        for Ei in E_candidates:
            if Ei.shape != (3, 3):
                continue
            try:
                if camera_matrix is not None:
                    n_inliers, R, t, _ = cv2.recoverPose(
                        Ei, p0, p1, camera_matrix, mask=mask)
                else:
                    n_inliers, R, t, _ = cv2.recoverPose(
                        Ei, p0, p1, focal=focal, pp=pp, mask=mask)
            except cv2.error:
                continue
            if best is None or n_inliers > best[2]:
                best = (R, t.ravel(), int(n_inliers))
        if best is not None and best[2] > 0:
            return best
        return None

    E, mask = cv2.findEssentialMat(
        pts0, pts1, K, method=cv2.USAC_MAGSAC, prob=0.999,
        threshold=threshold)
    if E is not None and mask is not None:
        pose = recover_best(E, mask, pts0, pts1, camera_matrix=K)
        if pose is not None:
            return pose

    # Keep USAC_MAGSAC, but avoid OpenCV-version brittleness in the
    # camera-matrix overload by running it on normalized image coordinates.
    norm0 = cv2.undistortPoints(pts0.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    norm1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    mean_focal = float((K[0, 0] + K[1, 1]) / 2.0)
    norm_thresh = threshold / max(mean_focal, 1e-8)
    E, mask = cv2.findEssentialMat(
        norm0, norm1, focal=1.0, pp=(0.0, 0.0),
        method=cv2.USAC_MAGSAC, prob=0.999, threshold=norm_thresh)
    if E is None or mask is None:
        return None
    return recover_best(E, mask, norm0, norm1, focal=1.0, pp=(0.0, 0.0))


def pose_auc(errors_all: List[float],
             thresholds: Tuple[int, ...] = (5, 10, 20)) -> Dict[str, float]:

    #AUC@t = (1/t) * ∫₀ᵗ F(θ) dθ

    errors = np.array(errors_all, dtype=float)
    n_total = len(errors)
    if n_total == 0:
        return {f"AUC@{t}deg": 0.0 for t in thresholds}

    sort_idx = np.argsort(errors)
    errors_sorted = errors[sort_idx]

    recall = (np.arange(n_total) + 1) / n_total
    errors_ext = np.r_[0.0, errors_sorted]
    recall_ext = np.r_[0.0, recall]

    aucs: Dict[str, float] = {}
    for t in thresholds:
        last = int(np.searchsorted(errors_sorted, t, side="right"))
        r_t = recall_ext[last]
        e_seg = np.r_[errors_ext[:last + 1], t]
        r_seg = np.r_[recall_ext[:last + 1], r_t]
        auc_val = float(_trapz(r_seg, x=e_seg) / t * 100.0)
        aucs[f"AUC@{t}deg"] = auc_val

    return aucs


def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = qx**2 + qy**2 + qz**2 + qw**2
    s = 2.0 / (n + 1e-10)
    return np.array([
        [1 - s*(qy**2+qz**2),  s*(qx*qy - qz*qw),  s*(qx*qz + qy*qw)],
        [s*(qx*qy + qz*qw),    1 - s*(qx**2+qz**2), s*(qy*qz - qx*qw)],
        [s*(qx*qz - qy*qw),    s*(qy*qz + qx*qw),   1 - s*(qx**2+qy**2)],
    ])

#KITTI Odometry loader

class KITTILoader:
    def __init__(self, root: Path, sequences: List[str],
                 step: int = 1,       
                 max_pairs: int = 500,
                 camera: str = "image_0"):
        self.root = Path(root)
        self.sequences = sequences
        self.step = step
        self.max_pairs = max_pairs
        self.camera = camera

    def _K(self, calib_path: Path) -> np.ndarray:
        #Extract 3×3 intrinsic K from calib.txt
        proj_key = "P2" if self.camera == "image_2" else "P0"
        with open(calib_path) as f:
            for line in f:
                parts = line.split()
                if parts[0].rstrip(":") == proj_key:
                    P = np.array(parts[1:], dtype=float).reshape(3, 4)
                    # P0 = K[I|0] -> P[:, :3] = K
                    return P[:, :3]
        raise ValueError(f"{proj_key} not found in {calib_path}")

    def _poses(self, seq: str) -> List[np.ndarray]:
        #Load list of 4×4 world-to-camera matrices.
        path = self.root / "poses" / f"{seq}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Poses not found: {path}\n"
                "Download data_odometry_poses.zip from the KITTI website "
                "and extract into the dataset root.")
        out = []
        with open(path) as f:
            for line in f:
                T34 = np.array(line.split(), dtype=float).reshape(3, 4)
                out.append(np.vstack([T34, [0, 0, 0, 1]]))
        return out

    def load_pairs(self) -> List[ImagePair]:
        pairs: List[ImagePair] = []
        for seq in self.sequences:
            seq_dir = self.root / "sequences" / seq
            img_dir = seq_dir / self.camera
            if not img_dir.exists():
                print(f"[KITTI] seq {seq}: {img_dir} not found, skipping")
                continue
            try:
                K = self._K(seq_dir / "calib.txt")
                gt = self._poses(seq)
            except FileNotFoundError as e:
                print(f"[KITTI] seq {seq}: {e}")
                continue

            imgs = sorted(
                p for p in img_dir.glob("*.png")
                if p.stem.isdigit() and int(p.stem) < len(gt)
            )
            n = 0
            for i in range(0, len(imgs) - self.step, self.step):
                if n >= self.max_pairs:
                    break
                j = i + self.step
                frame_i = int(imgs[i].stem)
                frame_j = int(imgs[j].stem)
                # KITTI poses are indexed by the original frame number, not by
                # the position inside a sliced image folder.
                T_0to1 = np.linalg.inv(gt[frame_j]) @ gt[frame_i]
                pairs.append(ImagePair(
                    img0_path=imgs[i], img1_path=imgs[j],
                    K0=K, K1=K, T_0to1=T_0to1,
                    name=f"kitti/{seq}/{frame_i:06d}-{frame_j:06d}",
                ))
                n += 1

        print(f"[KITTI] {len(pairs)} pairs from seqs {self.sequences}")
        return pairs

#4Seasons loader

class FourSeasonsLoader:
    def __init__(self, root: Path, poses_root: Path, recordings: List[str],
                 step: int = 1,        
                 max_pairs: int = 300):
        self.root = Path(root)
        self.poses_root = Path(poses_root)
        self.recordings = recordings
        self.step = step
        self.max_pairs = max_pairs

    def _load_K(self, undistorted_folder: Path) -> np.ndarray:
        calib_path = undistorted_folder / "calibration" / "undistorted_calib_0.txt"
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration not found: {calib_path}")
        with open(calib_path) as f:
            parts = f.readline().split()
        fx, fy, cx, cy = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

    def _poses_dir(self, rec: str, undist_folder: Path) -> Path:
        candidates = [
            self.poses_root / f"{rec}_reference_poses" / rec,
            undist_folder / f"{rec}_reference_poses" / rec,
        ]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    def _load_T_cam_imu(self, poses_dir: Path) -> np.ndarray:
        tf_path = poses_dir / "Transformations.txt"
        if not tf_path.exists():
            raise FileNotFoundError(f"Transformations.txt not found: {tf_path}")
        with open(tf_path) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "TS_cam_imu" in line:
                vals = [float(v) for v in lines[i + 1].strip().split(",")]
                tx, ty, tz = vals[0], vals[1], vals[2]
                qx, qy, qz, qw = vals[3], vals[4], vals[5], vals[6]
                T = np.eye(4)
                T[:3, :3] = quat_to_R(qx, qy, qz, qw)
                T[:3, 3] = [tx, ty, tz]
                return T
        raise ValueError(f"TS_cam_imu block not found in {tf_path}")

    def _load_gnss_poses(self, poses_dir: Path) -> Dict[int, np.ndarray]:
        poses_path = poses_dir / "GNSSPoses.txt"
        if not poses_path.exists():
            raise FileNotFoundError(
                f"GNSSPoses.txt not found: {poses_path}\n"
                "Download the 'Reference poses' zip and extract into "
                f"{self.poses_root}")
        poses: Dict[int, np.ndarray] = {}
        with open(poses_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                vals = line.split(",")
                if len(vals) < 8:
                    continue
                ts = int(vals[0])
                tx, ty, tz = float(vals[1]), float(vals[2]), float(vals[3])
                qx, qy, qz, qw = float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7])
                T = np.eye(4)
                T[:3, :3] = quat_to_R(qx, qy, qz, qw)
                T[:3, 3] = [tx, ty, tz]
                poses[ts] = T
        return poses

    def _load_image_timestamps(self, img_dir: Path) -> List[int]:
        return sorted(int(p.stem) for p in img_dir.glob("*.png") if p.stem.isdigit())

    def _match_to_poses(self, img_timestamps: List[int],
                        pose_ts: np.ndarray) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for ts in img_timestamps:
            idx = int(np.argmin(np.abs(pose_ts - ts)))
            if abs(pose_ts[idx] - ts) / 1e6 < 100:
                mapping[ts] = int(pose_ts[idx])
        return mapping

    def load_pairs(self) -> List[ImagePair]:
        pairs: List[ImagePair] = []
        for rec in self.recordings:
            layout_candidates = [
                (
                    self.root / f"{rec}_stereo_images_undistorted",
                    self.root / f"{rec}_stereo_images_undistorted" / rec / "undistorted_images" / "cam0",
                ),
                (
                    self.root,
                    self.root / rec / "undistorted_images" / "cam0",
                ),
            ]
            undist_folder = None
            img_dir = None
            for folder, candidate_img_dir in layout_candidates:
                if candidate_img_dir.exists():
                    undist_folder = folder
                    img_dir = candidate_img_dir
                    break

            if img_dir is None or undist_folder is None:
                expected = " or ".join(str(p) for _, p in layout_candidates)
                print(f"[4Seasons] {rec}: cam0 images not found ({expected}), skipping")
                continue

            try:
                K = self._load_K(undist_folder)
            except FileNotFoundError as e:
                print(f"[4Seasons] {rec}: {e}, skipping")
                continue

            poses_dir = self._poses_dir(rec, undist_folder)

            try:
                gnss_poses = self._load_gnss_poses(poses_dir)
            except FileNotFoundError as e:
                print(f"[4Seasons] {rec}: {e}, skipping")
                continue

            if not gnss_poses:
                print(f"[4Seasons] {rec}: GNSSPoses.txt is empty, skipping")
                continue

            img_ts_list = self._load_image_timestamps(img_dir)
            pose_ts_arr = np.array(sorted(gnss_poses.keys()), dtype=np.int64)
            ts_to_pose  = self._match_to_poses(img_ts_list, pose_ts_arr)

            matched_img_ts = sorted(ts_to_pose.keys())
            print(f"[4Seasons] {rec}: {len(img_ts_list)} images, "
                  f"{len(gnss_poses)} poses, {len(matched_img_ts)} matched")

            n = 0
            for i in range(0, len(matched_img_ts) - self.step, self.step):
                if n >= self.max_pairs:
                    break
                ts0 = matched_img_ts[i]
                ts1 = matched_img_ts[i + self.step]

                img0 = img_dir / f"{ts0}.png"
                img1 = img_dir / f"{ts1}.png"
                if not img0.exists() or not img1.exists():
                    continue

                T_gnss_0 = gnss_poses[ts_to_pose[ts0]]
                T_gnss_1 = gnss_poses[ts_to_pose[ts1]]

                T_0to1 = np.linalg.inv(T_gnss_1) @ T_gnss_0

                pairs.append(ImagePair(
                    img0_path=img0, img1_path=img1,
                    K0=K, K1=K, T_0to1=T_0to1,
                    name=f"4seasons/{rec}/{ts0}-{ts1}",
                ))
                n += 1

        print(f"[4Seasons] {len(pairs)} pairs total from {len(self.recordings)} recordings")
        return pairs

def save_match_image(img0: np.ndarray, img1: np.ndarray,
                     kpts0: np.ndarray, kpts1: np.ndarray,
                     matches: np.ndarray, save_path: Path,
                     max_display: int = 200) -> None:
    # Draw side-by-side match lines and save to save_path.
    H0, W0 = img0.shape[:2]
    H1, W1 = img1.shape[:2]
    H = max(H0, H1)

    canvas = np.zeros((H, W0 + W1, 3), dtype=np.uint8)
    canvas[:H0, :W0] = img0
    canvas[:H1, W0:] = img1

    rng = np.random.default_rng(42)
    idx = np.arange(len(matches))
    if len(idx) > max_display:
        idx = rng.choice(idx, max_display, replace=False)

    colors = rng.integers(100, 255, size=(len(idx), 3)).tolist()
    for k, i in enumerate(idx):
        p0 = kpts0[matches[i, 0]]
        p1 = kpts1[matches[i, 1]]
        pt0 = (int(p0[0]), int(p0[1]))
        pt1 = (int(p1[0]) + W0, int(p1[1]))
        color = tuple(int(c) for c in colors[k])
        cv2.line(canvas, pt0, pt1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt0, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, color, -1, cv2.LINE_AA)

    label = f"Matches: {len(matches)}"
    cv2.putText(canvas, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(canvas, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (0, 0, 255), 2, cv2.LINE_AA)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), canvas)

def scale_K(K: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
#    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
#    fx *= sx, cx *= sx, fy *= sy, cy *= sy

    K_s = K.copy()
    K_s[0, 0] *= scale_x   # fx
    K_s[0, 2] *= scale_x   # cx
    K_s[1, 1] *= scale_y   # fy
    K_s[1, 2] *= scale_y   # cy
    return K_s


def get_image_hw(path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"{path}")
    return img.shape[0], img.shape[1]

# Evaluator

class LightGlueEvaluator:
    def __init__(self, feature_type: str = "superpoint",
                 device: Optional[str] = None,
                 max_keypoints: int = 2048,
                 resize: Optional[int] = 1600):  
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.feature_type = feature_type
        self.resize = resize
        self._result_cache: Dict[Tuple[str, float, float], PairResult] = {}

        print(f"Device: {self.device}  |  Features: {feature_type}  |  "
              f"Max keypoints: {max_keypoints}  |  Resize: {resize}")

        if feature_type == "superpoint":
            self.extractor = SuperPoint(max_num_keypoints=max_keypoints)
        elif feature_type == "disk":
            self.extractor = DISK(max_num_keypoints=max_keypoints)
        elif feature_type == "sift":
            self.extractor = SIFT(max_num_keypoints=max_keypoints)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        self.extractor = self.extractor.eval().to(self.device)
        self.matcher = LightGlue(features=feature_type).eval().to(self.device)

    def _load_image_and_scale_K(self, path: Path,
                                 K: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:

        if self.resize is None:
            img = load_image(path).to(self.device)
            return img, K

        H0, W0 = get_image_hw(path)
        long_edge = max(H0, W0)
        if long_edge <= self.resize:
            img = load_image(path).to(self.device)
            return img, K

        img = load_image(path, resize=self.resize).to(self.device)
        H_new, W_new = int(img.shape[-2]), int(img.shape[-1])
        scale_x = W_new / W0
        scale_y = H_new / H0
        K_scaled = scale_K(K, scale_x=scale_x, scale_y=scale_y)
        return img, K_scaled

    @torch.no_grad()
    def _match(self, pair: ImagePair
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float,
                          np.ndarray, np.ndarray]:

        img0, K0_eff = self._load_image_and_scale_K(pair.img0_path, pair.K0)
        img1, K1_eff = self._load_image_and_scale_K(pair.img1_path, pair.K1)

        f0 = self.extractor.extract(img0)
        f1 = self.extractor.extract(img1)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.matcher({"image0": f0, "image1": f1})
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        f0, f1, out = [rbd(x) for x in [f0, f1, out]]
        kpts0   = f0["keypoints"].cpu().numpy()
        kpts1   = f1["keypoints"].cpu().numpy()
        matches = out["matches"].cpu().numpy()
        return kpts0, kpts1, matches, (t1 - t0) * 1000.0, K0_eff, K1_eff

    def _read_for_viz(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read {path}")
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if self.resize is not None:
            H, W = img.shape[:2]
            if max(H, W) > self.resize:
                scale = self.resize / max(H, W)
                img = cv2.resize(img, (int(W * scale), int(H * scale)))
        return img

    def save_sample_matches(self, pairs: List[ImagePair],
                            out_dir: Path, n: int = 5) -> None:
        if n <= 0 or not pairs:
            return
        sample = random.sample(pairs, min(n, len(pairs)))
        out_dir.mkdir(parents=True, exist_ok=True)
        for pair in sample:
            try:
                kpts0, kpts1, matches, _, _, _ = self._match(pair)
                img0 = self._read_for_viz(pair.img0_path)
                img1 = self._read_for_viz(pair.img1_path)
            except Exception as e:
                print(f"  [viz skip] {pair.name}: {e}")
                continue

            safe = pair.name.replace("/", "_").replace("\\", "_").replace(":", "_")
            save_path = out_dir / f"{safe}.jpg"
            save_match_image(img0, img1, kpts0, kpts1, matches, save_path)
            print(f"  [viz] {save_path}")

    def evaluate_pair(self, pair: ImagePair,
                      epi_thresh: float = 3.0,
                      ransac_thresh: float = 1.0) -> PairResult:
        res = PairResult()
        try:
            kpts0, kpts1, matches, t_ms, K0_eff, K1_eff = self._match(pair)
        except Exception as e:
            print(f"  [skip] {pair.name}: {e}")
            return res

        res.match_time_ms = t_ms
        res.num_matches = len(matches)
        if len(matches) == 0:
            return res

        pts0 = kpts0[matches[:, 0]]
        pts1 = kpts1[matches[:, 1]]
        R_gt = pair.T_0to1[:3, :3]
        t_gt = pair.T_0to1[:3, 3]

        res.precision = epipolar_precision(
            pts0, pts1, K0_eff, K1_eff, R_gt, t_gt, epi_thresh)

        pose = estimate_pose(pts0, pts1, K0_eff, ransac_thresh)
        if pose is not None:
            R_est, t_est, n_inliers = pose
            res.num_inliers  = n_inliers
            res.R_error_deg  = rotation_error_deg(R_est, R_gt)
            res.t_error_deg  = translation_error_deg(t_est, t_gt)
            res.pose_error_deg = max(res.R_error_deg, res.t_error_deg)
        return res

    def evaluate(self, pairs: List[ImagePair], tag: str,
                 epi_thresh: float = 3.0,
                 ransac_thresh: float = 1.0) -> Dict:
        print(f"\n{'-'*55}")
        print(f"  Evaluating: {tag}  ({len(pairs)} pairs)")
        results: List[PairResult] = []
        for p in tqdm(pairs):
            cache_key = (p.name, epi_thresh, ransac_thresh)
            if cache_key not in self._result_cache:
                self._result_cache[cache_key] = self.evaluate_pair(
                    p, epi_thresh, ransac_thresh)
            results.append(self._result_cache[cache_key])

        pose_errs_all = [r.pose_error_deg for r in results]
        n_estimated = sum(1 for e in pose_errs_all if e < float("inf"))

        summary = {
            "tag": tag,
            "feature_type": self.feature_type,
            "num_pairs": len(pairs),
            "num_pairs_with_matches": sum(1 for r in results if r.num_matches > 0),
            "avg_matches": float(np.mean([r.num_matches for r in results])),
            "avg_precision_pct": float(np.mean([r.precision for r in results]) * 100),
            "avg_inliers": float(np.mean([r.num_inliers for r in results])),
            "avg_match_time_ms": float(np.mean([r.match_time_ms for r in results])),
            "pose_estimated_pairs": n_estimated,
            **pose_auc(pose_errs_all),
            "avg_R_error_deg": (
                float(np.mean([r.R_error_deg for r in results
                               if r.R_error_deg < float("inf")]))
                if n_estimated else float("nan")),
            "avg_t_error_deg": (
                float(np.mean([r.t_error_deg for r in results
                               if r.t_error_deg < float("inf")]))
                if n_estimated else float("nan")),
        }
        _print_summary(summary)
        return summary

# Reporting

def _print_summary(s: Dict):
    print(f"\n  Tag             : {s['tag']}")
    print(f"  Feature type    : {s['feature_type']}")
    print(f"  Pairs           : {s['num_pairs']}  "
          f"(with matches: {s['num_pairs_with_matches']})")
    print(f"  Avg matches     : {s['avg_matches']:.1f}  "
          f"(RANSAC inliers: {s['avg_inliers']:.1f})")
    print(f"  Avg precision   : {s['avg_precision_pct']:.1f}%  (epipolar <=3px)")
    print(f"  Avg match time  : {s['avg_match_time_ms']:.1f} ms")
    print(f"  Pose estimated  : {s['pose_estimated_pairs']} / {s['num_pairs']} pairs")
    for thr in (5, 10, 20):
        print(f"  AUC @ {thr:2d} deg   : {s.get(f'AUC@{thr}deg', 0.0):.1f}%")
    print(f"  Avg R / t err   : "
          f"{s.get('avg_R_error_deg', float('nan')):.2f} deg / "
          f"{s.get('avg_t_error_deg', float('nan')):.2f} deg")

# CLI

def _build_args():
    p = argparse.ArgumentParser(
        description="LightGlue robustness evaluation — KITTI & 4Seasons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    p.add_argument("--feature_type", default="superpoint",
                   choices=["superpoint", "disk", "sift"])
    p.add_argument("--max_keypoints", type=int, default=2048)
    p.add_argument("--device", default=None, help="cuda | cpu (auto-detect)")
    p.add_argument("--epi_thresh", type=float, default=3.0,
                   help="Sampson-distance threshold (px) for precision")
    p.add_argument("--ransac_thresh", type=float, default=1.0,
                   help="RANSAC inlier threshold (px) for pose estimation.")

    p.add_argument("--resize", type=int, default=1600)

    # KITTI
    p.add_argument("--kitti_root", default=None,
                   help="Path to KITTI dataset root "
                        "(folder containing sequences/ and poses/)")
    p.add_argument("--kitti_seqs", nargs="+", default=["00", "05", "08"],
                   metavar="SEQ", help="Sequences with GT poses (00–10 only)")
    p.add_argument("--kitti_step", type=int, default=5,
                   help="Frame stride between consecutive pairs")
    p.add_argument("--kitti_max_pairs", type=int, default=300)

    # 4Seasons
    p.add_argument("--fourseasons_root", default=None,
                   help="Parent folder containing 4Seasons image folders")
    p.add_argument("--fourseasons_poses_root", default=None,
                   help="Parent folder containing 4Seasons reference_poses folders")
    p.add_argument("--fourseasons_recordings", nargs="+", default=None,
                   metavar="REC",
                   help="Base recording names, e.g. recording_2020-03-26_13-32-55")
    p.add_argument("--fourseasons_step", type=int, default=5,  
                   help="Frame stride between consecutive pairs")
    p.add_argument("--fourseasons_max_pairs", type=int, default=300)

    p.add_argument("--output", default="lightglue_results.json")
    p.add_argument("--viz_dir", default="match_images",
                   help="Directory to save sample match visualizations")
    p.add_argument("--viz_n", type=int, default=5,
                   help="Number of random pairs to visualize per dataset/recording")
    return p.parse_args()


def main():
    args = _build_args()

    if not args.kitti_root and not args.fourseasons_root:
        print("Nothing to evaluate — pass --kitti_root and/or --fourseasons_root")
        return

    resize = args.resize if args.resize and args.resize > 0 else None

    evaluator = LightGlueEvaluator(
        feature_type=args.feature_type,
        device=args.device,
        max_keypoints=args.max_keypoints,
        resize=resize,
    )
    all_results: Dict[str, Dict] = {}
    viz_dir = Path(args.viz_dir)

    # KITTI

    if args.kitti_root:
        loader = KITTILoader(
            root=Path(args.kitti_root),
            sequences=args.kitti_seqs,
            step=args.kitti_step,
            max_pairs=args.kitti_max_pairs,
            camera="image_0",
        )
        pairs = loader.load_pairs()
        if pairs:
            evaluator.save_sample_matches(pairs, viz_dir / "kitti", n=args.viz_n)
            r = evaluator.evaluate(pairs, tag="KITTI_all",
                                   epi_thresh=args.epi_thresh,
                                   ransac_thresh=args.ransac_thresh)
            all_results["KITTI_all"] = r
            for seq in args.kitti_seqs:
                sp = [p for p in pairs if f"kitti/{seq}/" in p.name]
                if sp:
                    r = evaluator.evaluate(sp, tag=f"KITTI_seq{seq}",
                                           epi_thresh=args.epi_thresh,
                                           ransac_thresh=args.ransac_thresh)
                    all_results[f"KITTI_seq{seq}"] = r

    # 4Seasons

    if args.fourseasons_root:
        root4 = Path(args.fourseasons_root)
        poses_root = (Path(args.fourseasons_poses_root)
                      if args.fourseasons_poses_root else root4)
        recordings = args.fourseasons_recordings
        if recordings is None:
            recordings = [
                d.name.replace("_stereo_images_undistorted", "")
                for d in sorted(root4.iterdir())
                if d.is_dir() and d.name.endswith("_stereo_images_undistorted")
            ]
            print(f"[4Seasons] Auto-discovered recordings: {recordings}")

        loader = FourSeasonsLoader(
            root=root4,
            poses_root=poses_root,
            recordings=recordings,
            step=args.fourseasons_step,
            max_pairs=args.fourseasons_max_pairs,
        )
        pairs = loader.load_pairs()
        if pairs:
            for rec in recordings:
                rp = [p for p in pairs if f"/{rec}/" in p.name]
                if rp:
                    evaluator.save_sample_matches(
                        rp, viz_dir / "4seasons" / rec, n=args.viz_n)
            r = evaluator.evaluate(pairs, tag="4Seasons_all",
                                   epi_thresh=args.epi_thresh,
                                   ransac_thresh=args.ransac_thresh)
            all_results["4Seasons_all"] = r
            for rec in recordings:
                rp = [p for p in pairs if f"/{rec}/" in p.name]
                if rp:
                    r = evaluator.evaluate(rp, tag=f"4Seasons_{rec}",
                                           epi_thresh=args.epi_thresh,
                                           ransac_thresh=args.ransac_thresh)
                    all_results[f"4Seasons_{rec}"] = r

    if not all_results:
        print("No pairs were loaded. Check dataset paths and missing downloads.")
        return

    out = Path(args.output)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved -> {out}")


if __name__ == "__main__":
    main()
