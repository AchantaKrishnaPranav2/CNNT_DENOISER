"""
utils.py

Minimal data utilities for denoising training. This file purposely avoids any
printing or plotting. It provides:
 - create_denoise_datasets(...) -> train_ds, val_ds
 - simple preprocessing helpers
"""

import os
from glob import glob
import numpy as np
import tensorflow as tf
import cv2


_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")


def _list_images(folder):
    files = []
    for e in _IMAGE_EXTS:
        files.extend(glob(os.path.join(folder, e)))
    files = sorted(files)
    return files


def _read_and_preprocess(path, target_size=(256, 256), as_gray=True):
    """
    Read image with OpenCV and return float32 array scaled to [0,1] with shape (H,W,1).
    This function is used with tf.numpy_function (so it accepts and returns numpy arrays).
    """
    p = path.decode() if isinstance(path, (bytes,)) else str(path)
    if as_gray:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(p, cv2.IMREAD_COLOR)

    if img is None:
        # If the image can't be read, return a black image to keep dataset stable.
        h, w = target_size
        if as_gray:
            return np.zeros((h, w, 1), dtype=np.float32)
        else:
            return np.zeros((h, w, 3), dtype=np.float32)

    img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    if as_gray:
        img = np.expand_dims(img, axis=-1)
    return img


def create_denoise_datasets(noisy_dir, clean_dir,
                            target_size=(256, 256),
                            batch_size=8, test_size=0.2, shuffle=True, seed=42):
    """
    Build train and validation tf.data.Datasets for denoising.
    Expects matching filenames between noisy_dir and clean_dir (same order).
    Returned datasets yield (noisy_batch, clean_batch), both float32 in [0,1].
    """

    noisy_paths = _list_images(noisy_dir)
    clean_paths = _list_images(clean_dir)

    if len(noisy_paths) == 0 or len(clean_paths) == 0:
        raise FileNotFoundError("No images found in provided folders.")

    if len(noisy_paths) != len(clean_paths):
        raise ValueError("Number of noisy images and clean images must match.")

    # deterministic split
    idx = np.arange(len(noisy_paths))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_test = int(len(idx) * test_size)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    def subset(paths, ids):
        return [paths[i] for i in ids]

    train_noisy = subset(noisy_paths, train_idx)
    train_clean = subset(clean_paths, train_idx)
    val_noisy = subset(noisy_paths, test_idx)
    val_clean = subset(clean_paths, test_idx)

    def _tf_loader(noisy_p, clean_p):
        noisy, clean = tf.numpy_function(
            lambda a,b: (_read_and_preprocess(a, target_size=target_size, as_gray=True),
                         _read_and_preprocess(b, target_size=target_size, as_gray=True)),
            [noisy_p, clean_p],
            [tf.float32, tf.float32]
        )
        noisy.set_shape([target_size[0], target_size[1], 1])
        clean.set_shape([target_size[0], target_size[1], 1])
        return noisy, clean

    train_ds = tf.data.Dataset.from_tensor_slices((train_noisy, train_clean))
    val_ds = tf.data.Dataset.from_tensor_slices((val_noisy, val_clean))

    if shuffle:
        train_ds = train_ds.shuffle(buffer_size=512, seed=seed)

    train_ds = train_ds.map(_tf_loader, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(_tf_loader, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
