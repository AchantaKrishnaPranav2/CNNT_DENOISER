"""
train.py

Minimal, non-verbose training script for the denoiser.
 - Uses build_cnnt_denoiser from model.py
 - Uses create_denoise_datasets from utils.py
 - No image displays, no sample printing, no dataset previews
"""

import os
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import build_cnnt_denoiser
from utils import create_denoise_datasets


def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


@tf.function
def train_step(model, optimizer, noisy_batch, clean_batch):
    with tf.GradientTape() as tape:
        pred = model(noisy_batch, training=True)
        loss_val = mse_loss(clean_batch, pred)
    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_val


def validate(model, val_ds):
    # Simple pass to aggregate average validation loss (no prints inside)
    total = 0.0
    count = 0
    for noisy, clean in val_ds:
        pred = model(noisy, training=False)
        loss_val = mse_loss(clean, pred)
        total += float(loss_val.numpy())
        count += 1
    return (total / count) if count > 0 else float('nan')


def main(args):
    # model config
    model = build_cnnt_denoiser(
        input_shape=(args.height, args.width, args.channels),
        base_filters=args.base_filters,
        depth=args.depth,
        cells_per_block=args.cells_per_block,
        heads=args.heads,
        ff_dim=args.ff_dim,
        mixer_filters=args.mixer_filters,
        dropout_rate=args.dropout_rate,
        output_channels=args.channels
    )

    optimizer = Adam(learning_rate=args.lr)

    # datasets
    train_ds, val_ds = create_denoise_datasets(
        noisy_dir=args.noisy_dir,
        clean_dir=args.clean_dir,
        target_size=(args.height, args.width),
        batch_size=args.batch_size,
        test_size=args.val_split,
        shuffle=True,
        seed=args.seed
    )

    # training loop
    best_val = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, args.save_name)

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for noisy_batch, clean_batch in train_ds:
            loss_val = train_step(model, optimizer, noisy_batch, clean_batch)
            epoch_losses.append(float(loss_val.numpy()))

        # compute validation loss (no printing)
        val_loss = validate(model, val_ds)

        # save best
        if val_loss < best_val:
            best_val = val_loss
            model.save(ckpt_path, include_optimizer=False)

    # final save (overwrite)
    model.save(ckpt_path, include_optimizer=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNNT denoiser (minimal, no-visuals).")
    parser.add_argument("--noisy_dir", type=str, required=True, help="Folder with noisy images (matching clean images).")
    parser.add_argument("--clean_dir", type=str, required=True, help="Folder with clean/ground-truth images.")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Where to save models.")
    parser.add_argument("--save_name", type=str, default="cnnt_denoiser.keras", help="Model filename.")
    parser.add_argument("--width", type=int, default=256, help="Image width.")
    parser.add_argument("--height", type=int, default=256, help="Image height.")
    parser.add_argument("--channels", type=int, default=1, help="Image channels (1 for grayscale).")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--cells_per_block", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=128)
    parser.add_argument("--mixer_filters", type=int, default=None)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
