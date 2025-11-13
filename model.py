"""
model.py

Denoiser-only CNNT-style model (clean, readable, and ready for use).
This file defines:
 - InstanceNormalization (small, explicit)
 - MultiHeadConvQKAttention (local conv Q-K attention)
 - cnnt_cell (attention + mixer)
 - build_cnnt_denoiser(...) -> returns a compiled Keras Model suitable for image denoising
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, BatchNormalization, Activation,
    MaxPooling2D, Add, Concatenate, LayerNormalization, Dense, Dropout, Layer, Input
)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


# -------------------------
# InstanceNormalization
# -------------------------
@tf.keras.utils.register_keras_serializable(package="CNNT")
class InstanceNormalization(Layer):
    """Simple instance normalization with optional learnable affine params."""
    def __init__(self, epsilon=1e-5, affine=False, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)
        self.affine = bool(affine)

    def build(self, input_shape):
        if self.affine:
            channel_dim = int(input_shape[-1])
            self.gamma = self.add_weight(
                name="gamma", shape=(channel_dim,), initializer="ones", trainable=True
            )
            self.beta = self.add_weight(
                name="beta", shape=(channel_dim,), initializer="zeros", trainable=True
            )
        super().build(input_shape)

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.epsilon)
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"epsilon": self.epsilon, "affine": self.affine})
        return cfg


# -------------------------
# Local Multi-Head Conv Q-K Attention
# -------------------------
@tf.keras.utils.register_keras_serializable(package="CNNT")
class MultiHeadConvQKAttention(Layer):
    """
    Local convolutional multi-head attention.
    For every spatial position it computes dot(Q_center, K_patch) over a kxk patch,
    softmaxes across the patch positions, and aggregates V accordingly.
    """

    def __init__(self, heads=4, kernel_size=3, ff_dim=None, proj_if_not_divisible=True, **kwargs):
        super().__init__(**kwargs)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.heads = int(heads)
        self.kernel_size = int(kernel_size)
        self.ff_dim = ff_dim
        self.proj_if_not_divisible = bool(proj_if_not_divisible)

    def build(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError("Input channel dimension must be known.")
        self.C = int(input_shape[-1])

        if self.proj_if_not_divisible and (self.C % self.heads != 0):
            outC = ((self.C + self.heads - 1) // self.heads) * self.heads
        else:
            outC = self.C

        self.out_channels = int(outC)
        self.depth = self.out_channels // self.heads

        name_base = self.name or "mhcqa"
        self.q_proj = Conv2D(self.out_channels, 1, padding="same", use_bias=False, name=f"{name_base}_q")
        self.k_proj = Conv2D(self.out_channels, 1, padding="same", use_bias=False, name=f"{name_base}_k")
        self.v_proj = Conv2D(self.out_channels, 1, padding="same", use_bias=False, name=f"{name_base}_v")

        if self.C != self.out_channels:
            self.res_proj = Conv2D(self.out_channels, 1, padding="same", use_bias=False, name=f"{name_base}_res")
        else:
            self.res_proj = None

        # small fusion and FFN
        self.fuse_conv = Conv2D(self.out_channels, 1, padding="same", use_bias=False, name=f"{name_base}_fuse")
        self.fuse_bn = BatchNormalization(name=f"{name_base}_fuse_bn")
        self.fuse_act = Activation("relu", name=f"{name_base}_fuse_act")

        self.attn_ln = LayerNormalization(name=f"{name_base}_attn_ln")
        ff_val = self.ff_dim if self.ff_dim is not None else max(64, self.out_channels * 2)
        self.ff1 = Dense(ff_val, activation="relu", name=f"{name_base}_ff1")
        self.ff2 = Dense(self.out_channels, name=f"{name_base}_ff2")
        self.ff_ln = LayerNormalization(name=f"{name_base}_ff_ln")

        super().build(input_shape)

    def call(self, x):
        # q,k,v convs
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        bs = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = self.out_channels
        ks = self.kernel_size
        ka = ks * ks

        # q -> (B,H,W,heads,depth)
        q = tf.reshape(q, [bs, H, W, self.heads, self.depth])

        # extract patches for k and v: (B,H,W,ka*C)
        patches_k = tf.image.extract_patches(k, sizes=[1, ks, ks, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding='SAME')
        patches_v = tf.image.extract_patches(v,     sizes=[1, ks, ks, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding='SAME')

        patches_k = tf.reshape(patches_k, [bs, H, W, ka, C])
        patches_v = tf.reshape(patches_v, [bs, H, W, ka, C])
        patches_k = tf.reshape(patches_k, [bs, H, W, ka, self.heads, self.depth])
        patches_v = tf.reshape(patches_v, [bs, H, W, ka, self.heads, self.depth])

        # dot product q (center) with each k in window
        q_exp = tf.expand_dims(q, axis=3)  # (B,H,W,1,heads,depth)
        sim = tf.reduce_sum(q_exp * patches_k, axis=-1)  # (B,H,W,ka,heads)

        # scale and softmax across patch positions
        sim = sim / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        attn = tf.nn.softmax(sim, axis=3)

        # weighted sum of v patches -> collapse ka
        attn_exp = tf.expand_dims(attn, axis=-1)  # (B,H,W,ka,heads,1)
        weighted_v = attn_exp * patches_v             # (B,H,W,ka,heads,depth)
        out_heads = tf.reduce_sum(weighted_v, axis=3) # (B,H,W,heads,depth)

        out = tf.reshape(out_heads, [bs, H, W, C])

        # residual connection (project if needed)
        res = x if self.res_proj is None else self.res_proj(x)
        out = Add()([res, out])
        out = self.attn_ln(out)

        # token-wise FFN over spatial sequence
        seq_len = H * W
        tokens = tf.reshape(out, [bs, seq_len, C])

        ff = self.ff1(tokens)
        ff = self.ff2(ff)
        ff = Add()([tokens, ff])
        ff = self.ff_ln(ff)

        out = tf.reshape(ff, [bs, H, W, C])
        out = self.fuse_conv(out)
        out = self.fuse_bn(out)
        out = self.fuse_act(out)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "heads": self.heads,
            "kernel_size": self.kernel_size,
            "ff_dim": self.ff_dim,
            "proj_if_not_divisible": self.proj_if_not_divisible
        })
        return cfg


# -------------------------
# cnnt_cell
# -------------------------
def cnnt_cell(x_in, heads=4, ff_dim=None, mixer_filters=None, dropout_rate=0.1, name=None):
    """
    Single CNNT cell: normalization -> attention -> mixer -> residual add
    """
    a = x_in
    x = InstanceNormalization(name=(None if name is None else f"{name}_in1"))(x_in)
    x = Add(name=(None if name is None else f"{name}_add_a"))([x, a])

    attn = MultiHeadConvQKAttention(heads=heads, kernel_size=3, ff_dim=ff_dim,
                                    proj_if_not_divisible=True,
                                    name=(None if name is None else f"{name}_attn"))(x)
    b = attn

    x = InstanceNormalization(name=(None if name is None else f"{name}_in2"))(attn)

    current_ch = K.int_shape(x)[-1]
    if mixer_filters is None:
        mixer_filters = current_ch

    x = Conv2D(mixer_filters, 3, padding="same", use_bias=False, name=(None if name is None else f"{name}_mix_conv1"))(x)
    x = Activation('gelu', name=(None if name is None else f"{name}_gelu"))(x)
    x = Conv2D(mixer_filters, 3, padding="same", use_bias=False, name=(None if name is None else f"{name}_mix_conv2"))(x)
    x = Dropout(dropout_rate, name=(None if name is None else f"{name}_drop"))(x)

    b_channels = K.int_shape(b)[-1]
    if b_channels != mixer_filters:
        raise ValueError(f"Channel mismatch inside cnnt_cell('{name}'): b has {b_channels} channels, mixer expected {mixer_filters}.")

    x = Add(name=(None if name is None else f"{name}_add_b"))([x, b])
    return x


# -------------------------
# down_block / up_block helpers
# -------------------------
def down_block(x, cells_per_block=3, heads=4, ff_dim=None, mixer_filters=None, dropout_rate=0.1, name=None):
    x_in = x
    for i in range(cells_per_block):
        x_in = cnnt_cell(x_in, heads=heads, ff_dim=ff_dim, mixer_filters=mixer_filters,
                         dropout_rate=dropout_rate, name=(None if name is None else f"{name}_cell{i}"))
    skip = x_in
    x_down = MaxPooling2D(pool_size=2, name=(None if name is None else f"{name}_pool"))(x_in)
    return x_down, skip


def up_block(x, skip, cells_per_block=3, heads=4, ff_dim=None, mixer_filters=None, dropout_rate=0.1, name=None):
    target_ch = K.int_shape(skip)[-1]
    x = Conv2DTranspose(target_ch, kernel_size=2, strides=2, padding="same", name=(None if name is None else f"{name}_deconv"))(x)
    x = BatchNormalization(name=(None if name is None else f"{name}_debn"))(x)
    x = Activation("relu", name=(None if name is None else f"{name}_deact"))(x)

    x = Concatenate(axis=-1, name=(None if name is None else f"{name}_concat"))([x, skip])
    x = Conv2D(target_ch, kernel_size=1, padding="same", use_bias=False, name=(None if name is None else f"{name}_concat_fuse"))(x)
    x = BatchNormalization(name=(None if name is None else f"{name}_concat_fuse_bn"))(x)
    x = Activation("relu", name=(None if name is None else f"{name}_concat_fuse_act"))(x)

    x_in = x
    for i in range(cells_per_block):
        x_in = cnnt_cell(x_in, heads=heads, ff_dim=ff_dim, mixer_filters=mixer_filters,
                         dropout_rate=dropout_rate, name=(None if name is None else f"{name}_cell{i}"))
    return x_in


# -------------------------
# Build CNNT denoiser (public function)
# -------------------------
def build_cnnt_denoiser(
    input_shape=(256, 256, 1),
    base_filters=32,
    depth=2,
    cells_per_block=3,
    heads=4,
    ff_dim=128,
    mixer_filters=None,
    dropout_rate=0.1,
    output_channels=1
):
    """
    Returns a compiled Keras Model for denoising.
    The model uses a small encoder-decoder with CNNT cells and local attention.
    """

    inputs = Input(shape=input_shape, name="noisy_input")
    x = Conv2D(base_filters, 3, padding="same", use_bias=False, name="pre_conv")(inputs)
    x = BatchNormalization(name="pre_bn")(x)
    x = Activation("relu", name="pre_act")(x)

    skip0 = x
    skips = []
    x_enc = x
    for d in range(depth):
        x_enc, skip = down_block(x_enc, cells_per_block=cells_per_block, heads=heads,
                                 ff_dim=ff_dim, mixer_filters=mixer_filters, dropout_rate=dropout_rate,
                                 name=f"down{d+1}")
        skips.append(skip)

    # bottleneck
    x_b = x_enc
    for i in range(cells_per_block):
        x_b = cnnt_cell(x_b, heads=heads, ff_dim=ff_dim, mixer_filters=mixer_filters,
                        dropout_rate=dropout_rate, name=(f"bottleneck_cell{i}"))

    # decoder
    x_dec = x_b
    for i, skip in enumerate(reversed(skips)):
        x_dec = up_block(x_dec, skip, cells_per_block=cells_per_block, heads=heads,
                         ff_dim=ff_dim, mixer_filters=mixer_filters, dropout_rate=dropout_rate,
                         name=f"up{depth-i}")

    # final add (ensure same channels)
    skip_ch = K.int_shape(skip0)[-1]
    dec_ch = K.int_shape(x_dec)[-1]
    if skip_ch != dec_ch:
        # fuse to match channels
        x_dec = Conv2D(skip_ch, 1, padding="same", use_bias=False, name="final_match_conv")(x_dec)
        x_dec = BatchNormalization(name="final_match_bn")(x_dec)
        x_dec = Activation("relu", name="final_match_act")(x_dec)

    x_dec = Add(name="final_add_skip0")([x_dec, skip0])

    x_out = Conv2D(base_filters, 3, padding="same", use_bias=False, name="post_conv1")(x_dec)
    x_out = BatchNormalization(name="post_bn1")(x_out)
    x_out = Activation("relu", name="post_act1")(x_out)
    outputs = Conv2D(output_channels, 1, activation="sigmoid", name="output_conv")(x_out)

    model = Model(inputs=inputs, outputs=outputs, name="CNNT_Denoiser")
    return model
