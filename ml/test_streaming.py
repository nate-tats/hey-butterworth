"""Check streaming inference == full-window Keras inference.

1. Streaming front end (random chunk sizes) == batch log-mel.
2. For every window start 24*j, Keras(model, mel[24j : 24j+N_FRAMES]) logit
   == j-th logit from StreamingButterworth fed frames in uneven chunks.
3. random_crop() returns valid crops covering all offsets.
4. Report MAC counts: full window vs streaming per hop.
"""
import numpy as np
import tensorflow as tf
import features as F
import train as T

rng = np.random.default_rng(0)

# ---- random model with non-trivial BN stats so BN folding is exercised
net = T.build_model(C=64)
for i in range(1, 7):
    bn = net.get_layer(f"bn{i}")
    C = bn.get_weights()[0].shape[0]
    bn.set_weights([rng.uniform(0.5, 1.5, C), rng.normal(0, 0.1, C),
                    rng.normal(0, 0.5, C), rng.uniform(0.5, 2.0, C)])
assert net.output_shape == (None, 1)
assert net.get_layer("relu6").output.shape[1] == 1

# ---- 6 s of fake audio
audio = rng.normal(0, 0.1, 6 * F.SR).astype(np.float32)

# 1. front end
mel_batch = F.logmel(audio)
fe, parts, pos = F.StreamingFrontend(), [], 0
while pos < len(audio):
    n = int(rng.integers(1, 5000))
    parts.append(fe.push(audio[pos:pos + n]))
    pos += n
mel_stream = np.concatenate(parts)
assert mel_stream.shape == mel_batch.shape, (mel_stream.shape, mel_batch.shape)
print(f"frontend: {len(mel_batch)} frames, max |diff| = "
      f"{np.abs(mel_stream - mel_batch).max():.2e}")

# 2. network
starts = np.arange(0, len(mel_batch) - F.N_FRAMES + 1, T.TOTAL_STRIDE)
windows = np.stack([mel_batch[s:s + F.N_FRAMES] for s in starts])
# Reference logit = Dense pre-activation on the Keras features
dense = net.get_layer("p_butterworth")
feat = tf.keras.Model(net.input, net.get_layer("feat").output)(windows).numpy()
ref = feat @ dense.get_weights()[0][:, 0] + dense.get_weights()[1][0]

stream = T.StreamingButterworth(net)
got, pos = [], 0
while pos < len(mel_batch):
    n = int(rng.integers(1, 60))
    got.append(stream.push(mel_batch[pos:pos + n]))
    pos += n
got = np.concatenate(got)

assert len(got) == len(ref), (len(got), len(ref))
err = np.abs(got - ref).max() / np.abs(ref).max()
print(f"network: {len(ref)} decisions, max rel. logit diff = {err:.2e}")
assert err < 1e-4

p_ref = net(windows).numpy()[:, 0]
print(f"prob max |diff| = {np.abs(T.sigmoid(got) - p_ref).max():.2e}")

# 3. random crops
clip = F.logmel(rng.normal(0, 0.1, F.CLIP_SAMPLES).astype(np.float32))
assert clip.shape == (F.CLIP_FRAMES, F.N_MELS)
seen = set()
for _ in range(2000):
    c = F.random_crop(clip, rng)
    assert c.shape == (F.N_FRAMES, F.N_MELS)
    off = int(np.flatnonzero((clip[:, 0] == c[0, 0]))[0])
    assert np.array_equal(c, clip[off:off + F.N_FRAMES])
    seen.add(off)
assert seen == set(range(F.N_CROPS)), sorted(seen)
c = F.random_crop(clip, rng, min_offset=10, max_offset=12)
print(f"crops: {F.CLIP_FRAMES}-frame clip -> {F.N_CROPS} offsets, all hit")

# 4. cost
full = sum(int(np.prod(net.get_layer(f"conv{i}").get_weights()[0].shape))
           * net.get_layer(f"conv{i}").output.shape[1] for i in range(1, 7)) + 64
print(f"MACs: full window {full/1e6:.3f} M | streaming per hop "
      f"{stream.macs_per_hop()/1e6:.3f} M | params {net.count_params():,}")
print("OK")
