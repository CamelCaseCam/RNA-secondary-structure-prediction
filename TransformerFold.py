#_______________________________________________________________________________________________________________________________
# TFUtils.py
#_______________________________________________________________________________________________________________________________
# MIT License

# Copyright (c) 2020 Streack, Jayakrishna Sahit

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
from tensorflow.keras import layers

def make_unit_length(x, epsilon=1e-6):
    norm = tf.norm(x,  ord=2, axis=-1, keepdims=True)
    return tf.math.truediv(x, norm + epsilon)

def sort_key_val(t1, t2, dim=-1):
    values = tf.sort(t1, axis=dim)
    t2 = tf.broadcast_to(t2, tf.shape(t1))
    return values, tf.gather(t2, tf.argsort(t1, axis=dim), axis=dim)

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return tf.squeeze(tf.gather(values, indices[:, :, None], axis=1))

def process_inputs_chunk(fn, *args, chunks=1):
    chunked_inputs = list(map(lambda x: tf.split(x, chunks, axis=0), args))
    outputs = [fn(*input_pair) for input_pair in zip(*chunked_inputs)]
    return outputs

def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tf.reshape(tensor,  [-1, last_dim])
    summed_tensors = [c.sum(axis=-1) for c in tf.chunk(tensor, chunks, axis=0)]
    return tf.reshape(tf.concat(summed_tensors, axis=0), orig_size)

def cache_fn(f):
    cache = None
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class ScaleNorm(layers.Layer):
    def __init__(self, emb, eps):
        super(ScaleNorm, self).__init__()
        self.g = tf.Variable(tf.ones(1, dtype=tf.float32),
                        trainable=True)
        self.eps = eps

    def call(self, inputs):
        n = tf.norm(inputs, axis=-1, keepdims=True).clip_by_value(min=self.eps)
        return inputs / n * self.g

class WithNorm(layers.Layer):
    def __init__(self, norm_class, emb, fn):
        super(WithNorm, self).__init__()
        self.emb = emb
        if isinstance(norm_class, ScaleNorm):
            self.norm = norm_class(emb)
        else:
            self.norm = norm_class()

        self.fn = fn

    def call(self, inputs):
        inputs = self.norm(inputs)
        return self.fn(inputs)

class Chunk(layers.Layer):
    def __init__(self, chunks, fn, along_axis = -1):
        super(Chunk, self).__init__()
        self.axis = along_axis
        self.chunks = chunks
        self.fn = fn

    def call(self, inputs):
        chunks = tf.split(inputs, self.chunks, axis= self.axis)
        return tf.concat([self.fn(c) for c in chunks], axis = self.axis)

#_______________________________________________________________________________________________________________________________
# TFEfficientAttention.py
#_______________________________________________________________________________________________________________________________

# MIT License

# Copyright (c) 2020 Streack, Jayakrishna Sahit

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense

class TFLSHAttention(tf.keras.Model):
    def __init__( self,
                  dropout = 0.,
                  bucket_size = 64,
                  n_hashes = 8,
                  causal = False,
                  allow_duplicate_attention = True,
                  attend_across_buckets = True,
                  rehash_each_round = True,
                  drop_for_hash_rate = 0.0,
                  random_rotations_per_head = False):
        super(TFLSHAttention, self).__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = Dropout(dropout)
        self.dropout_for_hash = Dropout(dropout)

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

    def hash_vectors(self, n_buckets, vecs):
        batch_size = tf.shape(vecs)[0]
        device = vecs.device

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        tf.assert_equal(n_buckets % 2, 0)

        rot_size = n_buckets

        rotations_shape = (
            batch_size if self._random_rotations_per_head else 1,
            tf.shape(vecs)[-1],
            self.n_hashes if self._rehash_each_round else 1,
            rot_size // 2)

        random_rotations = tf.broadcast_to(tf.random.normal(rotations_shape), (batch_size, tf.shape(vecs)[-1], self.n_hashes if self._rehash_each_round else 1, rot_size // 2))

        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = tf.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)

        if self._rehash_each_round:
            rotated_vecs = tf.concat([rotated_vecs, -rotated_vecs], axis=-1)
            buckets = tf.math.argmax(rotated_vecs, axis=-1)
            # buckets is now (self.n_hashes, seqlen). Next we add offsets so that
            # bucket numbers from different hashing rounds don't overlap.
            offsets = tf.range(self.n_hashes)
            offsets = tf.reshape(offsets * n_buckets, (1, -1, 1))
            offsets = tf.cast(offsets, tf.int64)
            buckets = tf.reshape(buckets + offsets, (batch_size, -1,))
        else:
            rotated_vecs = tf.concat([rotated_vecs, -rotated_vecs], axis=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = tf.squeeze(rotated_vecs, axis=0)
            bucket_range = tf.range(rotated_vecs.shape[-1])
            bucket_range = tf.reshape(bucket_range, (1, -1))
            bucket_range = tf.broadcast_to(bucket_range, rotated_vecs.shape)

            _, buckets = sort_key_val(rotated_vecs, bucket_range, axis=-1)
            buckets = buckets[:, -self.n_hashes:]

            h, *_ = buckets.shape 
            buckets = tf.reshape(buckets.permute((*_, h)), (-1,))

        return buckets

    def call(self, qk, v):
        batch_size, seqlen = tf.shape(qk)[0], tf.shape(qk)[1]
        device = qk.device

        n_buckets = seqlen // self.bucket_size
        n_bins = n_buckets

        buckets = tf.cast(self.hash_vectors(n_buckets, qk), tf.int32)
        # We use the same vector as both a query and a key.
        tf.assert_equal(tf.cast(tf.shape(buckets)[1], dtype=tf.int32), self.n_hashes * seqlen)

        ticker = tf.expand_dims(tf.range(self.n_hashes * seqlen), axis=0)
        buckets_and_t = seqlen * buckets + tf.cast((ticker % seqlen), tf.int32)
        buckets_and_t = tf.stop_gradient(buckets_and_t)

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sort_key_val(sticker, ticker, dim=-1)
        del ticker

        sbuckets_and_t = tf.stop_gradient(sbuckets_and_t)
        sticker = tf.stop_gradient(sticker)
        undo_sort = tf.stop_gradient(undo_sort)

        st = (sticker % seqlen)
        sqk = batched_index_select(qk, st)
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        bq_t = bkv_t = tf.reshape(st, (batch_size, self.n_hashes * n_bins, -1))
        bqk = tf.reshape(sqk, (batch_size, self.n_hashes * n_bins, -1, tf.shape(sqk)[-1]))
        bv = tf.reshape(sv, (batch_size, self.n_hashes * n_bins, -1, tf.shape(sv)[-1]))
        bq_buckets = bkv_buckets = tf.reshape(sbuckets_and_t // seqlen, (batch_size, self.n_hashes * n_bins, -1))

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = make_unit_length(bqk)

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = tf.concat([x[:, -1:, ...], x[:, :-1, ...]], axis=1)
            return tf.concat([x, x_extra], axis=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)
        bkv_buckets = look_one_back(bkv_buckets)

        # Dot-product attention.
        dots = tf.einsum('bhie,bhje->bhij', bq, bk) * (tf.cast(tf.shape(bq)[-1], dtype=tf.float32) ** -0.5)

        # Causal masking
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :] 
            dots = tf.math.multiply(dots, tf.cast(mask, tf.float32)) + (1-tf.cast(mask, tf.float32)) * float('-inf')
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots = tf.math.multiply(dots, tf.cast(self_mask, tf.float32)) + (1-tf.cast(self_mask, tf.float32)) * (- 1e5)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets:
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            dots = tf.math.multiply(dots, tf.cast(bucket_mask, tf.float32)) + (1-tf.cast(bucket_mask, tf.float32)) * float('-inf')
            del bucket_mask

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % (self.n_hashes * n_bins)
            if not self._attend_across_buckets:
                locs1 = buckets * (self.n_hashes * n_bins) + locs1
                locs2 = buckets * (self.n_hashes * n_bins) + locs2
            locs = tf.transpose(
                tf.concat([
                    tf.reshape(locs1, (batch_size, self.n_hashes, seqlen)),
                    tf.reshape(locs2, (batch_size, self.n_hashes, seqlen)),
                ], 1),
            perm=[0, 2, 1]) 

            slocs = batched_index_select(locs, st)
            b_locs = tf.reshape(slocs, (batch_size, self.n_hashes * n_bins, -1, 2 * self.n_hashes))

            b_locs1 = b_locs[:, :, :, None, :self.n_hashes]

            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, self.n_hashes))
            bq_locs = tf.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)

            dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :])
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(self.n_hashes * batch_size))
            dup_counts = tf.stop_gradient(dup_counts)
            assert dup_counts.shape == dots.shape
            dots = dots - tf.log(dup_counts + 1e-9)
            del dup_counts

        # Softmax.
        dots_logsumexp = tf.math.reduce_logsumexp(dots, axis=-1, keepdims=True)
        dots = tf.exp(dots - dots_logsumexp)
        dots = self.dropout(dots)

        bo = tf.einsum('buij,buje->buie', dots, bv)
        so = tf.reshape(bo, (batch_size, -1, tf.shape(bo)[-1]))
        slogits = tf.reshape(dots_logsumexp, (batch_size, -1,))

        class UnsortLogits(tf.keras.layers.Layer):
            def __init__(self):
                super(UnsortLogits, self).__init__()
            
            def call(self, so, slogits):
                so, slogits = tf.stop_gradient(so), tf.stop_gradient(slogits)
                o = batched_index_select(so, undo_sort)
                _, logits = sort_key_val(sticker, slogits, dim=-1)
                return o, logits

            
        unsortlogits = UnsortLogits()
        o, logits = unsortlogits(so, slogits)

        if self.n_hashes == 1:
            out = o
        else:
            o = tf.reshape(o, (batch_size, self.n_hashes, seqlen, tf.shape(o)[-1]))
            logits = tf.reshape(logits, (batch_size, self.n_hashes, seqlen, 1))
            probs = tf.exp(logits - tf.math.reduce_logsumexp(logits, axis=1, keepdims=True))
            out = tf.reduce_sum(o * probs, axis=1)

        assert out.shape == v.shape
        return out, buckets

class TFLSHSelfAttention(tf.keras.Model):
    def __init__(self, emb, heads = 8, bucket_size = 64, n_hashes = 8, causal = False, attn_chunks = None, random_rotations_per_head = False, attend_across_buckets = True, allow_duplicate_attention = True, **kwargs):
        super(TFLSHSelfAttention, self).__init__()
        assert emb % heads == 0, 'dimensions must be divisible by number of heads'

        self.emb = emb
        self.heads = heads
        self.attn_chunks = heads if attn_chunks is None else attn_chunks

        self.toqk = Dense(emb, use_bias = False)
        self.tov = Dense(emb, use_bias = False)
        self.to_out = Dense(emb)

        self.bucket_size = bucket_size
        self.lsh_attn = TFLSHAttention(bucket_size=bucket_size, causal=causal, random_rotations_per_head=random_rotations_per_head, attend_across_buckets = attend_across_buckets,  allow_duplicate_attention = allow_duplicate_attention, **kwargs)

    def call(self, inputs, value = None):
        b, t, e, h = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], self.heads
        tf.assert_equal((t % self.bucket_size), 0, f'Sequence length needs to be divisible by target bucket size - {self.bucket_size}')

        if value is None:
          qk = self.toqk(inputs)
          v = self.tov(inputs)
        else:
          qk = self.toqk(inputs)
          v = self.tov(value)

        def merge_heads(v):
            return tf.reshape(tf.transpose(tf.reshape(v, (b, t, h, -1)), perm=[0, 2, 1, 3]), (b * h, t, -1)) 

        def split_heads(v):
            return tf.transpose(tf.reshape(v, (b, t, h, -1)), perm=[0, 2, 1, 3])

        qk = merge_heads(qk)
        v = merge_heads(v)

        outputs = process_inputs_chunk(self.lsh_attn, qk, v, chunks=self.attn_chunks)
        attn_out = tf.concat([output for (output, _) in outputs], axis=0)

        out = tf.reshape(split_heads(attn_out), (b, t, e))

        return self.to_out(out)


#_______________________________________________________________________________________________________________________________
# Main model
#_______________________________________________________________________________________________________________________________




'''
Transformer model that predicts the secondary structure of an RNA sequence based on the primary sequence. Input is a 1-hot encoded sequence of RNA nucleotides. 
Output is a vector of values -1 to 1 indicating whether the nucleotide is bonded to a previous nucleotide (negative), not bonded (0), 
or bonded to a subsequent nucleotide (positive). The absolute value of the output indicates the end of the sequence that the bonded nucleotide is located.
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load data
dots = np.load('D:\Development\RNAFolding\\archiveII\output_directional/dots.npz')['arr_0']
seqs = np.load('D:\Development\RNAFolding\\archiveII\output_directional/seqs.npz')['arr_0']

# Split data into training and testing sets
train_dots = dots[:int(len(dots) * 0.8)]
train_seqs = seqs[:int(len(seqs) * 0.8)]
test_dots = dots[int(len(dots) * 0.8):]
test_seqs = seqs[int(len(seqs) * 0.8):]

MAX_SEQUENCE = 2048

def ProcessBatch(x, y):
    return (x, tf.expand_dims(tf.cast(x > 0, tf.float32), axis=-1)), y

# shuffle the training data
train_data = tf.data.Dataset.from_tensor_slices((train_seqs, train_dots)).shuffle(10000).batch(16).map(ProcessBatch)
test_data = tf.data.Dataset.from_tensor_slices((test_seqs, test_dots)).shuffle(10000).batch(16).map(ProcessBatch)



#_____________________________________________________________________________________________________________________________________________________
# Define the model
#_____________________________________________________________________________________________________________________________________________________



def PositionalEncoding(length, depth):
  depth = depth/2
  
  positions = np.arange(length)[:, np.newaxis]
  depths = np.arange(depth)[np.newaxis, :]/depth

  AngleRates = 1 / (10000**depths)
  AngleRads = positions * AngleRates

  PosEncoding = np.concatenate([np.sin(AngleRads), np.cos(AngleRads)], axis=-1)

  return tf.cast(PosEncoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, ModelDepth):
    super().__init__()
    self.ModelDepth = ModelDepth
    self.embedding = tf.keras.layers.Embedding(vocab_size, ModelDepth, mask_zero=True)
    self.pos_encoding = PositionalEncoding(length=2048, depth=ModelDepth)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.ModelDepth, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    #self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.mha = TFLSHSelfAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class CrossAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = TFLSHSelfAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

  def call(self, x, context):
    #attn_output, attn_scores = self.mha(query=x, key=context, value=context, return_attention_scores=True)
    #self.LastAttentionScores = attn_scores
    attn_output = self.mha(x, value=context)

    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    #attn_output = self.mha(query=x, value=x, key=x)
    attn_output = self.mha(x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
    
class CausalSelfAttention(BaseAttention):
  def __init__(self, **kwargs):
    super().__init__(**kwargs, causal=True)
  def call(self, x):
    #attn_output = self.mha(query=x, key=x, value=x, use_causal_mask=True)
    attn_output = self.mha(x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, ModelDepth, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(ModelDepth),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layernorm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layernorm(x)
    return x

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, *, ModelDepth, NumHeads, dff, dropout_rate=0.1):
    super().__init__()

    self.SelfAttention = GlobalSelfAttention(heads=NumHeads, emb=ModelDepth, dropout=dropout_rate)
    self.ffn = FeedForward(ModelDepth, dff)

  def call(self, x):
    x = self.SelfAttention(x)
    x = self.ffn(x)

    return x

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, NumLayers, ModelDepth, NumHeads, dff, VocabSize, dropout_rate=0.1):
    super().__init__()

    self.ModelDepth = ModelDepth
    self.NumLayers = NumLayers

    self.PosEmbedding = PositionalEmbedding(vocab_size=VocabSize, ModelDepth=ModelDepth)

    self.EncLayers = [
        EncoderLayer(ModelDepth=ModelDepth, NumHeads=NumHeads, dff=dff, dropout_rate=dropout_rate)
        for _ in range(NumLayers)
    ]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    x = self.PosEmbedding(x)

    x = self.dropout(x)

    for i in range(self.NumLayers):
      x = self.EncLayers[i](x)
    
    return x

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, *, ModelDepth, NumHeads, dff, dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    # In this case, we don't need to use the causal mask because we'll generate the bonds all at once
    # We can make this process iterative by feeding the previous iteration's output into the next iteration
    #self.CausalSelfAttention = CausalSelfAttention(num_heads=NumHeads, key_dim=ModelDepth, dropout=dropout_rate)

    self.CrossAttention = CrossAttention(heads=NumHeads, emb=ModelDepth, dropout=dropout_rate)

    self.ffn = FeedForward(ModelDepth, dff)

  def call(self, x, context):
    #x = self.CausalSelfAttention(x=x)
    x = self.CrossAttention(x, context=context)

    #self.LastAttentionScores = self.CrossAttention.LastAttentionScores

    x = self.ffn(x)
    return x

class BondInput(tf.keras.layers.Layer):
  def __init__(self, ModelDepth):
    super().__init__()
    self.dense = tf.keras.layers.Dense(ModelDepth)
    self.pos_encoding = PositionalEncoding(length=2048, depth=ModelDepth)
    self.ModelDepth = ModelDepth

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.dense(x)
    x *= tf.math.sqrt(tf.cast(self.ModelDepth, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, NumLayers, ModelDepth, NumHeads, dff, dropout_rate=0.1):
    super(Decoder, self).__init__()
    
    self.ModelDepth = ModelDepth
    self.NumLayers = NumLayers

    # Because we're generating the bonds all at once, we shouldn't use an embedding layer here. Instead, we'll use a dense layer to convert bonds to vectors
    self.pos_embedding = BondInput(ModelDepth=ModelDepth)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.DecLayers = [
        DecoderLayer(ModelDepth=ModelDepth, NumHeads=NumHeads, dff=dff, dropout_rate=dropout_rate)
        for _ in range(NumLayers)
    ]

    self.LastAttentionScores = None

  def call(self, x, context):
    x = self.pos_embedding(x)

    x = self.dropout(x)

    for i in range(self.NumLayers):
      x = self.DecLayers[i](x, context)

    #self.LastAttentionScores = self.DecLayers[-1].LastAttentionScores

    return x

class Transformer(tf.keras.Model):
  def __init__(self, *, NumLayers, ModelDepth, NumHeads, dff, InputVocabSize, MaxSeqLen, dropout_rate=0.1):
    super().__init__()
    
    self.encoder = Encoder(NumLayers=NumLayers, ModelDepth=ModelDepth, NumHeads=NumHeads, dff=dff, VocabSize=InputVocabSize, dropout_rate=dropout_rate)

    self.decoder = Decoder(NumLayers=NumLayers, ModelDepth=ModelDepth, NumHeads=NumHeads, dff=dff, dropout_rate=dropout_rate)

    self.FinalLayer = tf.keras.layers.Dense(MaxSeqLen, activation='tanh')

  def call(self, inputs):
    context, x = inputs

    context = self.encoder(context)

    x = self.decoder(x, context)

    logits = self.FinalLayer(x)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass
    
    return logits

NumLayers = 4
ModelDepth = 8
dff = 512
NumHeads = 4
dropout_rate = 0.1

transformer = Transformer(NumLayers=NumLayers, ModelDepth=ModelDepth, NumHeads=NumHeads, dff=dff, MaxSeqLen=MAX_SEQUENCE, InputVocabSize=5, dropout_rate=dropout_rate)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, WarmupSteps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.WarmupSteps = WarmupSteps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.WarmupSteps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

LearningRate = CustomSchedule(ModelDepth)

optimizer = tf.keras.optimizers.Adam(LearningRate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

def get_pad_mask(data):
  return data[1]

def get_data(data):
  return data[0]

def MaskedLoss(y_true, y_pred):
    pad_mask = tf.map_fn(get_pad_mask, y_true)
    num_actual = tf.reduce_sum(pad_mask)
    pad_mask = tf.reshape(pad_mask, (-1, 2048, 1))

    y_true = tf.map_fn(get_data, y_true)
    y_true = tf.reshape(y_true, (-1, 2048, 1))

    return tf.reduce_sum(tf.abs(y_true - y_pred) * pad_mask) / num_actual

def MaskedAccuracy(y_true, y_pred):
    pad_mask = tf.map_fn(get_pad_mask, y_true)
    num_actual = tf.reduce_sum(pad_mask)
    pad_mask = tf.reshape(pad_mask, (-1, 2048, 1))

    y_true = tf.map_fn(get_data, y_true)
    y_true = tf.reshape(y_true, (-1, 2048, 1))

    bonded_true = tf.cast(tf.abs(y_true) > 0.25, tf.float32)
    bonded_pred = tf.cast(tf.abs(y_true) > 0.25, tf.float32)
    bond_dirs_true = tf.sign(y_true) * bonded_true
    bond_dirs_pred = tf.sign(y_pred) * bonded_pred

    num_correct = tf.reduce_sum(tf.cast(bond_dirs_pred == bond_dirs_true, dtype=tf.float32) * pad_mask)
    return num_correct / num_actual

transformer.compile(optimizer=optimizer, loss=MaskedLoss, metrics=[MaskedAccuracy])

transformer.fit(train_data, epochs=10, validation_data=test_data)