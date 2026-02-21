# VRAM Breakdown for LiveAvatar S2V-14B Inference
## Settings: 384×256, 48 frames, FP8 quantization, Single GPU

---

## 1. MODEL WEIGHTS (When on GPU)

### Noise Model (14B DiT Transformer)
- **Parameters**: ~14 billion
- **Dtype**: BF16 (2 bytes) with FP8 quantization
- **Base size**: 14B × 2 bytes = 28 GB
- **With FP8**: ~14-20 GB (depends on which layers use FP8)
- **Current status**: ✅ OFFLOADED per-timestep (only on GPU momentarily)

### T5-XXL Text Encoder
- **Parameters**: ~11 billion  
- **Dtype**: BF16
- **Size**: ~22 GB
- **Current status**: ✅ ON CPU (`--t5_cpu`)

### VAE (Wan2.1)
- **Parameters**: ~300-500 million
- **Dtype**: BF16
- **Size**: ~600 MB - 1 GB
- **Current status**: ⚠️ ON GPU during encode/decode phases

### Wav2Vec2 Audio Encoder
- **Parameters**: ~300 million
- **Dtype**: BF16
- **Size**: ~600 MB
- **Current status**: ✅ ON CPU (moved after encoding)

### LoRA Weights
- **Rank**: 128, Alpha: 64
- **Size**: ~100-200 MB (merged into base model)
- **Current status**: Merged into noise_model

---

## 2. KV CACHE

### Streaming KV Cache (per layer, 40 layers)
```
Dimensions: [batch=1, seq_len=4608, heads=40, head_dim=128]
Size per tensor: 1 × 4608 × 40 × 128 × 2 bytes = 47.2 MB
K + V per layer: 94.4 MB
Total (40 layers): 3.78 GB
```

### Conditioning Cache (per layer, 40 layers)
```
Dimensions: [batch=1, cond_len=2800, heads=40, head_dim=128]
Size per tensor: 1 × 2800 × 40 × 128 × 2 bytes = 28.7 MB
K + V per layer: 57.4 MB  
Total (40 layers): 2.30 GB
```

### cond_end tracking (40 layers)
```
40 × 8 bytes = 320 bytes (negligible)
```

**KV Cache Total: ~6.08 GB**
- **Current status**: ⚠️ Partially on GPU during forward pass, offloaded to CPU between

---

## 3. CROSS-ATTENTION CACHE

### Cross-Attention KV Cache (40 layers)
```
For text context (~19 tokens typical):
Dimensions: [batch=1, context_len=19, heads=40, head_dim=128]
K: 1 × 19 × 40 × 128 × 2 = 195 KB
V: 1 × 19 × 40 × 128 × 2 = 195 KB
Total per layer: 390 KB
Total (40 layers): 15.6 MB
```

**Cross-Attn Cache Total: ~16 MB** (negligible)

---

## 4. INPUT EMBEDDINGS

### Audio Embeddings
```
Dimensions: [batch=1, audio_dim=1024, frames=48]
Size: 1 × 1024 × 48 × 2 bytes = 98.3 KB per clip
For 10 clips: ~983 KB
```

### Text Embeddings (Context)
```
Dimensions: [batch=1, seq_len=~19, dim=4096]
Size: 1 × 19 × 4096 × 2 = 156 KB
```

**Embeddings Total: ~1-2 MB** (negligible)

---

## 5. LATENT TENSORS

### Reference Image Latents
```
Dimensions: [batch=1, channels=16, frames=4, H=48, W=32]
Size: 1 × 16 × 4 × 48 × 32 × 2 = 393 KB
```

### Motion Latents  
```
Dimensions: [batch=1, channels=16, motion_frames=17, H=48, W=32]
Size: 1 × 16 × 17 × 48 × 32 × 2 = 1.67 MB
```

### Clip Noise/Latents (per clip being processed)
```
Dimensions: [channels=16, lat_frames=12, H=48, W=32]
Size: 16 × 12 × 48 × 32 × 2 = 590 KB
```

### Block Latents (during processing)
```
Dimensions: [channels=16, frames_per_block=3, H=48, W=32]  
Size: 16 × 3 × 48 × 32 × 2 = 147 KB
```

**Latent Tensors Total: ~3-4 MB** (negligible)

---

## 6. ACTIVATIONS (During Forward Pass)

### Input Embeddings to Transformer
```
Dimensions: [batch=1, seq_len=4608, dim=5120]
Size: 1 × 4608 × 5120 × 2 = 47.2 MB
```

### Attention Intermediate Buffers (per layer)
```
Q, K, V projections: [1, 4608, 40, 128] each
Size per projection: 47.2 MB
Total for Q+K+V: 141.6 MB per layer
```

### Attention Output (per layer)
```
After attention: [1, 4608, 5120]
Size: 47.2 MB
```

### FFN Intermediate (per layer)
```
FFN expands to dim=13824
Size: 1 × 4608 × 13824 × 2 = 127.5 MB per layer
```

### Peak Activations (single layer with worst case)
```
Q+K+V buffers: 141.6 MB
Attention scores: ~85 MB (heads × seq × seq)
FFN intermediate: 127.5 MB
Residuals and norms: ~100 MB
Peak per layer: ~450-500 MB
```

**With torch.cuda.empty_cache() between layers/steps, peak is dominated by single layer**

**Activations Total: ~500 MB - 1 GB**

---

## 7. VAE ENCODE/DECODE BUFFERS

### VAE Encoder Input
```
Pixel space: [1, 3, frames=17, H=384, W=256]
Size: 1 × 3 × 17 × 384 × 256 × 4 (float32) = 50.3 MB
```

### VAE Decoder Output  
```
Pixel space: [1, 3, frames=48, H=384, W=256]
Size: 1 × 3 × 48 × 384 × 256 × 4 = 142 MB
```

### VAE Internal Activations
```
VAE has multiple downsampling/upsampling layers
Peak intermediate: ~200-400 MB
```

**VAE Buffers Total: ~400-600 MB** (during encode/decode only)

---

## 8. PYTORCH OVERHEAD

### CUDA Context & Allocator
```
Base CUDA context: ~500 MB
PyTorch memory allocator overhead: ~200-500 MB
```

### Optimizer States
```
In inference mode: 0 MB (no gradients)
```

**PyTorch Overhead: ~700 MB - 1 GB**

---

## 9. GRADIENT BUFFERS (Should be Zero)

### Gradients
```
With torch.no_grad(): 0 MB ✅
```

---

# TOTAL VRAM BREAKDOWN

## Current Configuration (with aggressive offloading):

| Component | Size | Status |
|-----------|------|--------|
| **Noise Model** | 14-20 GB | ✅ CPU (loaded per-step) |
| **T5 Encoder** | 22 GB | ✅ CPU |
| **VAE** | 600 MB - 1 GB | ⚠️ GPU (during encode/decode) |
| **Audio Encoder** | 600 MB | ✅ CPU |
| **KV Cache** | 6.08 GB | ⚠️ Partially GPU (offloaded) |
| **Cross-Attn Cache** | 16 MB | GPU |
| **Input Embeddings** | 1-2 MB | GPU |
| **Latent Tensors** | 3-4 MB | GPU |
| **Activations** | 500 MB - 1 GB | GPU (during forward) |
| **VAE Buffers** | 400-600 MB | GPU (during VAE ops) |
| **PyTorch Overhead** | 700 MB - 1 GB | GPU |

---

## PEAK VRAM Scenarios:

### Scenario A: During Denoising Step (Model Loaded)
```
Noise Model:        16 GB (FP8)
KV Cache:           2 GB (working portion)
Activations:        1 GB
Cross-attn cache:   16 MB
Overhead:           1 GB
─────────────────────────
PEAK:              ~20 GB
```

### Scenario B: During VAE Decode (Model Offloaded)
```
VAE Model:          1 GB
VAE Buffers:        600 MB
Motion latents:     2 MB
Overhead:           1 GB
─────────────────────────
PEAK:              ~2.6 GB
```

### Scenario C: Transition (Both Loaded Momentarily)
```
If model not fully cleared before VAE loads:
Noise Model:        16 GB
VAE:                1 GB
Overhead:           1 GB
─────────────────────────
PEAK:              ~18 GB
```

---

## WHY YOU'RE STILL HITTING OOM:

### Most Likely Culprits:

1. **Noise Model Not Fully Cleared Between Clips**
   - Even with `.cpu()`, PyTorch may not immediately free GPU memory
   - Fragmentation accumulates across clips

2. **KV Cache Size**
   - 6 GB is substantial even when "offloaded"
   - The working portion stays on GPU during forward pass

3. **Activation Accumulation**
   - Without gradient checkpointing, all layer activations may persist
   - 40 layers × 500 MB = 20 GB if not cleared

4. **CUDA Memory Fragmentation**
   - 10 clips × 64 GPU↔CPU transfers = high fragmentation
   - PyTorch allocator may hold onto freed blocks

5. **VAE During Online Decode**
   - If `--enable_online_decode` loads VAE before model fully offloads
   - Brief overlap = OOM

---

## SOLUTIONS TO TRY:

### Immediate (Most Aggressive):

1. **Force synchronous cleanup**:
   Add after each model offload:
   ```python
   torch.cuda.synchronize()
   torch.cuda.empty_cache()
   gc.collect()
   ```

2. **Reduce KV cache size**:
   Try `--infer_frames 32` with manual cache size override (if possible)

3. **Disable FP8**:
   Remove `--fp8` - quantization can cause fragmentation

4. **Further reduce resolution**:
   Try custom size like `320*192` (may need code change)

5. **Generate clips sequentially in separate processes**:
   Run script 10 times with `--num_clip 1`, combine videos after

### Nuclear Option:

**Layer-by-layer offloading**: Move each transformer layer to GPU only when processing it, offload immediately after. Would require modifying the model forward pass itself.

---

## MONITORING COMMAND:

```bash
# Watch VRAM in real-time
watch -n 0.5 nvidia-smi

# Detailed memory stats  
nvidia-smi dmon -s mu 1

# From Python during run:
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```
