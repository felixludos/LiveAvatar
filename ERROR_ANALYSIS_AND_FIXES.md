# Shape Mismatch Error Analysis and Fixes

## Error Summary
```
RuntimeError: The expanded size of the tensor (768) must match the existing size (1152) at non-singleton dimension 1. 
Target sizes: [1, 768, 40, 128]. Tensor sizes: [1152, 40, 128]
```

This error occurs at line 317 (formerly 303) of `liveavatar/models/wan/causal_model_s2v.py` during KV cache assignment:
```python
kv_cache["k"][:, current_start:(current_start+seg_len_block)] = roped_key[:,seg_idx[0]:seg_idx[1]]
```

## Root Cause
The `roped_key` tensor returned from `causal_rope_apply()` is missing its batch dimension:
- **Expected**: `[batch=1, seq_len=768, heads=40, head_dim=128]`
- **Actual**: `[1152, 40, 128]` (missing batch dimension, shape is 3D instead of 4D)

This indicates that somewhere in the `rope_apply()` or `causal_rope_apply()` function chain, the batch dimension is being lost or incorrectly flattened.

## Contributing Factors
1. **Sequence Parallel Mode**: The code uses `pad_chunk()` which may affect tensor dimensions
2. **Tensor Reshaping**: Multiple view() and reshape() operations could inadvertently collapse dimensions
3. **Batch Dimension Loss**: The rope_apply function uses `for i, _ in enumerate(x)` which assumes `x` has a batch dimension at position 0

## Implemented Fixes

### 1. Debug Logging (for diagnosis)
Added comprehensive logging to detect tensor shapes at critical points:

**In `causal_model_s2v.py`** (line ~290):
- Prints shapes of `k`, `q`, `roped_key`, `roped_query`, and `kv_cache['k']`
- Displays `seg_idx` values for understanding the slicing operations

**In `model_s2v.py`** rope functions:
- `rope_apply()`: Logs input and output shapes
- `rope_apply_cond()`: Logs input and output shapes
- Helps trace where the batch dimension is lost

### 2. Dimensional Shape Checking (line ~295)
```python
# FIX: Ensure batch dimension exists - if roped_key is missing batch dim, add it
if roped_key.dim() == 3 and kv_cache['k'].dim() == 4:
    print(f"WARNING: roped_key missing batch dimension!")
    roped_key = roped_key.unsqueeze(0)
    if roped_query.dim() == 3:
        roped_query = roped_query.unsqueeze(0)
```

This workaround:
- Detects when `roped_key` is 3D while `kv_cache['k']` is 4D
- Automatically adds the missing batch dimension using `unsqueeze(0)`
- Applies the same fix to `roped_query` for consistency

### 3. Robustified `rope_apply_cond()`` (line ~100)
Added defensive checking when accessing freqs:
```python
freqs_i = freqs[i, :s] if hasattr(freqs, 'shape') and freqs.dim() >= 2 else freqs
```

## Testing the Fix
When you run the inference script after these changes:

1. **Check Debug Output**: Look for shape diagnostic prints to understand tensor flow
2. **Look for WARNING**: If you see "WARNING: roped_key missing batch dimension!", the workaround is being triggered
3. **Verify Success**: If inference completes without the original RuntimeError

## Long-term Solution
The permanent fix requires identifying where the batch dimension is lost in the `rope_apply()` function and ensuring tensors maintain correct dimensional structure through all operations.

## Files Modified
- `liveavatar/models/wan/causal_model_s2v.py` - Added dimension checking and unsqueeze workaround
- `liveavatar/models/wan/wan_2_2/modules/s2v/model_s2v.py` - Added debug logging to rope functions

## Next Steps
1. Run inference and check the debug output
2. Look for patterns in the logged shapes to identify the root cause
3. Once root cause is identified, implement a permanent fix in the rope_apply logic
