# Q-Former Projector for LLAVA

Added Q-Former as an alternative to MLP projectors. Q-Former uses learnable query tokens with cross-attention to project vision features.

## Quick Usage

Set these in your config:
```
mm_projector_type: qformer_blip2  # or qformer_instructblip
mm_qformer_num_query_tokens: 64   # default, can be 32/64/128
mm_qformer_pretrained_path: None  # optional, use Salesforce/blip2-opt-2.7b
```

## Available Types

- `qformer` - basic Q-Former (same as qformer_blip2)
- `qformer_blip2` - BLIP-2 style 
- `qformer_instructblip` - InstructBLIP style

## Key Differences from MLP

- More parameters (~180M vs ~8M)
- Slower but more expressive
- Reduces visual tokens (256 patches -> 64 queries)
- Can load pretrained weights from BLIP-2/InstructBLIP

## Training Script Example

```bash
python -m llava.train.train \
    --mm_projector_type qformer_blip2 \
    --mm_qformer_num_query_tokens 64 \
    --mm_qformer_pretrained_path Salesforce/blip2-opt-2.7b \
    --mm_projector_lr 2e-4 \
    # ... other args
```

## Notes

- Use higher learning rate for Q-Former (2e-4 vs 2e-5)
- More memory intensive, start with smaller batch sizes
- Random init if no pretrained path specified
- Works with existing LLAVA training pipeline

## Common Issues

- OOM errors: reduce num_query_tokens or batch size
- Slow training: expected, Q-Former is heavier than MLP
- Import errors: make sure transformers is updated 