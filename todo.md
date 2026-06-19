# LBA Todo

当前没有必须继续实现的事项。已经拍板的设计和 benchmark 结论放在 `docs/`：

- `docs/design.md`
- `docs/usage.md`
- `docs/edge_cases.md`
- `docs/benchmark_145.md`

## 已完成

- 包名和 import namespace 使用 `lba`。
- 公共 API 使用 `LengthBatchingAdapter`，短别名为 `LBA`。
- 长度函数参数使用 `len_fn`。
- 默认 `max_padding_ratio=0.05`，偏向控制 padding。
- `prefetch_batches` 默认关闭，设置为正数时用后台线程预取 batch。
- 第一阶段性能目标按 CPU producer 不低于 `5 it/s` 评估。

## 后续可选

- 用真实训练脚本验证 `max_padding_ratio=0.05` 和 `prefetch_batches=4`。
- 如果真实训练中 producer 仍然跟不上 GPU，再考虑进程版 producer。
- 如果需要 per-worker planner，先设计明确的 worker flush 协议。
