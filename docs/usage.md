# LBA 用法设计

LBA 的第一目标是让用户尽量少改训练代码。用户先照常创建原始 PyTorch
`DataLoader`，再用 `LBA` 包一层。

## 推荐用法

```python
from lba import LBA

loader = LBA(
    dataloader,
    len_fn=lambda sample: len(sample["input_ids"]),
)

for batch in loader:
    ...
```

## 完整类名

```python
from lba import LengthBatchingAdapter

loader = LengthBatchingAdapter(dataloader, len_fn=len_fn)
```

`LBA` 是 `LengthBatchingAdapter` 的短别名，两者行为完全一致。

## 显式目标长度

用户可以直接指定最大 padded length：

```python
loader = LBA(
    dataloader,
    len_fn=len_fn,
    max_padded_length=8192,
)
```

显式设置 `max_padded_length` 时，LBA 会发出 warning，提醒用户该值会覆盖
由原始 batch size 推导目标长度的流程。

## Padding 阈值

`max_padding_ratio` 默认是 `0.05`。也就是候选 batch 的 padding ratio 低于
5% 时，planner 可以直接提交这个 batch：

```python
loader = LBA(dataloader, len_fn=len_fn, max_padding_ratio=0.05)
```

如果更重视吞吐，可以调高这个值；如果更重视 padding，则调低这个值。

## 日志路径

默认日志目录设计为：

```text
~/.lba/logs/lba-YYYYmmdd-HHMMSS-PID.log
```

用户也可以指定：

```python
loader = LBA(dataloader, len_fn=len_fn, log_dir="outputs/lba_logs")
```

## 预取

默认 `prefetch_batches=4`，LBA 会用 bounded prefetch queue 在后台提前准备
batch。也可以显式调整 queue 深度：

```python
loader = LBA(dataloader, len_fn=len_fn, prefetch_batches=4)
```

需要严格同步迭代或排查线程相关问题时，可以设置 `prefetch_batches=0` 关闭后台
producer。
