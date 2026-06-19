# LBA 边界情况

## 超长样本

如果单个样本长度超过最大 padded length，LBA 应将该样本单独组成 batch，并同时：

- 发出 Python warning。
- 写入 LBA 日志文件。
- 记录 sample 自己的 repr。

## 动态 Batch Size

LBA 会改变 batch size。训练代码不应假设每个 batch 的样本数固定。

第一版不实现 `__len__`，避免进度条或训练框架拿到误导性的 batch 数。

## 迭代顺序

LBA 会改变样本迭代顺序。它会尽量保证样本不丢失，但不保证原始 dataloader
的严格顺序。

## 多进程

原始 dataloader 的 worker 用于读取 raw samples，并在 source collate 中执行
`len_fn`。planner 仍在主进程中维护全局缓存。

## IterableDataset

LBA 会自动识别 `IterableDataset`，并使用原始 dataloader 的 `batch_size` 和
`drop_last` 构造内部 source loader。`batch_size=None` 的 unbatched iterable
loader 暂不支持，因为原始 `collate_fn` 通常不是面向样本列表的 batch collate。

## Collate 开销

第一版设计中，原始 `collate_fn` 在主进程调用。如果用户的 `collate_fn` 很重，
包装后可能影响吞吐。后续可以单独设计 worker-side collate 优化。
