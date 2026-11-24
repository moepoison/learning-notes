# 学习计划：mini-sglang

> 换新 session 时，跟 Claude 说「读一下 `~/Playground/learning-notes/PLAN.md`，继续学」就能接上。
> 每次学完，更新「当前进度」和「待回答的问题」。

## 目标

读懂一个 LLM 推理服务从收到请求到流式返回的全过程，重点是调度、KV cache 和 GPU 执行。

- 代码：<https://github.com/sgl-project/mini-sglang>，本地 `~/Playground/llm_serving/llm/mini-sglang`（commit `9a91cfa`）
- 环境：Mac，没有 NVIDIA GPU，跑不了推理。靠读代码学习，纯 Python 的部分可以写小脚本模拟
- 笔记：<https://moepoison.github.io/learning-notes/>

## 学习方法：费曼学习法

1. **我先读代码**，不先听讲解
2. **我用大白话讲给 Claude 听**，假装它是外行，不许用术语
3. **Claude 扮演新手追问**，问「为什么」「如果……会怎样」，卡住的地方就是知识缺口
4. **回去补缺口，再讲一遍**，最后用一句话或一个比喻总结
5. 学完一块，整理成笔记推到 repo，首页打卡

想被狠狠挑战时，说「用费曼的视角看」，会触发 `feynman-perspective` skill。

## 路线图

### 第一阶段：全局（已完成）
- [x] 请求在各进程之间怎么流转 → [请求流程图](mini-sglang/request-map.html)
- [x] API server 的 async 和 ZMQ 消息 → [异步与消息](mini-sglang/async-and-messaging.html)

### 第二阶段：Scheduler（进行中）
- [x] `core.py` 的 `Req`：`cached_len` / `device_len` / `extend_len` / `remain_len` → [从 Req 到下一个 token](mini-sglang/req-and-generation.html)
- [x] `scheduler.py` 的 `normal_loop` 主循环骨架
- [x] overlap 的底层机制：CUDA stream、`wait_stream`、`pin_memory`、Event → [Overlap 笔记](mini-sglang/overlap-mechanics.html)
- [x] `table.py`：`table_idx` 是请求在 `page_table` 里的座位号
- [ ] **`decode.py`：`filter_reqs`、按 uid 排序、`inflight_tokens`** ← 现在在这里
- [ ] `prefill.py`：新请求的准入条件、chunked prefill、`reserved_size`
- [ ] `cache.py`：KV 页怎么分配和回收、`cache_req`
- [ ] `scheduler.py` 的 `overlap_loop`：为什么 `finished_reqs` 能防止重复释放

### 第三阶段：KV cache
- [ ] `kvcache/radix_cache.py`：前缀树怎么匹配、插入、淘汰
- [ ] 两个请求共享 system prompt 时，第二个请求省掉了哪些计算

### 第四阶段：Engine 与模型
- [x] token 生成链：logits → LM head → sampler（已在笔记里）
- [x] Llama 怎么接进 mini-sglang：注册表、meta device、全局 Context（已在笔记里）
- [ ] `engine/graph.py`：CUDA Graph，为什么 decode 能用、prefill 不行
- [ ] `attention/fi.py`：注意力后端怎么读写分页 KV cache
- [ ] `layers/linear.py` + `distributed/`：张量并行，列切分和行切分，all-reduce 在哪

## 当前进度

**2026-11-15**：读 `scheduler/decode.py` 读到一半。用大白话讲了 DecodeManager 是管 decode 阶段请求的，`running_reqs` 装的是 `Req`。卡在 `filter_reqs` 为什么要 filter。

**下次从这里开始**，先回答：

1. `filter_reqs` 那一行里，`self.running_reqs.union(reqs)` 把什么和什么合并？如果传进来的是刚跑完的 prefill 批次，会发生什么？
2. `if req.can_decode` 留下谁、扔掉谁？没有它会怎样？
3. 一句大白话：`filter_reqs` 每一轮在干什么？
4. `schedule_next_batch` 为什么按 `uid` 排序？（提示：commit「Stabilize decode batch request order across TP ranks」）
5. `inflight_tokens` 在算什么，给谁用？

## 待回答的问题

费曼检验（用大白话讲）：
- [ ] 不用 prefill / decode / token 这些词，给不懂 AI 的朋友解释 ChatGPT 是怎么回答问题的
- [ ] prefill 和 decode 生成 token 用的是同一条链，两者唯一的区别是什么
- [ ] 为什么 mini-sglang 要自己重写 Llama，不用 HuggingFace 现成的

读到相关代码时再回答：
- [ ] 删掉 `wait_stream` 会出什么错（用厨房比喻说）
- [ ] overlap 模式下，生成了 EOS 的请求会不会被错误放进下一批？代码怎么防
- [ ] 在 `core.py` 的 `Req` 注释里补上 `table index：` 那一行

## 学过但还没进笔记的

- `dataclass`、`__post_init__`、`eq=False`；Python 3.10 语法（`X | None`、海象运算符 `:=`）
