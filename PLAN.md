# 学习计划：mini-sglang

> 换新 session 时，跟 Claude 说「读一下 `~/Playground/learning-notes/PLAN.md`，继续学」就能接上。
> 每次学完，更新「当前进度」和「待回答的问题」。

## 目标

读懂一个 LLM 推理服务从收到请求到流式返回的全过程，重点是调度、KV cache 和 GPU 执行。

**终点（2026-11-23 定）：读 SGLang 本体的代码，给 SGLang 贡献代码。**
中间关卡：自己手写一个 mini-sglang，写完能加载真实模型生成文字。

**原则：先懂再测。** 读代码阶段不跑代码；搞懂了再去跑，跑是为了验证理解。

**四个阶段：**
1. **读懂 mini-sglang**（纯读代码，费曼学习法）。读的时候顺带标注这块在 SGLang 本体里对应哪里
2. **手写 mini-sglang**，在 RunPod 上跑，mini-sglang 当参考答案。每一步先想清楚应该输出什么，再跑出来对答案。结构和命名尽量贴近 SGLang：
   1. 加载 HF 权重，手写 Llama forward，逐字生成，输出和 HuggingFace 完全一致
   2. 加 KV cache，拆成 prefill / decode
   3. 多请求：Req、PrefillManager、DecodeManager、主循环
   4. 分页 KV cache + `page_table`，接 FlashInfer
   5. Radix 前缀缓存
   6. 多进程 + ZMQ + API server 流式输出
   7. CUDA Graph + overlap
   8. （可选，两张卡）张量并行
3. **读 SGLang 本体**，带着手写时的理解去认代码
4. **第一个贡献**：先挑小 issue 或文档改动，熟悉流程，再碰核心代码

下面的路线图是阶段一的细分。

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
- [ ] **`server/launch.py`：几个进程、怎么互相确认准备好** ← 现在在这里。scheduler 循环已读完 → [启动与 GPU 分工](mini-sglang/launch-and-gpu.html)

### 第二阶段：Scheduler（进行中）
- [x] `core.py` 的 `Req`：`cached_len` / `device_len` / `extend_len` / `remain_len` → [从 Req 到下一个 token](mini-sglang/req-and-generation.html)
- [x] `scheduler.py` 的 `normal_loop` 主循环骨架
- [x] overlap 的底层机制：CUDA stream、`wait_stream`、`pin_memory`、Event → [Overlap 笔记](mini-sglang/overlap-mechanics.html)
- [x] `table.py`：`table_idx` 是请求在 `page_table` 里的座位号
- [ ] `decode.py`：`filter_reqs`、按 uid 排序、`inflight_tokens`（读了一半，题目见下）
- [ ] `prefill.py`：新请求的准入条件、chunked prefill、`reserved_size`
- [ ] `cache.py`：KV 页怎么分配和回收、`cache_req`
- [ ] `scheduler.py` 的 `overlap_loop`：为什么 `finished_reqs` 能防止重复释放

### 第三阶段：KV cache
- [ ] `kvcache/radix_cache.py`：前缀树怎么匹配、插入、淘汰
- [ ] 两个请求共享 system prompt 时，第二个请求省掉了哪些计算

### 第四阶段：Engine 与模型
- [x] token 生成链：logits → LM head → sampler（已在笔记里）
- [x] Llama 怎么接进 mini-sglang：注册表、meta device、全局 Context（已在笔记里）
- [ ] `engine/graph.py`：CUDA Graph。已懂：录的是什么、prefill 为什么不行、pad + dummy_req（[笔记 07 节](mini-sglang/launch-and-gpu.html#s07)）；待答：KV 长度变化为什么不影响回放
- [ ] `attention/fi.py`：注意力后端怎么读写分页 KV cache
- [ ] `layers/linear.py` + `distributed/`：张量并行，列切分和行切分，all-reduce 在哪

## 当前进度

**2026-11-24**：按请求流程图第 1 课读 `server/launch.py`。读完了 scheduler 那段循环：world_size / rank、`replace` + frozen、`_run_scheduler` 四步。顺带理清了 CPU 和 GPU 的分工（Python 全在 CPU，权重 / KV cache / token_pool 启动时就上 GPU，每轮只搬索引和 next_tokens），以及 overlap 是 CPU 和 GPU 重叠。还聊了 CUDA Graph：录的是命令清单、prefill 长度不固定所以不能录、decode 用 pad + dummy_req 凑齐 batch size。

**下次从 `launch.py` 第 70 行往下开始**，先回答：

1. 默认参数（`--tp-size 1`、`--num-tokenizer 0`）下一共几个进程？分别叫什么？
2. 主进程怎么确认大家都准备好了？为什么等 `num_tokenizers + 2` 个 ack？为什么只有 rank 0 报到？
3. `sync_all_ranks()` 删掉会怎样？
4. `start_subprocess` 为什么交给 `run_api_server` 去调？
5. `next_tokens_cpu` 回到 CPU 后拿去做哪两件事？
6. KV cache 启动时划多大？多了少了各会怎样？

**之前挂着的 `decode.py` 题目**（读到 decode 时再答）：
- `filter_reqs` 里 `self.running_reqs.union(reqs)` 把什么和什么合并？传进来的是刚跑完的 prefill 批次会怎样？
- `if req.can_decode` 留下谁、扔掉谁？
- `schedule_next_batch` 为什么按 `uid` 排序？（提示：commit「Stabilize decode batch request order across TP ranks」）
- `inflight_tokens` 在算什么，给谁用？

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
