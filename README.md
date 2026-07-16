# Prehuman Edit-R1 使用说明

这是一个面向人像编辑任务的 Edit-R1 改造版本。当前版本保留 `FLUX.1-Kontext` 的训练主线，使用 Gemini API 按任务 YAML rubric 给生成结果打分，并把打分结果作为 reward 参与训练。

最重要的运行入口是：

```bash
examples/train_kontext_gemini.sh
```

这个仓库上传的是轻量版：包含代码、50 个任务的 rubric、必要 metadata、以及每个任务 1 张测试图。仓库不包含完整训练集、模型权重、LoRA checkpoint、训练日志或生成结果。

## 仓库结构

```text
.
├── examples/
│   └── train_kontext_gemini.sh          # 主训练脚本
├── scripts/
│   └── train_nft_kontext.py             # Kontext 训练主程序
├── config/
│   ├── base.py
│   └── kontext_nft.py                   # 训练配置
├── flow_grpo/                           # 采样、训练、reward 接入相关代码
├── reward_server/
│   ├── test_gemini.py                   # rubric 解析、打分聚合、failure propagation
│   ├── test_gemini_reward.py            # 训练时调用 Gemini reward 的入口
│   └── gemini_schema_v2_rejudge*.py     # Gemini 结构化评分辅助代码
├── rubrics/                             # 50 个任务的 YAML 打分规则
├── metadata/
│   ├── test_metadata_50_one_each.jsonl  # 50 个任务各 1 张图的测试 metadata
│   └── test_metadata_*.jsonl            # 其他小规模固定测试 metadata
└── data/test/                           # 50 张测试图，每个任务一张
```

## 需要你自己准备什么

仓库里没有放大文件。运行前需要自己准备：

1. Python / PyTorch 训练环境。
2. `FLUX.1-Kontext-dev` 基模型权重。
3. 可选的初始 LoRA 权重，例如 `step-30000.safetensors`。
4. 完整训练数据集和 `train_metadata.jsonl`。
5. Gemini API key，或者 OpenAI-compatible 中转站 API key。

仓库内只带了一个 50 张图的小测试集：

```text
data/test/*/000001.*
metadata/test_metadata_50_one_each.jsonl
```

它用于快速检查 eval/reward 流程是否能跑通，不是完整训练集。

## 安装环境

推荐 Python 3.10：

```bash
conda create -n Edit-R1 python=3.10 -y
conda activate Edit-R1

pip install -e .
pip install -r reward_server/requirements.txt
```

PyTorch 请按机器 CUDA 版本单独安装。

## 配置 Gemini API Key

不要把 API key 写进代码或提交到 GitHub。推荐新建一个私有环境文件：

```bash
cat > ~/.gemini_api_env <<'EOF'
export GEMINI_API_KEY="你的 Gemini API Key"
EOF

chmod 600 ~/.gemini_api_env
```

运行训练前指定这个文件：

```bash
export GEMINI_API_ENV_FILE=$HOME/.gemini_api_env
```

如果使用中转站，而不是 Gemini 官方 API，可以写：

```bash
export SINGLE_REWARD_API_KEY="你的中转站 Key"
export GPT_REWARD_BASE_URL="https://你的中转站/v1"
```

## 数据格式

训练 metadata 是 JSONL，每行一个样本。至少需要这些字段：

```json
{
  "image": "data/train/01_头发编辑/000001.png",
  "prompt": "Please edit the hair ...",
  "category": "01_hair_edit",
  "zh_category": "头发编辑",
  "rubric_key": "01_hair_edit"
}
```

其中 `image` 是相对路径，会相对于 `DATASET_ROOT` 读取。

推荐的数据集结构：

```text
YOUR_DATASET_ROOT/
└── data/
    ├── train/
    │   ├── 01_头发编辑/
    │   ├── 02_胡须/
    │   └── ...
    ├── test/
    │   ├── 01_头发编辑/
    │   ├── 02_胡须/
    │   └── ...
    ├── train_metadata.jsonl
    └── test_metadata.jsonl
```

对应环境变量：

```bash
export DATASET_ROOT=/path/to/YOUR_DATASET_ROOT
export PROMPT_METADATA_FILE=/path/to/YOUR_DATASET_ROOT/data/train_metadata.jsonl
```

如果要用仓库自带的 50 张测试图做固定 eval，需要保证 `DATASET_ROOT/data/test/...` 能找到这些图片。最简单的方式是直接在仓库根目录下做 eval：

```bash
export DATASET_ROOT=$PWD
export EVAL_PROMPT_METADATA_FILE=$PWD/metadata/test_metadata_50_one_each.jsonl
```

如果训练集在外部数据盘，而 eval 想用仓库里的 50 张图，可以把 `data/test` 复制或软链接到你的 `DATASET_ROOT/data/test`。

## 模型权重路径

基模型：

```bash
export PRETRAINED_MODEL=/path/to/FLUX.1-Kontext-dev
```

初始 LoRA：

```bash
export LORA_PATH=/path/to/step-30000.safetensors
```

如果你不想载入初始 LoRA，可以尝试：

```bash
export LORA_PATH=
```

是否允许空 LoRA 取决于当前 `config/kontext_nft.py` 里的训练配置。

## 最小训练命令

下面是一个 4 卡训练例子：训练 `01,02,03` 三个任务，每个 source prompt 采样 8 张候选图，采样步数 20，CFG 为 1.25。

```bash
cd /path/to/EditR1
conda activate Edit-R1

export GEMINI_API_ENV_FILE=$HOME/.gemini_api_env
export REWARD_BACKEND=official_gemini_native
export GEMINI_REWARD_MODEL=gemini-3.5-flash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4

export DATASET_ROOT=/path/to/YOUR_DATASET_ROOT
export PROMPT_METADATA_FILE=/path/to/YOUR_DATASET_ROOT/data/train_metadata.jsonl
export EVAL_PROMPT_METADATA_FILE=/path/to/YOUR_DATASET_ROOT/data/test_metadata.jsonl

export PRETRAINED_MODEL=/path/to/FLUX.1-Kontext-dev
export LORA_PATH=/path/to/step-30000.safetensors

export EDIT_R1_TASK_PREFIXES=01,02,03
export NUM_EPOCHS=10
export NUM_GROUPS_PER_EPOCH=2
export SINGLE_NUM_CANDIDATES=8
export SINGLE_INITIAL_BSZ=2
export SAMPLE_NUM_STEPS=20
export SAMPLE_GUIDANCE_SCALE=1.25
export EVAL_FREQ=1

bash examples/train_kontext_gemini.sh
```

启动后会打印类似信息：

```text
[train_kontext] reward_backend=official_gemini_native config=kontext_single_api_reward tasks=01,02,03 reward=mllm_single_api_score model=gemini-3.5-flash ...
```

这行信息很重要，用来确认：

- 当前 reward backend 是不是 Gemini。
- 当前 reward model 是不是你想用的模型。
- 当前训练任务是不是 `01,02,03`。
- 当前 rubric 目录是不是正确。

## 当前脚本默认值

`examples/train_kontext_gemini.sh` 里有一些默认路径是作者机器上的路径，例如：

```bash
/nvmedata/workspace2/users/wzt/dataset
/nvmedata/workspace2/users/wzt/pretrained/FLUX.1-Kontext-dev
/nvmedata/workspace2/users/wzt/DiffSynth-Studio/checkpoint/step-30000.safetensors
```

别人使用时不需要改脚本，直接在命令行用环境变量覆盖即可：

```bash
export DATASET_ROOT=/your/dataset/root
export PROMPT_METADATA_FILE=/your/dataset/root/data/train_metadata.jsonl
export PRETRAINED_MODEL=/your/model/FLUX.1-Kontext-dev
export LORA_PATH=/your/lora/step-30000.safetensors
```

## Reward 后端选择

推荐使用 Gemini 官方原生接口：

```bash
export REWARD_BACKEND=official_gemini_native
export GEMINI_REWARD_MODEL=gemini-3.5-flash
```

这个模式会走 Gemini native API，并使用结构化 JSON 返回 score vector。

如果使用 Gemini 的 OpenAI-compatible 接口：

```bash
export REWARD_BACKEND=official_gemini
```

如果使用中转站：

```bash
export REWARD_BACKEND=relay_gemini
export GPT_REWARD_BASE_URL=https://your-relay.example/v1
export SINGLE_REWARD_API_KEY=YOUR_RELAY_KEY
export GEMINI_REWARD_MODEL=gemini-2.5-flash
```

如果使用原始 Qwen/vLLM 本地 reward server：

```bash
export REWARD_BACKEND=qwen
export REWARD_SERVER=127.0.0.1:12341
```

当前仓库主要维护的是 Gemini reward 路径，Qwen 路径不是推荐默认路径。

## Rubric 逻辑

所有任务的打分规则在：

```text
rubrics/
```

训练时默认读取：

```bash
export SINGLE_REWARD_RUBRIC_YAML=$PWD/rubrics
```

reward 代码会根据 metadata 里的 `category` / `rubric_key` 自动匹配 YAML。例如：

```text
01_hair_edit  -> rubrics/01_头发编辑.yaml
02_beard      -> rubrics/02_胡须.yaml
03_lipstick   -> rubrics/03_口红.yaml
```

现在的 rubric 是 0-9 整数 L3 score vector 逻辑，且包含 failure propagation。训练日志里常见字段：

```text
reward_logic=integer_v18
reward_logic=integer_auto
rubric_yaml=...
l3_scores=...
failure_tags=...
```

如果你怀疑某个任务打分不准，优先检查 `reward_details.jsonl` 里的 `rubric_yaml` 是否匹配到了正确文件。

## 选择训练哪些任务

只训练头发、胡须、口红：

```bash
export EDIT_R1_TASK_PREFIXES=01,02,03
```

训练前五个任务：

```bash
export EDIT_R1_TASK_PREFIXES=01,02,03,04,05
```

训练所有任务：

```bash
export EDIT_R1_TASK_PREFIXES=
```

注意：如果 `NUM_GROUPS_PER_EPOCH=2`，每个 epoch 只会从任务池里抽 2 个 source prompt group，不代表每轮都会覆盖所有任务。

## 常用训练参数

| 变量 | 含义 | 常用值 |
| --- | --- | --- |
| `EDIT_R1_TASK_PREFIXES` | 任务前缀过滤，逗号分隔 | `01,02,03` |
| `NUM_EPOCHS` | 训练轮数 | `10` |
| `NUM_GROUPS_PER_EPOCH` | 每轮采样多少个 source prompt group | `2` |
| `SINGLE_NUM_CANDIDATES` | 每个 source prompt 采样多少张候选图 | `8` |
| `SINGLE_INITIAL_BSZ` | 候选图采样 batch size | 多卡常用 `2` |
| `SAMPLE_NUM_STEPS` | 训练采样步数 | `20` |
| `SAMPLE_EVAL_NUM_STEPS` | eval 采样步数 | `15` 或 `20` |
| `SAMPLE_GUIDANCE_SCALE` | CFG / guidance scale | `1.25` |
| `SINGLE_REWARD_WORKERS` | 本地 reward worker 数 | `8` |
| `SINGLE_REWARD_API_CHANNELS` | API 并发通道数 | `8` |
| `SINGLE_REWARD_FAIL_OPEN` | API 失败后是否用 fallback 分数继续 | 推荐 `0` |
| `MAX_EVAL_BATCHES` | 限制 eval batch 数，`0` 表示完整 eval | `0` |

## 输出文件

训练输出目录由 `config/kontext_nft.py` 控制，常见结构如下：

```text
logs.../
└── nft/kontext/single_api_score_tasks_01_02_03/
    ├── checkpoints/
    ├── reward_details.jsonl
    ├── reward_metrics.jsonl
    ├── eval_metrics.jsonl
    ├── scored_candidate_grids/
    └── reward_curves/
```

重点看这些文件：

```text
reward_details.jsonl     # 每张候选图的 Gemini 打分、rubric、score vector、failure tags
reward_metrics.jsonl     # 每个 step 的 reward 统计
eval_metrics.jsonl       # 固定测试集 eval 结果
scored_candidate_grids/  # 拼图，可直接看 source、候选图和分数
reward_curves/           # reward 曲线
checkpoints/             # 保存的 LoRA checkpoint
```

这些输出都不应该提交到 GitHub。

## 使用仓库自带 50 张测试图

仓库自带 50 张测试图，每个任务 1 张：

```bash
find data/test -type f | wc -l
# 50
```

对应 metadata：

```bash
metadata/test_metadata_50_one_each.jsonl
```

如果只想快速检查 eval/reward 路径：

```bash
export DATASET_ROOT=$PWD
export EVAL_PROMPT_METADATA_FILE=$PWD/metadata/test_metadata_50_one_each.jsonl
```

如果要边训练边 eval，通常应该让 `DATASET_ROOT` 指向完整数据集，并确保完整数据集里也有 `data/test/01_.../000001.*` 这些测试图路径。

## 断点和续训

checkpoint 一般保存在：

```text
checkpoints/checkpoint-*
```

是否自动续训取决于 `config/kontext_nft.py` 和训练脚本里的配置。续训时要保证：

1. 使用同一个输出目录或明确指定 resume checkpoint。
2. 训练任务、rubric、reward model 不要无意间变掉。
3. 不要把 checkpoint 提交到 GitHub。

## 常见问题

### 1. 提示 Missing API key

检查：

```bash
echo $GEMINI_API_ENV_FILE
cat $GEMINI_API_ENV_FILE
```

环境文件里至少要有一个：

```bash
export GEMINI_API_KEY=...
export GOOGLE_API_KEY=...
export SINGLE_REWARD_API_KEY=...
```

### 2. Gemini API 连接失败

如果机器需要代理：

```bash
export USE_PROXY=1
export HTTP_PROXY=http://...
export HTTPS_PROXY=http://...
```

如果机器不需要代理：

```bash
export USE_PROXY=0
```

### 3. 分数全是 0 或者很像 fallback

查看：

```bash
tail -n 5 reward_details.jsonl
```

重点看：

```text
error
reward_parse_failure_count
reward_timeout_count
reward_fallback_score_count
```

调试时可以打开：

```bash
export SINGLE_REWARD_DEBUG=1
export SINGLE_REWARD_DEBUG_LIMIT=3
```

### 4. 用错 rubric

查看 `reward_details.jsonl` 里的：

```text
rubric_yaml
rubric_task_key
rubric_version
```

如果路径不对，检查：

```bash
export SINGLE_REWARD_RUBRIC_YAML=$PWD/rubrics
```

以及 metadata 里的：

```text
category
rubric_key
```

### 5. 采样结果重复

建议保持：

```bash
export EDIT_R1_EXPLICIT_CANDIDATE_SEEDS=1
export SAMPLE_DETERMINISTIC=0
```

### 6. 显存不够

降低这些参数：

```bash
export SINGLE_INITIAL_BSZ=1
export SINGLE_NUM_CANDIDATES=4
export TRAIN_RESOLUTION=768
```

或者改成单卡：

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
```

## 不要提交的文件

`.gitignore` 已经排除了常见大文件和运行输出。请不要提交：

```text
*.safetensors
*.ckpt
*.pth
*.pt
logs*/
outputs/
output/
launch_logs/
wandb/
```

也不要提交 API key。API key 应该只放在环境变量或私有 `.gemini_api_env` 文件里。

## 当前版本的定位

这个版本不是原始 Edit-R1 的完整复刻，而是一个面向 Prehuman 人像编辑任务的研究版本：

- 基模：`FLUX.1-Kontext-dev`
- 训练入口：`examples/train_kontext_gemini.sh`
- Reward：Gemini API + task-specific YAML rubric
- 评分方式：0-9 L3 score vector + failure propagation
- 测试集：50 个任务各 1 张图的小规模 sanity check

如果只想复现当前实验，优先看 `examples/train_kontext_gemini.sh`，不要从旧的 Qwen 或 vLLM reward server 路径开始。

## 致谢

本代码基于 Edit-R1、DiffusionNFT、Flow-GRPO 等项目改造。当前仓库主要整理了 Prehuman 任务、Kontext 训练、Gemini reward 以及 task-specific rubric 打分逻辑。

模型权重遵循各自原始模型许可证。`FLUX.1-Kontext-dev` 权重不包含在本仓库中。
