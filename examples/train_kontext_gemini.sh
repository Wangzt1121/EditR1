export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,5,6}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Optional private API-key env file. Keep secrets out of the repository script.
GEMINI_API_ENV_FILE=${GEMINI_API_ENV_FILE:-/nvmedata/workspace2/users/wzt/.gemini_api_env}
if [ -f "${GEMINI_API_ENV_FILE}" ]; then
    set -a
    . "${GEMINI_API_ENV_FILE}"
    set +a
fi

# Reward 后端选择：
#   qwen            : 原始本地 Qwen / mllm_score_continue reward server
#   relay_gemini    : 中转站 + OpenAI 兼容接口 + gemini-2.5-flash
#   official_gemini : Gemini 官方 OpenAI 兼容接口
#   official_gemini_native : Gemini 官方 generateContent 原生接口 + responseJsonSchema
export REWARD_BACKEND=${REWARD_BACKEND:-official_gemini_native}
case "${REWARD_BACKEND}" in
    1|qwen|original|original_qwen|mllm_score_continue)
        CONFIG_NAME_DEFAULT=kontext_mllm_reward
        export USE_PROXY=${USE_PROXY:-0}
        export REWARD_SERVER=${REWARD_SERVER:-127.0.0.1:12341}
        ;;
    2|relay|relay_gemini|gemini_relay|openai_compatible)
        CONFIG_NAME_DEFAULT=kontext_single_api_reward
        export USE_PROXY=${USE_PROXY:-0}
        export GPT_REWARD_BASE_URL=${GPT_REWARD_BASE_URL:-https://grsai.dakka.com.cn/v1}
        export SINGLE_REWARD_API_KEY=${SINGLE_REWARD_API_KEY:-${OPENAI_API_KEY:-}}
        export GEMINI_REWARD_MODEL=${GEMINI_REWARD_MODEL:-gemini-2.5-flash}
        ;;
    3|official|official_gemini|gemini_official)
        CONFIG_NAME_DEFAULT=kontext_single_api_reward
        export USE_PROXY=${USE_PROXY:-1}
        export GPT_REWARD_BASE_URL=${GPT_REWARD_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai}
        export SINGLE_REWARD_API_KEY=${SINGLE_REWARD_API_KEY:-${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}}
        export GEMINI_REWARD_MODEL=${GEMINI_REWARD_MODEL:-gemini-3-flash}
        ;;
    official_native|official_gemini_native|gemini_native|native_gemini)
        CONFIG_NAME_DEFAULT=kontext_single_api_reward
        export USE_PROXY=${USE_PROXY:-1}
        export GEMINI_API_BASE_URL=${GEMINI_API_BASE_URL:-https://generativelanguage.googleapis.com/v1beta}
        export GPT_REWARD_BASE_URL=${GPT_REWARD_BASE_URL:-${GEMINI_API_BASE_URL}}
        export SINGLE_REWARD_BASE_URL=${GEMINI_API_BASE_URL}
        export SINGLE_REWARD_API_KEY=${SINGLE_REWARD_API_KEY:-${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}}
        export GEMINI_REWARD_MODEL=${GEMINI_REWARD_MODEL:-gemini-3.1-pro-preview}
        export SINGLE_REWARD_NATIVE_GEMINI=1
        export GEMINI_NATIVE_API=1
        ;;
    *)
        echo "ERROR: Unknown REWARD_BACKEND=${REWARD_BACKEND}. Use qwen, relay_gemini, official_gemini, or official_gemini_native." >&2
        exit 1
        ;;
esac

# Gemini native API/schema with the repository-local test_gemini.py Original reward flow.
if [ "${SINGLE_REWARD_NATIVE_GEMINI:-0}" = "1" ]; then
    export SINGLE_REWARD_HELPER_PATH=${PROJECT_ROOT}/reward_server/test_gemini_reward.py
fi

# 中转站或 Gemini 官方模式可按需使用代理；Qwen 本地 reward 默认关闭代理。
if [ "${USE_PROXY}" = "1" ] && [ -f /nvmedata/workspace2/users/wzt/paofu_mihomo/proxy_env.sh ]; then
    source /nvmedata/workspace2/users/wzt/paofu_mihomo/proxy_env.sh
    export HTTP_PROXY=${HTTP_PROXY:-${http_proxy:-}}
    export HTTPS_PROXY=${HTTPS_PROXY:-${https_proxy:-}}
    export ALL_PROXY=${ALL_PROXY:-${all_proxy:-}}
else
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
fi
if [ "${CONFIG_NAME_DEFAULT}" != "kontext_mllm_reward" ]; then
    export OPENAI_BASE_URL=${OPENAI_BASE_URL:-${GPT_REWARD_BASE_URL}}
    export GEMINI_OPENAI_BASE_URL=${GEMINI_OPENAI_BASE_URL:-${GPT_REWARD_BASE_URL}}
    export OPENAI_API_KEY=${OPENAI_API_KEY:-${SINGLE_REWARD_API_KEY}}
    export GEMINI_OPENAI_PLAIN_JSON=${GEMINI_OPENAI_PLAIN_JSON:-1}
    if [ -z "${OPENAI_API_KEY}" ]; then
        echo "ERROR: Missing API key for ${REWARD_BACKEND}. Export SINGLE_REWARD_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY." >&2
        exit 1
    fi
fi
# 运行环境：
#   PYTHONPATH: 把当前仓库放到 Python 搜索路径最前面，确保读取本地 config/flow_grpo 模块。
#   TOKENIZERS_PARALLELISM: 允许 HuggingFace tokenizer 使用并行 worker。
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=true

# 本地 Qwen reward server：
#   REWARD_SERVER: 原始 mllm_score_continue reward 的 host:port，仅 REWARD_BACKEND=qwen 时使用。
export REWARD_SERVER=${REWARD_SERVER:-127.0.0.1:12341}

# 分布式启动参数：
#   WORLD_SIZE: 训练节点数量。
#   MASTER_ADDR / MASTER_PORT: torch 分布式 rendezvous 地址。
#   RANK: 当前节点编号。
#   NPROC_PER_NODE: 当前节点启动的 GPU 进程数。
export WORLD_SIZE=${WORLD_SIZE:-1}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29501}
export RANK=${RANK:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}

# 训练长度与 checkpoint / 日志策略：
#   NUM_EPOCHS: 总训练 epoch 数。
#   SAVE_FREQ: 每 N 个 epoch 保存一次 LoRA。
#   NUM_GROUPS_PER_EPOCH: 每个 epoch 采样多少个 source prompt group。
#   EVAL_FREQ: 每 N 个 global step/epoch 做一次固定 eval；默认 1 表示每一步都测。
#   SKIP_EVAL: 设为 1 表示训练中跳过 eval sampling。
#   MAX_EVAL_BATCHES: eval 最多跑多少个 batch；0 表示完整 eval。
#   WANDB_MODE: offline 表示 W&B 日志只保存在本地。
export NUM_EPOCHS=${NUM_EPOCHS:-100}
export SAVE_FREQ=${SAVE_FREQ:-5}
export NUM_GROUPS_PER_EPOCH=${NUM_GROUPS_PER_EPOCH:-2}
export EVAL_FREQ=${EVAL_FREQ:-1}
export SKIP_EVAL=${SKIP_EVAL:-0}
export MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-0}
export WANDB_MODE=${WANDB_MODE:-offline}

# 数据集选择：
#   DATASET_ROOT: 图像数据根目录。
#   PROMPT_METADATA_FILE: 训练 jsonl metadata 文件，包含 image、prompt、category、rubric 等字段。
#   EVAL_PROMPT_METADATA_FILE: 固定验证集 jsonl metadata 文件；为空时优先读 DATASET_ROOT/test_metadata.jsonl。
#   EDIT_R1_TASK_PREFIXES: 可选任务前缀过滤；为空表示不额外过滤。
export DATASET_ROOT=/nvmedata/workspace2/users/wzt/dataset
export PROMPT_METADATA_FILE=/nvmedata/workspace2/users/wzt/dataset/data/train_metadata.jsonl
export EVAL_PROMPT_METADATA_FILE=${EVAL_PROMPT_METADATA_FILE:-/nvmedata/workspace2/users/wzt/Prehuman/metadata/test_metadata_01_one_image.jsonl}
export EDIT_R1_TASK_PREFIXES=${EDIT_R1_TASK_PREFIXES:-01,02,03}

# 图像画布与训练分辨率：
#   EDIT_R1_PAD_TO_SQUARE: 非正方形输入在送入模型前是否 padding 成正方形。
#   EDIT_R1_CANVAS_SIZE: padding 后的画布边长。
#   TRAIN_RESOLUTION: 模型训练和采样使用的分辨率。
export EDIT_R1_PAD_TO_SQUARE=${EDIT_R1_PAD_TO_SQUARE:-1}
export EDIT_R1_CANVAS_SIZE=${EDIT_R1_CANVAS_SIZE:-1024}
export TRAIN_RESOLUTION=${TRAIN_RESOLUTION:-1024}

# Diffusion 采样参数：
#   SAMPLE_NUM_STEPS: 训练阶段采样步数。
#   SAMPLE_EVAL_NUM_STEPS: eval 阶段采样步数。
#   SAMPLE_GUIDANCE_SCALE: CFG / guidance 强度。
#   SAMPLE_NOISE_LEVEL: Kontext 采样时使用的编辑噪声强度。
#   SAMPLE_DETERMINISTIC: 0 表示随机采样；1 表示确定性采样。
#   SAMPLE_SOLVER: diffusion solver 名称。
#   EDIT_R1_EXPLICIT_CANDIDATE_SEEDS: 为同一 source 的不同 candidate 显式设置不同 seed。
#   EDIT_R1_PRINT_SAMPLE_SEEDS: 打印 candidate seed，方便调试重复采样问题。
export SAMPLE_NUM_STEPS=${SAMPLE_NUM_STEPS:-20}
export SAMPLE_EVAL_NUM_STEPS=${SAMPLE_EVAL_NUM_STEPS:-15}
export SAMPLE_GUIDANCE_SCALE=${SAMPLE_GUIDANCE_SCALE:-1.25}
export SAMPLE_NOISE_LEVEL=${SAMPLE_NOISE_LEVEL:-0.7}
export SAMPLE_DETERMINISTIC=${SAMPLE_DETERMINISTIC:-0}
export SAMPLE_SOLVER=${SAMPLE_SOLVER:-flow}
export EDIT_R1_EXPLICIT_CANDIDATE_SEEDS=${EDIT_R1_EXPLICIT_CANDIDATE_SEEDS:-1}
export EDIT_R1_PRINT_SAMPLE_SEEDS=${EDIT_R1_PRINT_SAMPLE_SEEDS:-0}

# 模型与 LoRA：
#   PRETRAINED_MODEL: 基础 FLUX.1-Kontext-dev 模型路径。
#   LORA_PATH: RL 训练开始前可选加载的初始 LoRA 路径。
export PRETRAINED_MODEL=${PRETRAINED_MODEL:-/nvmedata/workspace2/users/wzt/pretrained/FLUX.1-Kontext-dev}
export LORA_PATH=${LORA_PATH:-${TRAIN_LORA_PATH:-/nvmedata/workspace2/users/wzt/DiffSynth-Studio/checkpoint/step-30000.safetensors}}

# Relative API reward，可用于相对评分对比模式：  目前不用这个版本
#   RELATIVE_REWARD_BASE_URL / RELATIVE_REWARD_MODEL: API endpoint 与模型名。
#   RELATIVE_RUBRIC_YAML: relative 模式使用的 rubric yaml。
#   RELATIVE_NUM_CANDIDATES: 一次共同送入评分的候选图数量。
#   RELATIVE_REWARD_TIMEOUT: 请求超时时间，单位秒。
export RELATIVE_REWARD_BASE_URL=${RELATIVE_REWARD_BASE_URL:-${GPT_REWARD_BASE_URL}}
export RELATIVE_REWARD_MODEL=${RELATIVE_REWARD_MODEL:-${GPT_RUBRIC_MODEL:-${GEMINI_REWARD_MODEL:-gemini-3.1-flash-lite}}}
export RELATIVE_RUBRIC_YAML=${RELATIVE_RUBRIC_YAML:-/nvmedata/workspace2/users/wzt/hair_edit.yaml}
export RELATIVE_NUM_CANDIDATES=${RELATIVE_NUM_CANDIDATES:-8}
export RELATIVE_REWARD_TIMEOUT=${RELATIVE_REWARD_TIMEOUT:-180}

# Single-image API reward：
#   SINGLE_REWARD_BASE_URL / SINGLE_REWARD_MODEL: mllm_single_api_score 使用的 API endpoint 与模型名。
#   SINGLE_REWARD_RUBRIC_YAML: 存放任务 rubric 的文件夹或文件。
#   SINGLE_NUM_CANDIDATES: 每个 source prompt 生成的候选图数量。
#   SINGLE_INITIAL_BSZ: 采样 batch size 搜索的初始值。
#   SINGLE_REWARD_TIMEOUT / MAX_RETRIES: 单次请求超时时间与最大重试次数。
#   SINGLE_REWARD_MAX_IMAGE_SIDE / JPEG_QUALITY: API 打分前的图像压缩尺寸和 JPEG 质量。
#   SINGLE_REWARD_WORKERS: 本地 reward 打分 worker 数量。
#   SINGLE_REWARD_DEADLINE_SECONDS: 整个 reward batch 的总超时时间。
#   SINGLE_REWARD_FAIL_OPEN: 1 表示失败后使用 fallback 分数继续训练；0 表示报错并停止训练。
#   EDIT_R1_REWARD_PER_IMAGE: 1 表示候选图生成后尽快逐张送入 reward。
#   SINGLE_REWARD_API_LOCK / API_CHANNELS: 控制 API 并发，避免同时请求过多。
#   SINGLE_REWARD_DEBUG / DEBUG_LIMIT: 打印解析后的 reward 细节，用于调试。
#   SINGLE_REWARD_INTEGER_RUBRIC_MODE: auto 表示新版 0-9 整数 rubric 自动走 v18/v22 scorer。
export SINGLE_REWARD_BASE_URL=${SINGLE_REWARD_BASE_URL:-${GPT_REWARD_BASE_URL}}
export SINGLE_REWARD_MODEL=${SINGLE_REWARD_MODEL:-${GPT_RUBRIC_MODEL:-${GEMINI_REWARD_MODEL:-gemini-3.1-flash-lite}}}
export SINGLE_REWARD_RUBRIC_YAML=${SINGLE_REWARD_RUBRIC_YAML:-/nvmedata/workspace2/users/wzt/Prehuman/rubrics}
export SINGLE_REWARD_INTEGER_RUBRIC_MODE=${SINGLE_REWARD_INTEGER_RUBRIC_MODE:-auto}
export SINGLE_NUM_CANDIDATES=${SINGLE_NUM_CANDIDATES:-8}
export SINGLE_INITIAL_BSZ=${SINGLE_INITIAL_BSZ:-$([ "${NPROC_PER_NODE}" = "1" ] && echo 8 || echo 2)}
export SINGLE_REWARD_TIMEOUT=${SINGLE_REWARD_TIMEOUT:-240}
export SINGLE_REWARD_MAX_RETRIES=${SINGLE_REWARD_MAX_RETRIES:-8}
export SINGLE_REWARD_MAX_IMAGE_SIDE=${SINGLE_REWARD_MAX_IMAGE_SIDE:-512}
export SINGLE_REWARD_JPEG_QUALITY=${SINGLE_REWARD_JPEG_QUALITY:-90}
export SINGLE_REWARD_WORKERS=${SINGLE_REWARD_WORKERS:-8}
export SINGLE_REWARD_DEADLINE_SECONDS=${SINGLE_REWARD_DEADLINE_SECONDS:-600}
export SINGLE_REWARD_FAIL_OPEN=${SINGLE_REWARD_FAIL_OPEN:-0}
export EDIT_R1_REWARD_PER_IMAGE=${EDIT_R1_REWARD_PER_IMAGE:-1}
export SINGLE_REWARD_API_LOCK=${SINGLE_REWARD_API_LOCK:-1}
export SINGLE_REWARD_API_CHANNELS=${SINGLE_REWARD_API_CHANNELS:-8}
export SINGLE_REWARD_DEBUG=${SINGLE_REWARD_DEBUG:-0}
export SINGLE_REWARD_DEBUG_LIMIT=${SINGLE_REWARD_DEBUG_LIMIT:-0}

# NCCL 多卡 / 多节点通信参数：
#   这些是当前集群相关的通信参数，通常不需要修改。
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_NET_PLUGIN=none

# 启动配置：
#   CONFIG_NAME: config/kontext_nft.py 里的配置函数名，默认跟随 REWARD_BACKEND。
#   TORCHRUN_BIN: torchrun 可执行文件路径。
CONFIG_NAME=${1:-${CONFIG_NAME:-${CONFIG_NAME_DEFAULT}}}
TORCHRUN_BIN=${TORCHRUN_BIN:-/nvmedata/workspace2/conda_data/envs/Edit-R1/bin/torchrun}

# 启动摘要：
#   打印当前选择的 reward 路径，但不暴露 API key。
if [ "${CONFIG_NAME_DEFAULT}" = "kontext_mllm_reward" ]; then
    echo "[train_kontext] reward_backend=${REWARD_BACKEND} config=${CONFIG_NAME} reward=mllm_score_continue server=${REWARD_SERVER}"
else
    echo "[train_kontext] reward_backend=${REWARD_BACKEND} config=${CONFIG_NAME} tasks=${EDIT_R1_TASK_PREFIXES:-all} reward=mllm_single_api_score model=${SINGLE_REWARD_MODEL} base_url=${SINGLE_REWARD_BASE_URL} native_gemini=${SINGLE_REWARD_NATIVE_GEMINI:-0} reward_logic=integer_${SINGLE_REWARD_INTEGER_RUBRIC_MODE} rubric=${SINGLE_REWARD_RUBRIC_YAML} helper=${SINGLE_REWARD_HELPER_PATH:-default}"
fi

${TORCHRUN_BIN} --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${WORLD_SIZE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --node_rank ${RANK} \
    scripts/train_nft_kontext.py --config config/kontext_nft.py:${CONFIG_NAME}
