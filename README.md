# PRML Project 1

Pattern Recognition and Machine Learning - Project 1 (Spring 2026)

## 环境配置

本仓库需要 Python 3.12+、CUDA 12.6、PyTorch 和 Marimo。

请根据你的环境选择以下任一方式安装：

### 方式 1: Conda（推荐）

```bash
conda env create -f environment.yml
conda activate prml-project1
```

### 方式 2: Nix + direnv

需要 [Nix](https://nixos.org/download/) 和 [direnv](https://direnv.net/)：

```bash
git clone <repo-url> && cd project1
direnv allow
```

进入目录后自动创建 `.venv` 并通过 `uv sync` 安装依赖。

### 方式 3: uv

需要 [uv](https://docs.astral.sh/uv/)：

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

### 方式 4: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 验证环境

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import marimo; print(f'Marimo {marimo.__version__}')"
```

预期输出：

```
PyTorch 2.11.0+cu126, CUDA: True
Marimo 0.23.2
```

## 启动 Marimo

```bash
# 交互式 notebook 编辑器
marimo edit

# 编辑特定文件
marimo edit notebook.py

# 作为 Web 应用运行
marimo run app.py
```

## 项目结构

```
.
├── pyproject.toml        # Python 依赖声明（uv/pip 使用）
├── environment.yml       # Conda 环境定义
├── requirements.txt      # 锁定版本依赖（pip 使用，自动生成）
├── uv.lock               # uv 依赖锁定文件
├── flake.nix             # Nix 开发环境
├── .envrc                # direnv 配置
└── data/                 # 数据目录（已 gitignore）
```

## 包含的主要依赖

| 包             | 用途                      |
| -------------- | ------------------------- |
| PyTorch + CUDA | 深度学习框架              |
| numpy          | 数值计算                  |
| matplotlib     | 可视化                    |
| marimo         | 响应式 Python notebook    |
| duckdb         | SQL 查询引擎              |
| polars         | 高性能 DataFrame          |
| altair         | 声明式统计可视化          |
| openai         | AI 辅助（marimo 内置支持）|
