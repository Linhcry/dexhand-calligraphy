# 五指灵巧操作仿真

本工程在 Windows 上运行五指灵巧手艺术技能场景，完成题面识别、抓笔、蘸墨、汉字轨迹生成、机械臂执行和碰撞审计。仿真模型采用 DexRobot 官方仓库中的 DexHand021 右手与 JAKA Zu7 组合模型，安装脚本会下载并锁定指定版本。

## 从这里开始

- [学习与操作入口](docs/00_从这里开始.md)
- [2026-09-11 修订规则与项目差距](docs/01_新版规则与项目差距.md)
- [2026 初赛/决赛程序区分与运行说明](docs/06_新版初赛决赛运行说明.md)
- [VS Code 与仿真操作手册](docs/02_VSCode与仿真操作手册.md)
- [架构与文件地图](docs/03_架构与文件地图.md)
- [训练、调参与验收方法](docs/04_训练调参与验收方法.md)
- [代码协作任务模板](docs/05_代码协作任务模板.md)

当前仓库同时保留两条决赛路径：旧的单张 A3 开发基线用于回归对照；新的 `2026` 程序按 2026-09-11 修订版运行，决赛使用 4 张 A3 纸组成 2×2 阵列，初赛使用“画龙点睛”三次独立流程。新版规则参数集中在 `config/competition_20260911.json`，新程序均读取该规则文件。

## 环境准备

系统要求：

- Windows 10/11
- Python 3.12
- Git
- [uv](https://docs.astral.sh/uv/)
- FFmpeg（仅导出视频时需要）

在仓库根目录执行：

```powershell
Set-Location .\five_finger
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

`setup.ps1` 会创建 `.venv`、安装固定版本依赖、下载官方模型并检查环境。需要运行手部重定向实验时，再执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup_retarget.ps1
```

## 基础运行

检查环境：

```powershell
.\run.ps1 -Mode doctor
```

打开 MuJoCo 交互窗口：

```powershell
.\run.ps1 -Mode viewer
```

生成单字轨迹：

```powershell
.\run.ps1 -Mode plan -Text 永
```

运行无界面基础仿真：

```powershell
.\run.ps1 -Mode demo
```

## 2026 修订版：初赛与决赛

初赛和决赛是两个独立入口，共用物理抓笔/蘸墨/安全层，但任务编排、场景、计时和评分完全分开。

### 初赛“画龙点睛”（1 次开发烟测）

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe -m experiments.run_preliminary_round --attempts 1
```

正式规则为 `--attempts 3`。每次独立出题、独立 180 秒、独立评分，最后只取最好一次。结果在 `outputs/preliminary_2026/`。

### 决赛“古诗书法”（4×A3 2×2，300 秒）

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe -m experiments.run_final_round_2026 `
  --profile config/profile_final_2026_v1.json
```

结果在 `outputs/final_2026/`。`profile_final_2026_v1.json` 只把已验证的旧决赛参数复用到 4 张 A3 场景，并按新规则保留了 300 秒预算；不要与旧单张 A3 基线互相覆盖。

### 快速烟测

初赛和决赛都支持少次数或已知题目的开发模式：

```powershell
# 初赛只跑一次
.\.venv\Scripts\python.exe -m experiments.run_preliminary_round --attempts 1

# 决赛使用已知题目，缩短回归时间
.\.venv\Scripts\python.exe -m experiments.run_final_round_2026 `
  --development-title 静夜思 `
  --profile config/profile_final_2026_smoke.json
```

### 实时观看 MuJoCo（开发调试）

上面的赛程默认是无头仿真，不会弹出窗口，和正式比赛一致。开发调试时可加 `--viewer`：

```powershell
# 初赛：弹窗观看一次完整点睛
.\.venv\Scripts\python.exe -m experiments.run_preliminary_round --attempts 1 --viewer

# 决赛：弹窗观看完整书写（推荐先用 smoke，300 秒完整版会较长）
.\.venv\Scripts\python.exe -m experiments.run_final_round_2026 `
  --development-title 静夜思 `
  --profile config/profile_final_2026_smoke.json `
  --viewer
```

窗口关闭后仿真会自动停止；初赛/决赛的控制器日志和结果仍会正常写入各自 `outputs` 目录。

## 随机题目完整流程（旧基线）

完整流程会先随机生成题面图像，控制程序只能读取图像和公开场景参数；识别完成后执行抓笔、保持、蘸墨和全文书写，最后由独立检查程序公开答案并审计结果。

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv-retarget\Scripts\python.exe -m experiments.run_blind_round --video
```

生成结果保存在 `outputs/blind_competition/<case_id>`。加入 `--open-video` 可在导出完成后打开视频。

指定控制参数运行：

```powershell
.\.venv-retarget\Scripts\python.exe -m experiments.run_blind_round `
  --profile config/profile_pinch_reserve_v7.json `
  --video
```

## 场景扰动

运行相机、纸张、笔架和书写区域变化测试：

```powershell
.\.venv-retarget\Scripts\python.exe -m experiments.scene_matrix
```

运行多轮鲁棒性实验：

```powershell
.\.venv-retarget\Scripts\python.exe -m experiments.train_robustness `
  --profile config/profile_pinch_reserve_v7.json `
  --rounds 3
```

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 目录

| 路径 | 内容 |
|---|---|
| `fivefinger/` | 场景、控制、视觉、轨迹与自主任务模块 |
| `experiments/` | 随机题目、完整物理流程、扰动实验与审计工具 |
| `experiments/preliminary_*.py` | 2026 初赛：出题、龙眼识别、点睛控制器、三次赛程 |
| `experiments/final_*_2026.py` | 2026 决赛：4×A3 出题、控制器包装、规则评分、单轮赛程 |
| `config/competition_20260911.json` | 2026-09-11 官方修订规则的唯一结构化来源 |
| `config/` | 场景参数、硬件接口模板与控制器参数 |
| `assets/ocr/` | 离线 OCR 权重及许可证 |
| `assets/hanzi/` | 汉字笔画数据及来源说明 |
| `tests/` | 控制、感知、碰撞与任务流程测试 |
| `lessons/` | MuJoCo 入门与调参示例 |
| `scripts/` | 环境检查、资源下载、录制和来源核验工具 |

## 第三方资源

- [DexRobot MuJoCo models](https://github.com/DexRobot/dexrobot_mujoco)，Apache-2.0
- [MuJoCo](https://github.com/google-deepmind/mujoco)，Apache-2.0
- [RapidOCR](https://github.com/RapidAI/RapidOCR)，Apache-2.0
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)，Apache-2.0
- [Hanzi Writer Data](https://github.com/chanind/hanzi-writer-data)，ARPHIC Public License
- [dex-retargeting](https://github.com/dexsuite/dex-retargeting)，Apache-2.0

各资源的版本和来源记录在 `config/model_sources.json`、`assets/ocr/sources.json`、`assets/hanzi/sources.json` 与 `fivefinger/third_party/provenance.json`。

## 当前边界

仓库中的碰撞、力传感、毛笔和墨迹均为仿真模型。控制器尚未接入比赛现场的相机内外参、机械臂驱动、灵巧手电机映射、力传感单位和急停接口。取得正式 SDK 后，应先完成坐标标定、低速空载验证、单动作验证和限位检查，再进行完整任务测试。
