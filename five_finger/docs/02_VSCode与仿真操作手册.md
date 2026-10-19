# VS Code 与仿真操作手册

## 第一次打开项目

1. 安装 Python 3.12、Git、VS Code 和 `uv`。
2. 在 VS Code 扩展商店安装 **Python** 和 **Pylance**。
3. 打开 PowerShell，进入仓库中的 `five_finger`：

```powershell
Set-Location <LOCAL_PROJECT>
code .
```

4. 在 VS Code 终端中安装基础环境：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

5. 安装完整重定向环境：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup_retarget.ps1
```

6. 按 `Ctrl+Shift+P`，选择 `Python: Select Interpreter`，再选择：

```text
five_finger\.venv\Scripts\python.exe
```

`.vscode/settings.json` 已保存这个默认值。完整抓笔仿真会由任务配置自动改用 `.venv-retarget`。

## 用菜单运行

按 `Ctrl+Shift+P`，输入 `Tasks: Run Task`，选择：

| 任务 | 用途 |
|---|---|
| `1 环境检查` | 检查 Python、依赖、模型和资源 |
| `2 打开 MuJoCo 窗口` | 只打开交互式仿真窗口 |
| `3 随机诗题完整仿真` | 随机生成题面，识别后执行并导出视频 |
| `4 固定题目开发试验` | 用固定题目调试，减少变量 |
| `5 随机鲁棒性试验 3 轮` | 连续运行随机场景并汇总指标 |
| `6 快速测试` | 检查代码改动是否破坏已有功能 |

## 直接在终端运行

检查环境：

```powershell
.\run.ps1 -Mode doctor
```

打开 MuJoCo：

```powershell
.\run.ps1 -Mode viewer
```

在窗口中，鼠标左键拖动旋转视角，右键拖动平移，中键或滚轮缩放。窗口只用于观察，完整自主流程应从下面的脚本启动。

运行随机题目完整仿真并导出视频：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:FIVEFINGER_OCR_PYTHON = "$PWD\.venv\Scripts\python.exe"
.\.venv-retarget\Scripts\python.exe -m experiments.run_blind_round --video
```

运行三轮随机鲁棒性试验：

```powershell
.\.venv-retarget\Scripts\python.exe -m experiments.train_robustness `
  --profile config/profile_pinch_reserve_v7.json `
  --rounds 3
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 如何单步调试

1. 在想暂停的 Python 行号左侧单击，出现红点。
2. 打开左侧“运行和调试”。
3. 选择 `调试：随机诗题完整仿真` 或其他配置。
4. 按 `F5` 启动；停住后查看“变量”“监视”和调用堆栈。
5. 用 `F10` 执行下一行，用 `F11` 进入函数，用 `Shift+F11` 返回上一层。

适合设置断点的位置：

- `fivefinger/autonomy/supervisor.py`：观察任务阶段如何切换；
- `fivefinger/autonomy/perception.py`：检查相机观测和目标位姿；
- `fivefinger/control.py`：检查关节目标和执行器命令；
- `experiments/blind_competition.py`：检查完整仿真流程；
- `fivefinger/robot_surface_monitor.py`：检查桌面间隙和停止条件。

不要在物理仿真的每一个时间步设置普通断点，否则会显著拖慢运行。需要观察高速信号时，记录到结果文件或只在越界条件满足时停下。

## 结果在哪里

随机完整流程保存在：

```text
outputs\blind_competition\<case_id>\
```

其中通常包含题面、识别结果、运行日志、审计结果和视频。比较两组参数时，不要只看视频是否顺眼，还要对比成功率、掉笔、接触力峰值、桌面最小间隙、墨迹接触率和用时。

## 修改代码后的基本流程

```powershell
git switch -c feature/你的改动名称
.\.venv\Scripts\python.exe -m pytest -q
git status
git diff
```

确认改动只包含本次任务后再提交。一个提交只解决一个问题，参数试验结果和大体积输出不要提交到仓库。

## 常见问题

### 找不到 Python 环境

重新运行 `setup.ps1`，然后执行 `Python: Select Interpreter`。不要选择系统中其他版本的 Python。

### 完整仿真提示缺少重定向依赖

运行 `setup_retarget.ps1`，确认存在 `.venv-retarget\Scripts\python.exe`。

### OCR 中文乱码

先在当前终端执行：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
```

### MuJoCo 窗口黑屏或立即关闭

先运行环境检查；再从 VS Code 终端启动，让错误信息保留在终端中。显卡驱动问题应先更新官方驱动，不能通过删除碰撞或安全检查来绕过。

### 仿真很慢

调参阶段先不加 `--video`。固定题目定位问题，确认后再运行随机多轮和录像。
