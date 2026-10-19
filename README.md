# 五指灵巧手书写仿真

围绕 DexHand021 右手与 JAKA Zu7 机械臂，构建从题目识别、笔画规划、抓笔到接触书写和质量评估的本地 MuJoCo 程序。2026 年 9 月起开发。

## 算法与程序

```text
题目/相机图像 → OCR / 字形笔画 → 落笔与抬笔轨迹
             → 机械臂 IK + 手指抓握 → 接触 / 墨迹反馈
             → 书写质量与比赛规则评分 → 运行记录
```

- [five_finger/fivefinger](five_finger/fivefinger/)：感知、轨迹、IK、接触与抓握、书写评估的核心实现。
- [five_finger/config](five_finger/config/)：模型来源、规则和场景配置。
- [five_finger/experiments](five_finger/experiments/)：实验与评估入口。
- [five_finger/tests](five_finger/tests/)：几何、评分、时序和物理相关测试。
- [架构与文件地图](five_finger/docs/03_架构与文件地图.md)。

## 本地运行

Windows / PowerShell、Python 3.12：进入 `five_finger` 后执行 `./setup.ps1`，再按该目录的 [运行手册](five_finger/README.md) 启动。安装脚本按 `config/model_sources.json` 获取上游模型；模型、汉字笔画库和权重不打包进公开源码。汉字笔画使用 Hanzi Writer 数据，遵守其上游许可。

此次公开副本已通过 24 项评分、规则、命令时钟、坐标几何、墨迹质量等测试，详见 [VALIDATION.md](VALIDATION.md)。它们验证选定软件逻辑；完整接触仿真仍需模型与场景，实机和赛事正式验收没有由这些测试建立。
