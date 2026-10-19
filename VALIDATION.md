# 公开版本验证

2026-10-20，公开源码副本的选定测试 **24 passed**。

在 `five_finger` 目录运行：

```text
python -m pytest -q tests/test_2026_scoring.py tests/test_competition_rules.py tests/test_command_clock.py tests/test_axial_frame.py tests/test_ink_quality.py tests/test_reference_velocity.py tests/test_profile_freeze.py
```

覆盖比赛评分与规则、命令时钟、坐标几何、墨迹质量、参考速度及配置冻结等软件逻辑。此次未在公开副本重新取得完整外部模型与笔画资产，未运行完整接触物理测试，不将 24 项软件测试换算成书写成功率或赛事成绩。
