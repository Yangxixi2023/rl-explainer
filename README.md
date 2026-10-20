# RL Explainer

[在线体验](https://yangxixi2023.github.io/rl-explainer/)

面向桌面的强化学习说明器，交互结构参考 Transformer Explainer。

主图按同一批数据展示候选回答、奖励、优势、概率比、裁剪目标与参数更新。点击图中的阶段或数值可展开计算，点击梯度可查看逐样本贡献。PPO、GRPO、DPO、SFT 使用各自的数据流。经典 RL 页面运行 Q-learning 与 SARSA。

“单步”按样本累加梯度，逐项计算参数增量、指数权重、归一化分母与新概率。“更新一轮”依次播放当前批次的全部 epoch。“暂停”保留计算位置，继续从下一小步开始；一个 epoch 的参数在其计算完成后写入。切换算法保留共用的回答和参数节点，连线与数值随布局连续变化。

## 本地运行

```powershell
node server.mjs
```

打开 http://127.0.0.1:4317/ 。应用运行不需要远程模型或 API key。KaTeX、数学字体和样式位于 `dist/vendor/katex/`。

## 检查

```powershell
npm test
```

测试覆盖解析梯度与数值微分、PPO 裁剪方向、固定 rollout、零方差组、DPO 偏好方向、序列概率链、终止与截断、实际网格训练以及公式解析、逐步累加的梯度与完整 epoch 更新的一致性。

## 单文件版本

```powershell
npm install
node scripts/export-standalone.mjs
```

输出 `output/RL-Explainer.html`，包含脚本、样式、KaTeX 和字体，能直接在桌面浏览器中打开。

## 计算设置

后训练实验使用四个候选回答的 categorical softmax；完整回答视为一次动作。采样与解析梯度均在浏览器中执行。GRPO 使用完整候选分布的精确 KL，区别于 DeepSeekMath 论文中的采样 KL 项。PPO 使用单步任务的标量 Critic，GAE 另以三步轨迹展示。详细定义见页面中的“计算设置”。

## 发布

推送到 `main` 后，GitHub Actions 运行计算与公式测试，并将 `dist/` 发布到 GitHub Pages。网页使用相对资源路径，可在 `/rl-explainer/` 子路径运行。
