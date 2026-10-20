// All displayed training measurements originate here. No model or API is fetched.
export function randomSource(seed) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
}
export function softmax(logits) {
  const max = Math.max(...logits), exps = logits.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}
export function sample(probabilities, random) {
  const x = random(); let total = 0;
  for (let i = 0; i < probabilities.length - 1; i++) { total += probabilities[i]; if (x < total) return i; }
  return probabilities.length - 1;
}
export const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
export const entropy = p => -p.reduce((sum, x) => sum + (x === 0 ? 0 : x * Math.log(x)), 0);
export const kl = (p, q) => p.reduce((sum, x, i) => sum + (x === 0 ? 0 : x * Math.log(x / q[i])), 0);
export const sigmoid = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
export const softplus = x => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
export function standardize(xs) {
  const average = mean(xs), sd = Math.sqrt(mean(xs.map(x => (x - average) ** 2)));
  return { mean: average, sd, advantages: sd === 0 ? xs.map(() => 0) : xs.map(x => (x - average) / sd) };
}
export function clippedSurrogate(ratio, advantage, epsilon) {
  const clippedRatio = Math.max(1 - epsilon, Math.min(1 + epsilon, ratio));
  return { value: Math.min(ratio * advantage, clippedRatio * advantage), clippedRatio,
    active: (advantage > 0 && ratio > 1 + epsilon) || (advantage < 0 && ratio < 1 - epsilon) };
}

export const TASKS = {
  math: {
    name: '算术推理', kind: '可验证奖励', prompt: '17 × 3 等于多少？请给出计算过程。',
    reference: [0.24, 0.32, 0.20, 0.24],
    responses: [
      { tokens: ['17', '×', '3', '=', '17+17+17', '=', '51', '。', '<eos>'], answer: 51, note: '结果正确 · 完整计算' },
      { tokens: ['17', '×', '3', '=', '54', '。', '<eos>'], answer: 54, note: '结果错误 · 乘法错误' },
      { tokens: ['答案', '是', '51', '。', '<eos>'], answer: 51, note: '结果正确 · 简短回答' },
      { tokens: ['17', '×', '3', '=', '34', '。', '<eos>'], answer: 34, note: '结果错误 · 少加一次' }
    ],
    rewards() { return this.responses.map(response => Number(response.answer === 17 * 3)); },
    explanation: '将回答中的最终数值与标准答案 51 比较：匹配得 1 分，否则得 0 分。两条正确回答同分。'
  },
  code: {
    name: '代码生成', kind: '单元测试奖励', prompt: '实现 square(n)，返回整数 n 的平方。',
    reference: [0.22, 0.34, 0.26, 0.18],
    responses: [
      { tokens: ['return', ' n', ' *', ' n', ';', '<eos>'], fn: n => n * n, note: '标准乘法' },
      { tokens: ['return', ' n', ' +', ' n', ';', '<eos>'], fn: n => n + n, note: '把平方误写成加倍' },
      { tokens: ['return', ' n', ' **', ' 2', ';', '<eos>'], fn: n => n ** 2, note: '幂运算' },
      { tokens: ['return', ' Math.abs(n)', ';', '<eos>'], fn: n => Math.abs(n), note: '返回了绝对值' }
    ],
    rewards() { const tests = [-3, 0, 2, 5]; return this.responses.map(r => mean(tests.map(n => Number(r.fn(n) === n * n)))); },
    explanation: '分别以 −3、0、2、5 运行候选函数，比较输出与输入的平方。奖励取四项测试的通过比例。'
  },
  preference: {
    name: '助教回答', kind: '人工偏好评分', prompt: '为什么训练模型时，需要保留验证集？',
    reference: [0.20, 0.30, 0.24, 0.26],
    responses: [
      { tokens: ['验证集', '用于', '选择超参数', '与', '检测过拟合', '；', '测试集', '用于最终评估', '。', '<eos>'], note: '准确、区分了验证与测试' },
      { tokens: ['验证集', '直接', '参与梯度更新', '，', '能提高准确率', '。', '<eos>'], note: '把验证集当成训练集' },
      { tokens: ['用未见过的数据', '检查', '模型', '能否泛化', '。', '<eos>'], note: '正确但省略了使用边界' },
      { tokens: ['因为', '大家', '都这样做', '。', '<eos>'], note: '缺少解释' }
    ],
    rewards: () => [1, 0, 0.7, 0.1],
    explanation: '四条回答各有一个可编辑分数。策略按这些分数计算优势；提高某条回答的评分，可以改变更新方向。'
  }
};

export const DEFAULT_CONFIG = { group: 8, lr: 0.18, clip: 0.2, beta: 0.04, dpoBeta: 1, epochs: 4 };
export function createLanguageState(taskKey = 'math', seed = 42) {
  const task = TASKS[taskKey];
  return { taskKey, theta: task.reference.map(Math.log), reference: [...task.reference], value: 0,
    random: randomSource(seed), rounds: 0, updates: 0, history: [], scores: task.rewards(), winner: 0, loser: 1 };
}
export function languageMetrics(state) {
  const p = softmax(state.theta);
  return { p, reward: p.reduce((s, x, i) => s + x * state.scores[i], 0), entropy: entropy(p), kl: kl(p, state.reference) };
}
export function prepareBatch(state, method, config) {
  const old = softmax(state.theta), oldValue = state.value;
  const ids = method === 'dpo' ? [state.winner, state.loser] : method === 'sft' ? [state.winner] : Array.from({ length: config.group }, () => sample(old, state.random));
  const rewards = ids.map(i => state.scores[i]);
  const shaped = rewards.map((r, k) => method === 'ppo' ? r - config.beta * Math.log(old[ids[k]] / state.reference[ids[k]]) : r);
  const standardized = standardize(rewards);
  const advantages = method === 'grpo' ? standardized.advantages : shaped.map(r => r - oldValue);
  return { method, old, oldValue, thetaBefore: [...state.theta], ids, rewards, shaped, advantages, mean: standardized.mean, sd: standardized.sd,
    config: { ...config }, winner: state.winner, loser: state.loser, before: languageMetrics(state), detail: null, epochDetails: [], optimized: false };
}

// Gradient of the finite categorical objective, holding rollout samples and old policy fixed.
export function languageObjective(theta, reference, batch) {
  const p = softmax(theta), n = p.length, gradient = Array(n).fill(0);
  const { method, config, ids } = batch;
  let objective = 0, clipped = 0;
  const samples = [];
  if (method === 'dpo') {
    const w = batch.winner, l = batch.loser;
    const margin = Math.log(p[w] / reference[w]) - Math.log(p[l] / reference[l]);
    const z = config.dpoBeta * margin;
    objective = -softplus(-z);
    gradient[w] = config.dpoBeta * sigmoid(-z);
    gradient[l] = -gradient[w];
    return { objective, gradient, clipped, margin, z, preference: sigmoid(z), samples };
  }
  if (method === 'sft') {
    const w = batch.winner;
    objective = Math.log(p[w]);
    for (let j = 0; j < n; j++) gradient[j] = Number(j === w) - p[j];
    return { objective, gradient, clipped, samples };
  }
  ids.forEach((id, k) => {
    const ratio = p[id] / batch.old[id], advantage = batch.advantages[k];
    const term = clippedSurrogate(ratio, advantage, config.clip);
    const refRatio = reference[id] / p[id];
    const sampledKL = refRatio - Math.log(refRatio) - 1;
    // PPO's KL penalty is fixed in rollout rewards. GRPO uses exact categorical KL below.
    const coefficient = term.active ? 0 : ratio * advantage;
    objective += term.value / ids.length;
    for (let j = 0; j < n; j++) gradient[j] += coefficient * (Number(j === id) - p[j]) / ids.length;
    clipped += Number(term.active);
    samples.push({ id, ratio, advantage, surrogate: term.value, clipped: term.active, sampledKL });
  });
  if (method === 'grpo') {
    const exactKL = kl(p, reference);
    objective -= config.beta * exactKL;
    for (let j = 0; j < n; j++) gradient[j] -= config.beta * p[j] * (Math.log(p[j] / reference[j]) - exactKL);
  }
  return { objective, gradient, clipped, samples };
}
export function optimizeEpoch(state, batch) {
  if (batch.optimized) throw new Error('This rollout has already been optimized. Collect a new batch.');
  const epochs = batch.method === 'sft' || batch.method === 'dpo' ? 1 : batch.config.epochs;
  const result = languageObjective(state.theta, state.reference, batch);
  result.thetaBefore = [...state.theta];
  state.theta = state.theta.map((x, j) => x + batch.config.lr * result.gradient[j]);
  result.thetaAfter = [...state.theta];
  if (batch.method === 'ppo') state.value += batch.config.lr * (mean(batch.shaped) - state.value);
  state.updates++;
  batch.epochDetails.push(result);
  batch.detail = languageObjective(state.theta, state.reference, batch);
  if (batch.epochDetails.length < epochs) return batch;
  batch.optimized = true;
  state.rounds++;
  state.history.push({ round: state.rounds, ...languageMetrics(state), objective: batch.detail.objective,
    method: batch.method, config: { ...batch.config }, samples: [...batch.ids], rewards: [...batch.rewards],
    old: [...batch.old], oldValue: batch.oldValue, advantages: [...batch.advantages], logits: [...state.theta] });
  return batch;
}
export function optimizeBatch(state, batch) {
  if (batch.optimized) throw new Error('This rollout has already been optimized. Collect a new batch.');
  while (!batch.optimized) optimizeEpoch(state, batch);
  return batch;
}

// Conditional token probabilities are exact marginals of the finite response tree.
export function tokenTrace(task, responseIndex, probabilities) {
  const tokens = task.responses[responseIndex].tokens;
  return tokens.map((token, position) => {
    let parent = 0, child = 0;
    task.responses.forEach((response, i) => {
      if (tokens.slice(0, position).every((t, j) => response.tokens[j] === t)) {
        parent += probabilities[i];
        if (response.tokens[position] === token) child += probabilities[i];
      }
    });
    return { token, probability: child / parent, logProbability: Math.log(child / parent) };
  });
}

export const ACTIONS = [{ symbol: '↑', name: '上', dr: -1, dc: 0 }, { symbol: '→', name: '右', dr: 0, dc: 1 }, { symbol: '↓', name: '下', dr: 1, dc: 0 }, { symbol: '←', name: '左', dr: 0, dc: -1 }];
export function createGrid(seed = 42) {
  return { size: 5, start: 20, goal: 4, walls: new Set([6, 7, 16]), traps: new Set([12, 18]),
    q: Array.from({ length: 25 }, () => Array(4).fill(0)), position: 20, episodes: 0, totalSteps: 0,
    episodeSteps: 0, episodeReturn: 0, path: [20], history: [], last: null, nextAction: null, random: randomSource(seed) };
}
export function gridTransition(grid, position, action, stepReward) {
  const a = ACTIONS[action], row = Math.floor(position / grid.size) + a.dr, col = position % grid.size + a.dc;
  let next = position;
  if (row >= 0 && row < grid.size && col >= 0 && col < grid.size && !grid.walls.has(row * grid.size + col)) next = row * grid.size + col;
  const terminal = next === grid.goal || grid.traps.has(next);
  return { next, reward: next === grid.goal ? 1 : grid.traps.has(next) ? -1 : stepReward, terminal };
}
export function epsilonPolicy(values, epsilon) {
  const best = Math.max(...values), bestIds = values.map((x, i) => x === best ? i : -1).filter(i => i !== -1);
  return values.map((_, i) => epsilon / values.length + (bestIds.includes(i) ? (1 - epsilon) / bestIds.length : 0));
}
export function tdTarget(reward, gamma, bootstrap, terminal) { return reward + (terminal ? 0 : gamma * bootstrap); }
export function gridStep(grid, config) {
  if (grid.last?.terminal || grid.last?.truncated) {
    grid.position = grid.start; grid.episodeSteps = 0; grid.episodeReturn = 0; grid.path = [grid.start]; grid.nextAction = null;
  }
  const s = grid.position;
  const action = config.method === 'sarsa' && grid.nextAction !== null ? grid.nextAction : sample(epsilonPolicy(grid.q[s], config.epsilon), grid.random);
  const transition = gridTransition(grid, s, action, config.stepReward);
  const nextAction = transition.terminal ? null : sample(epsilonPolicy(grid.q[transition.next], config.epsilon), grid.random);
  const bootstrap = transition.terminal ? 0 : config.method === 'sarsa' ? grid.q[transition.next][nextAction] : Math.max(...grid.q[transition.next]);
  const old = grid.q[s][action], target = tdTarget(transition.reward, config.gamma, bootstrap, transition.terminal), delta = target - old;
  grid.q[s][action] += config.alpha * delta;
  grid.position = transition.next; grid.episodeSteps++; grid.totalSteps++; grid.episodeReturn += transition.reward;
  grid.path.push(grid.position); grid.nextAction = nextAction;
  const truncated = !transition.terminal && grid.episodeSteps >= config.horizon;
  grid.last = { s, action, ...transition, truncated, old, target, delta, updated: grid.q[s][action], bootstrap, nextAction, config: { ...config } };
  if (transition.terminal || truncated) {
    grid.episodes++;
    grid.history.push({ episode: grid.episodes, reward: grid.episodeReturn, success: grid.position === grid.goal, steps: grid.episodeSteps });
  }
  return grid.last;
}

export function gae(rewards, values, gamma, lambda, terminated = true, bootstrapValue = 0) {
  const advantages = Array(rewards.length).fill(0), deltas = Array(rewards.length).fill(0);
  let nextAdvantage = 0;
  for (let t = rewards.length - 1; t >= 0; t--) {
    const last = t === rewards.length - 1;
    const nextValue = last ? (terminated ? 0 : bootstrapValue) : values[t + 1];
    deltas[t] = rewards[t] + gamma * nextValue - values[t];
    advantages[t] = deltas[t] + (last ? 0 : gamma * lambda * nextAdvantage);
    nextAdvantage = advantages[t];
  }
  return { advantages, deltas, returns: advantages.map((a, t) => a + values[t]) };
}
