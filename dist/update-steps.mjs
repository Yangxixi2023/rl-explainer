import { softmax, languageObjective, kl, mean } from './engine.mjs';

const updateFixed=(value,digits=4)=>value.toFixed(digits);
const updateLetter=index=>String.fromCharCode(65+index);
const updateVector=values=>String.raw`\left[${values.map(value=>updateFixed(value)).join(',')}\right]`;

// Each frame reveals one calculation from the same pre-update parameter vector.
export function createUpdateSequence(state,batch) {
  const thetaBefore=[...state.theta],p=softmax(thetaBefore);
  const objective=languageObjective(thetaBefore,state.reference,batch);
  const thetaAfter=thetaBefore.map((value,j)=>value+batch.config.lr*objective.gradient[j]);
  const probabilities=softmax(thetaAfter),shift=Math.max(...thetaAfter);
  const weights=thetaAfter.map(value=>Math.exp(value-shift)),normalizer=weights.reduce((sum,value)=>sum+value,0);
  const frame={gradient:thetaBefore.map(()=>0),theta:[...thetaBefore],probabilities:[...p],weights:thetaBefore.map(()=>null),normalizer:null,thetaReady:thetaBefore.map(()=>false)};
  const steps=[];
  const push=(kind,title,note,tex,extra={})=>steps.push({kind,title,note,tex,...extra,gradient:[...frame.gradient],theta:[...frame.theta],probabilities:[...frame.probabilities],weights:[...frame.weights],normalizer:frame.normalizer,thetaReady:[...frame.thetaReady]});
  const online=batch.method==='ppo'||batch.method==='grpo';
  if(online) {
    objective.samples.forEach((sample,k)=>{
      const coefficient=sample.clipped?0:sample.ratio*sample.advantage;
      const contribution=p.map((probability,j)=>coefficient*(Number(j===sample.id)-probability)/batch.ids.length);
      frame.gradient=frame.gradient.map((value,j)=>value+contribution[j]);
      push('sample',`累加样本 ${k+1} / ${batch.ids.length}`,sample.clipped?'裁剪项进入平坦区，这条样本的贡献为零。':`回答 ${updateLetter(sample.id)} 的梯度除以组大小，再累加到四个参数。`,String.raw`\begin{aligned}c_{${k+1}}&=${sample.clipped?'0':String.raw`${updateFixed(sample.ratio)}\times(${updateFixed(sample.advantage)})`}\\g&\leftarrow g+${updateVector(contribution)}\end{aligned}`,{sample:k});
    });
    if(batch.method==='grpo') {
      const K=kl(p,state.reference);
      const contribution=p.map((probability,j)=>-batch.config.beta*probability*(Math.log(probability/state.reference[j])-K));
      frame.gradient=frame.gradient.map((value,j)=>value+contribution[j]);
      push('regularize','加入 KL 梯度','样本贡献取平均后，加上相对参考策略的约束。',String.raw`g\leftarrow g+${updateVector(contribution)}`);
    }
  } else {
    frame.gradient=[...objective.gradient];
    push('gradient','计算目标梯度',batch.method==='dpo'?'偏好与拒绝的梯度方向相反，其他坐标为零。':'示范坐标取一，再减去当前概率向量。',String.raw`g=${batch.method==='dpo'?String.raw`\beta\sigma(-m)(e_w-e_l)`:'e_{y^*}-p'}=${updateVector(frame.gradient)}`);
  }
  thetaBefore.forEach((old,j)=>{
    frame.theta[j]=thetaAfter[j];frame.thetaReady[j]=true;
    const delta=batch.config.lr*objective.gradient[j];
    push('parameter',`计算参数 ${updateLetter(j)} 的增量`,'四个增量都使用这一 epoch 开始时的梯度。',String.raw`\begin{aligned}\Delta\theta_{${updateLetter(j)}}&=${updateFixed(batch.config.lr,2)}\times(${updateFixed(objective.gradient[j])})=${updateFixed(delta)}\\\theta^{\rm new}_{${updateLetter(j)}}&=${updateFixed(old)}+(${updateFixed(delta)})=${updateFixed(thetaAfter[j])}\end{aligned}`,{parameter:j});
  });
  frame.probabilities.fill(null);
  weights.forEach((weight,j)=>{
    frame.weights[j]=weight;
    push('weight',`指数变换 ${updateLetter(j)}`,'减去最大 logit 后取指数，保持相同的归一化概率。',String.raw`z_{${updateLetter(j)}}=e^{\theta^{\rm new}_{${updateLetter(j)}}-b}=e^{${updateFixed(thetaAfter[j])}-(${updateFixed(shift)})}=${updateFixed(weight)},\quad b=\max_j\theta^{\rm new}_j`,{parameter:j});
  });
  frame.normalizer=normalizer;
  push('sum','求归一化分母','把四个指数权重相加。',String.raw`Z=\sum_j z_j=${weights.map(value=>updateFixed(value)).join('+')}=${updateFixed(normalizer)}`);
  probabilities.forEach((probability,j)=>{
    frame.probabilities[j]=probability;
    push('probability',`归一化概率 ${updateLetter(j)}`,'每项除以同一个分母，四项之和为一。',String.raw`p^{\rm new}_{${updateLetter(j)}}=\frac{z_{${updateLetter(j)}}}{Z}=\frac{${updateFixed(weights[j])}}{${updateFixed(normalizer)}}\approx${updateFixed(probability)}`,{parameter:j});
  });
  if(batch.method==='ppo'){
    const target=mean(batch.shaped),valueAfter=state.value+batch.config.lr*(target-state.value);
    push('critic','更新 Critic','以本批整形回报的均值为目标，更新标量价值预测。',String.raw`\begin{aligned}\overline{\widetilde R}&=${updateFixed(target)}\\V^{\rm new}&=${updateFixed(state.value)}+${updateFixed(batch.config.lr,2)}\,(${updateFixed(target)}-${updateFixed(state.value)})\approx${updateFixed(valueAfter)}\end{aligned}`,{valueAfter});
  }
  return {epoch:batch.epochDetails.length+1,thetaBefore,thetaAfter,objective,steps,cursor:-1};
}

export function currentUpdateFrame(sequence) {
  return sequence.cursor<0?null:sequence.steps[sequence.cursor];
}
export function updateComplete(sequence) {
  return sequence.cursor===sequence.steps.length-1;
}
