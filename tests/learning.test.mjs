import test from 'node:test';
import assert from 'node:assert/strict';
import { TASKS, DEFAULT_CONFIG, softmax, mean, randomSource, createLanguageState, languageMetrics, prepareBatch, languageObjective, optimizeEpoch, optimizeBatch, standardize, clippedSurrogate, tokenTrace, tdTarget, createGrid, gridTransition, gridStep, epsilonPolicy, gae } from '../dist/engine.mjs';

const close = (actual, expected, tolerance = 1e-8) => assert.ok(Math.abs(actual-expected) < tolerance, `${actual} ≠ ${expected}`);

test('pausing after one real epoch and resuming produces the same completed update',()=>{
  for(const method of ['ppo','grpo']){
    const whole=createLanguageState(),stepped=createLanguageState();
    const one=prepareBatch(whole,method,DEFAULT_CONFIG),two=prepareBatch(stepped,method,DEFAULT_CONFIG);
    optimizeEpoch(stepped,two);
    assert.equal(stepped.updates,1);assert.equal(stepped.rounds,0);assert.equal(stepped.history.length,0);assert.equal(two.optimized,false);
    assert.deepEqual(two.epochDetails[0].thetaAfter,stepped.theta);
    optimizeBatch(stepped,two);optimizeBatch(whole,one);
    assert.deepEqual(stepped.theta,whole.theta);assert.equal(stepped.value,whole.value);assert.equal(stepped.updates,4);assert.equal(stepped.rounds,1);
    assert.deepEqual(two.old,one.old);assert.deepEqual(two.advantages,one.advantages);
  }
});

test('all training objective gradients agree with central finite differences', () => {
  for (const method of ['grpo','ppo','dpo','sft']) {
    const state = createLanguageState();
    const batch = prepareBatch(state,method,DEFAULT_CONFIG);
    // The second point crosses clipping boundaries and exercises a nonzero KL gradient.
    for (const theta of [state.theta, state.theta.map((x,i) => x + [.6,-.5,.2,-.2][i])]) {
      const result = languageObjective(theta,state.reference,batch);
      for(let j=0;j<theta.length;j++) {
        const plus=[...theta],minus=[...theta],h=1e-6; plus[j]+=h;minus[j]-=h;
        const numeric=(languageObjective(plus,state.reference,batch).objective-languageObjective(minus,state.reference,batch).objective)/(2*h);
        close(result.gradient[j],numeric,2e-7);
      }
      close(result.gradient.reduce((x,y)=>x+y),0);
    }
  }
});

test('PPO clipping follows the sign of advantage, including both unclipped harmful directions', () => {
  assert.equal(clippedSurrogate(1.4,2,.2).active,true);
  close(clippedSurrogate(1.4,2,.2).value,2.4);
  assert.equal(clippedSurrogate(.6,-2,.2).active,true);
  close(clippedSurrogate(.6,-2,.2).value,-1.6);
  assert.equal(clippedSurrogate(.6,2,.2).active,false);
  close(clippedSurrogate(.6,2,.2).value,1.2);
  assert.equal(clippedSurrogate(1.4,-2,.2).active,false);
  close(clippedSurrogate(1.4,-2,.2).value,-2.8);
  close(clippedSurrogate(2,0,.2).value,0);
});

test('GRPO standardization uses population variance and defines zero-information groups explicitly', () => {
  const group=standardize([1,1,0,0]);close(group.mean,.5);close(group.sd,.5);
  assert.deepEqual(group.advantages,[1,1,-1,-1]);
  assert.deepEqual(standardize([1,1,1,1]).advantages,[0,0,0,0]);
  assert.deepEqual(standardize([0,0,0,0]).advantages,[0,0,0,0]);
});

test('zero group advantage can still produce a KL gradient toward the reference', () => {
  const state=createLanguageState();state.scores=[1,1,1,1];state.theta[0]+=.6;
  const batch=prepareBatch(state,'grpo',DEFAULT_CONFIG),before=languageMetrics(state).kl;
  assert.equal(batch.sd,0);optimizeBatch(state,batch);
  assert.ok(languageMetrics(state).kl<before);
});

test('each real training method improves its well-labelled finite task from seed 42', () => {
  for(const method of ['grpo','ppo','dpo','sft']) {
    const state=createLanguageState(),before=languageMetrics(state).reward,reference=[...state.reference];
    for(let i=0;i<40;i++)optimizeBatch(state,prepareBatch(state,method,DEFAULT_CONFIG));
    const after=languageMetrics(state);assert.ok(after.reward>before+.2,method);
    close(after.p.reduce((x,y)=>x+y),1);assert.deepEqual(state.reference,reference);assert.equal(state.history.length,40);
    if(method==='ppo'||method==='grpo')assert.equal(state.updates,160);else assert.equal(state.updates,40);
  }
});

test('rollout old policy, advantages and reward shaping stay fixed across epochs', () => {
  const state=createLanguageState(),batch=prepareBatch(state,'ppo',DEFAULT_CONFIG);
  const old=[...batch.old],advantages=[...batch.advantages],shaped=[...batch.shaped],oldValue=batch.oldValue;
  optimizeBatch(state,batch);
  assert.deepEqual(batch.old,old);assert.deepEqual(batch.advantages,advantages);assert.deepEqual(batch.shaped,shaped);assert.equal(batch.oldValue,oldValue);
  assert.notDeepEqual(softmax(state.theta),old);assert.notEqual(state.value,oldValue);
  assert.throws(()=>optimizeBatch(state,batch),/already been optimized/);
});

test('SFT follows its demonstration even when that lowers the independent validation reward', () => {
  const state=createLanguageState();state.winner=1;const before=languageMetrics(state).reward;
  for(let i=0;i<20;i++)optimizeBatch(state,prepareBatch(state,'sft',DEFAULT_CONFIG));
  assert.ok(languageMetrics(state).reward<before);assert.ok(softmax(state.theta)[1]>.7);
});

test('DPO follows chosen over rejected, with no online reward dependence', () => {
  const state=createLanguageState(),other=createLanguageState();other.scores=[100,-10,42,77];
  const a=prepareBatch(state,'dpo',DEFAULT_CONFIG),b=prepareBatch(other,'dpo',DEFAULT_CONFIG);
  assert.deepEqual(languageObjective(state.theta,state.reference,a).gradient,languageObjective(other.theta,other.reference,b).gradient);
  const before=languageObjective(state.theta,state.reference,a).objective;optimizeBatch(state,a);
  assert.ok(languageObjective(state.theta,state.reference,a).objective>before);
});

test('token conditional products reproduce exact sequence probability for every task', () => {
  const p=softmax([.7,-.4,.1,-.1]);
  for(const task of Object.values(TASKS))for(let i=0;i<4;i++) {
    const trace=tokenTrace(task,i,p);
    close(trace.reduce((x,t)=>x*t.probability,1),p[i]);
    close(trace.reduce((x,t)=>x+t.logProbability,0),Math.log(p[i]));
  }
  assert.deepEqual(TASKS.code.rewards(),[1,.5,1,.25]);
});

test('seeded samplers reproduce trajectories and epsilon greedy shares all ties', () => {
  const a=randomSource(42),b=randomSource(42);for(let i=0;i<30;i++)close(a(),b());
  assert.deepEqual(epsilonPolicy([0,0,0,0],0),[.25,.25,.25,.25]);
  assert.deepEqual(epsilonPolicy([1,1,0,0],.2),[.45,.45,.05,.05]);
});

test('true terminal masks bootstrap; time truncation does not', () => {
  close(tdTarget(1,.9,10,true),1);close(tdTarget(1,.9,10,false),10);
  const grid=createGrid();grid.position=0;grid.q[1]=[2,2,2,2];
  const c={method:'q',epsilon:0,alpha:.5,gamma:.9,stepReward:-.04,horizon:1};
  grid.q[0]=[0,1,0,0];const result=gridStep(grid,c);
  assert.equal(result.next,1);assert.equal(result.truncated,true);close(result.target,1.76);close(result.updated,1.38);
  const terminal=createGrid();terminal.position=3;terminal.q[3]=[0,2,0,0];terminal.q[4]=[100,100,100,100];
  const last=gridStep(terminal,c);assert.equal(last.terminal,true);close(last.target,1);
});

test('walls keep the agent in place, and SARSA actually executes its previously selected next action', () => {
  const grid=createGrid();assert.equal(gridTransition(grid,5,1,-.04).next,5);
  const c={method:'sarsa',epsilon:1,alpha:.3,gamma:.95,stepReward:-.04,horizon:80};
  const first=gridStep(grid,c);assert.equal(first.terminal,false);const nextAction=first.nextAction;
  const second=gridStep(grid,c);assert.equal(second.action,nextAction);
});

test('real grid training finds the goal consistently for both TD control methods', () => {
  for(const method of ['q','sarsa']) {
    const grid=createGrid(),c={method,epsilon:.2,alpha:.3,gamma:.95,stepReward:-.04,horizon:80};
    while(grid.episodes<100)gridStep(grid,c);
    assert.ok(mean(grid.history.slice(-20).map(x=>Number(x.success)))>.8,method);
    assert.ok(grid.history.every(x=>x.steps<=80));assert.ok(grid.totalSteps>100);
  }
});

test('GAE with lambda=1 telescopes to discounted full return minus value', () => {
  const rewards=[.1,.2,1],values=[.2,.35,.6],gamma=.95;
  const terminated=gae(rewards,values,gamma,1,true);
  close(terminated.advantages[0],.1+gamma*.2+gamma**2-values[0]);
  const truncated=gae(rewards,values,gamma,1,false,.7);
  close(truncated.advantages[0],.1+gamma*.2+gamma**2+gamma**3*.7-values[0]);
  assert.deepEqual(gae(rewards,values,gamma,0,true).advantages,terminated.deltas);
});
