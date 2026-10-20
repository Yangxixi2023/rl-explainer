import test from 'node:test';
import assert from 'node:assert/strict';
import { EQ, typeset } from '../dist/math-render.mjs';
import { ALGORITHMS, CONCEPTS, DIALOGS, TERMS } from '../dist/content.mjs';
import { createUpdateSequence, currentUpdateFrame } from '../dist/update-steps.mjs';
import { createLanguageState, prepareBatch, optimizeEpoch, optimizeBatch, softmax, DEFAULT_CONFIG } from '../dist/engine.mjs';
import { renderComputationGraph } from '../dist/graph.mjs';

test('every authored equation parses as TeX and includes accessible MathML',()=>{
  const equations=[...Object.values(EQ).filter(x=>typeof x==='string'),...Object.values(EQ.algorithms)];
  for(const equation of equations){const rendered=typeset(equation,true);assert.match(rendered,/<math /);assert.match(rendered,/class="katex-html"/);}
  assert.equal(new Set(ALGORITHMS.map(algorithm=>algorithm.id)).size,ALGORITHMS.length);
  assert.ok(CONCEPTS.every(([title,body])=>title&&body.includes('<p>')));
  for(const [,body] of [...Object.values(DIALOGS),...Object.values(TERMS)]){
    assert.doesNotMatch(body,/\$\{(?:displayEquation|typeset)\(/);
  }
  for(const [,body] of Object.values(TERMS))assert.match(body,/class="katex"/);
});

test('incremental update frames sum the actual gradients and leave parameters untouched until the epoch is applied',()=>{
  const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-12,`${a} differs from ${b}`);
  for(const method of ['grpo','ppo','dpo','sft']){
    const state=createLanguageState(),batch=prepareBatch(state,method,DEFAULT_CONFIG);
    while(!batch.optimized){
      const before=[...state.theta],updates=state.updates,sequence=createUpdateSequence(state,batch);
      for(let i=0;i<sequence.steps.length;i++){
        sequence.cursor=i;
        const frame=currentUpdateFrame(sequence);
        assert.doesNotMatch(frame.tex,/\t|,,/, 'TeX operators and vector separators must survive JavaScript string interpolation');
        assert.match(typeset(frame.tex,true),/<math /);
        const graph=renderComputationGraph(state,batch,0,0,{preview:true,operation:sequence});
        assert.doesNotMatch(graph,/NaN|undefined|>null</);
        assert.deepEqual(state.theta,before);assert.equal(state.updates,updates);
        if(frame.kind==='parameter')frame.gradient.forEach((value,j)=>close(value,sequence.objective.gradient[j]));
      }
      optimizeEpoch(state,batch);
      const last=currentUpdateFrame(sequence);
      state.theta.forEach((value,j)=>close(value,last.theta[j]));
      softmax(state.theta).forEach((value,j)=>close(value,last.probabilities[j]));
      if(method==='ppo')close(state.value,last.valueAfter);
      assert.equal(state.updates,updates+1);
    }
  }
});

test('all four computation graphs render their actual batch data before and after updating',()=>{
  for(const method of ['grpo','ppo','dpo','sft'])for(const task of ['math','code','preference']){
    const state=createLanguageState(task),batch=prepareBatch(state,method,DEFAULT_CONFIG);
    const before=renderComputationGraph(state,batch,0);assert.match(before,/<svg /);assert.doesNotMatch(before,/NaN|undefined|#aaa12/);assert.match(before,/class="katex"/);
    const keys=[...before.matchAll(/data-motion-key="([^"]+)"/g)].map(match=>match[1]);
    assert.equal(new Set(keys).size,keys.length,`${method}: motion keys must identify one visible entity each`);
    optimizeBatch(state,batch);
    for(let epoch=0;epoch<batch.epochDetails.length;epoch++){
      const after=renderComputationGraph(state,batch,0,epoch);assert.doesNotMatch(after,/NaN|undefined|#aaa12/);
    }
  }
});
