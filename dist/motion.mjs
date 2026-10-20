const SVG_NS='http://www.w3.org/2000/svg';
export const motionPreference=matchMedia('(prefers-reduced-motion: reduce)');
export const timings={layout:560,number:440,flow:510,epoch:480,detail:300};
const ease=t=>1-(1-t)**3;

export function waitForMotion(milliseconds,signal){
  if(signal?.aborted)return Promise.resolve(false);
  return new Promise(resolve=>{
    const finish=completed=>{clearTimeout(timer);signal?.removeEventListener('abort',abort);resolve(completed);};
    const abort=()=>finish(false);
    const timer=setTimeout(()=>finish(true),motionPreference.matches?0:milliseconds);
    signal?.addEventListener('abort',abort,{once:true});
  });
}
function play(element,keyframes,options){
  if(motionPreference.matches)return Promise.resolve();
  const animation=element.animate(keyframes,{easing:'cubic-bezier(.22,1,.36,1)',...options});
  return new Promise(resolve=>{animation.onfinish=animation.oncancel=()=>resolve();});
}
function pathInterpolator(from,to){
  const commands=value=>value.replace(/-?\d*\.?\d+(?:e[-+]?\d+)?/gi,'#');
  const numbers=value=>[...value.matchAll(/-?\d*\.?\d+(?:e[-+]?\d+)?/gi)].map(x=>Number(x[0]));
  let a=numbers(from),b=numbers(to);
  if(commands(from)!==commands(to)){
    if(!/^[ML\d.,\s-]+$/.test(from)||!/^[ML\d.,\s-]+$/.test(to))throw new Error('Motion paths must use matching cubic commands or polylines.');
    while(a.length<b.length)a.push(...a.slice(-2));
    while(b.length<a.length)b.push(...b.slice(-2));
    return progress=>Array.from({length:b.length/2},(_,i)=>`${i?'L':'M'}${a[i*2]+(b[i*2]-a[i*2])*progress},${a[i*2+1]+(b[i*2+1]-a[i*2+1])*progress}`).join(' ');
  }
  return progress=>{let i=0;return to.replace(/-?\d*\.?\d+(?:e[-+]?\d+)?/gi,()=>{const index=i++;return a[index]+(b[index]-a[index])*progress;});};
}
function formatNumber(value,node){
  const precision=Number(node.dataset.precision),suffix=node.dataset.suffix??'';
  const n=Number(value.toFixed(precision));
  return (n>0&&node.dataset.plus==='true'?'+':'')+n.toFixed(precision).replace('-','−')+suffix;
}

export class SvgMotion {
  constructor(host){this.host=host;this.tweens=[];this.frame=0;this.finishTweens=null;this.pending=Promise.resolve();}
  settle(){
    cancelAnimationFrame(this.frame);this.frame=0;
    for(const tween of this.tweens)tween.apply(1);
    this.tweens=[];
    this.finishTweens?.();this.finishTweens=null;
    this.host.querySelectorAll('[data-motion-exit]').forEach(node=>node.remove());
    this.host.getAnimations({subtree:true}).forEach(animation=>animation.cancel());
    this.host.style.overflow='';
    this.host.dataset.motionState='idle';
  }
  render(markup,{animate=true,duration=timings.layout}={}){
    const template=document.createElement('template');template.innerHTML=markup;
    const next=template.content.firstElementChild,current=this.host.firstElementChild;
    if(!current){this.host.append(next);this.host.dataset.motionState='idle';return Promise.resolve();}
    const previousViewBox=current.getAttribute('viewBox');
    const before=new Map([...current.querySelectorAll('[data-motion-key]')].map(node=>[node.dataset.motionKey,{node,box:node.getBoundingClientRect(),transform:getComputedStyle(node).transform}]));
    const numberValues=new Map([...current.querySelectorAll('[data-number]')].map(node=>[node,Number.parseFloat(node.textContent.replace('−','-'))]));
    const previousHeight=this.host.getBoundingClientRect().height;
    this.settle();
    const enabled=animate&&!motionPreference.matches;
    this.host.dataset.motionState=enabled?'moving':'idle';
    const animations=[];
    const used=new Set();
    const patch=(node,target)=>{
      if(node.nodeType===Node.TEXT_NODE){if(node.data!==target.data)node.data=target.data;return;}
      const oldNumber=numberValues.has(node)?numberValues.get(node):null;
      for(const attribute of [...node.attributes])if(!target.hasAttribute(attribute.name))node.removeAttribute(attribute.name);
      for(const attribute of target.attributes){
        const previous=node.getAttribute(attribute.name);
        if(previous===attribute.value)continue;
        node.setAttribute(attribute.name,attribute.value);
        if(enabled&&attribute.name==='d'&&previous){const interpolate=pathInterpolator(previous,attribute.value);this.tweens.push({apply:t=>node.setAttribute('d',interpolate(t))});}
        if(enabled&&(attribute.name==='fill'||attribute.name==='stroke')&&previous&&previous!=='none'&&attribute.value!=='none')animations.push(play(node,[{[attribute.name]:previous},{[attribute.name]:attribute.value}],{duration}));
      }
      if(node.hasAttribute('data-number')){
        const value=Number(node.dataset.number);node.textContent=target.textContent;
        if(enabled&&oldNumber!==null&&oldNumber!==value)this.tweens.push({apply:t=>node.textContent=formatNumber(oldNumber+(value-oldNumber)*t,node)});
        return;
      }
      if(node.classList.contains('graph-equation')){
        if(node.querySelector('annotation')?.textContent!==target.querySelector('annotation')?.textContent){node.replaceChildren(...target.cloneNode(true).childNodes);if(enabled)animations.push(play(node,[{opacity:.25},{opacity:1}],{duration:280}));}
        return;
      }
      const oldChildren=[...node.childNodes],keep=new Set();let cursor=node.firstChild;
      for(const wanted of [...target.childNodes]){
        const key=wanted.nodeType===1?wanted.getAttribute('data-motion-key'):null;
        let child=key?before.get(key)?.node:null;
        if(!child&&cursor&&cursor.nodeType===wanted.nodeType&&cursor.nodeName===wanted.nodeName&&!(cursor.nodeType===1&&cursor.hasAttribute('data-motion-key')))child=cursor;
        if(!child)child=wanted.cloneNode(true);
        if(child!==cursor)node.insertBefore(child,cursor);
        keep.add(child);if(key)used.add(key);
        patch(child,wanted);cursor=child.nextSibling;
      }
      for(const child of oldChildren)if(!keep.has(child)&&child.parentNode===node)child.remove();
    };
    patch(current,next);
    if(!enabled){for(const tween of this.tweens)tween.apply(1);this.tweens=[];return Promise.resolve();}
    const scale=current.getScreenCTM().a;
    for(const node of current.querySelectorAll('[data-motion-key]')){
      const previous=before.get(node.dataset.motionKey),box=node.getBoundingClientRect();
      if(node.tagName.toLowerCase()==='path'){
        if(!previous){const length=node.getTotalLength();animations.push(play(node,[{opacity:0,strokeDasharray:length,strokeDashoffset:length},{opacity:1,strokeDasharray:length,strokeDashoffset:0}],{duration:420,delay:80}));}
        continue;
      }
      if(previous){
        const x=(previous.box.left-box.left)/scale,y=(previous.box.top-box.top)/scale;
        const sx=box.width?previous.box.width/box.width:1,sy=box.height?previous.box.height/box.height:1;
        if(Math.abs(x)+Math.abs(y)+Math.abs(sx-1)+Math.abs(sy-1)>.1){
          node.style.transformBox='fill-box';node.style.transformOrigin='0 0';
          animations.push(play(node,[{transform:`translate(${x}px,${y}px) scale(${sx},${sy})`},{transform:'none'}],{duration}));
        }
      }else{
        const stage=Number(node.dataset.stage??0),row=Number(node.dataset.sampleIndex??0);
        animations.push(play(node,[{opacity:0,transform:'translate(14px,0)'},{opacity:1,transform:'none'}],{duration:360,delay:60+stage*35+row*7}));
      }
    }
    const removed=[...before].filter(([key])=>!used.has(key));
    if(removed.length){
      const layer=document.createElementNS(SVG_NS,'svg');layer.setAttribute('viewBox',previousViewBox);
      layer.dataset.motionExit='';layer.setAttribute('aria-hidden','true');layer.classList.add('motion-exit-layer');layer.style.height=previousHeight+'px';
      for(const [,old] of removed){const clone=old.node.cloneNode(true);clone.removeAttribute('data-motion-key');clone.querySelectorAll('[tabindex]').forEach(node=>node.removeAttribute('tabindex'));clone.removeAttribute('tabindex');if(old.transform!=='none')clone.style.transform=old.transform;layer.append(clone);}
      this.host.append(layer);animations.push(play(layer,[{opacity:.8},{opacity:0,transform:'translateX(-12px)'}],{duration:210}).then(()=>layer.remove()));
    }
    const afterHeight=current.getBoundingClientRect().height;
    if(Math.abs(previousHeight-afterHeight)>1){
      animations.push(play(this.host,[{height:previousHeight+'px'},{height:afterHeight+'px'}],{duration}));
    }
    if(this.tweens.length){
      const tweens=this.tweens,start=performance.now();
      animations.push(new Promise(resolve=>{
        this.finishTweens=resolve;
        const frame=now=>{const t=Math.min(1,(now-start)/timings.number);for(const tween of tweens)tween.apply(ease(t));if(t<1)this.frame=requestAnimationFrame(frame);else{this.frame=0;this.tweens=[];this.finishTweens=null;resolve();}};
        this.frame=requestAnimationFrame(frame);
      }));
    }
    const pending=Promise.all(animations);this.pending=pending;
    pending.then(()=>{if(this.pending===pending)this.host.dataset.motionState='idle';});return pending;
  }
}

export async function traceFlow(host,stage,signal,{direction='forward',duration=timings.flow,keys=null}={}){
  if(signal.aborted||motionPreference.matches)return !signal.aborted;
  const svg=host.querySelector('.computation-svg');
  const paths=[...svg.querySelectorAll(`[data-flow-stage="${stage}"]`)].filter(path=>(!keys||keys.includes(path.dataset.motionKey))&&(direction!=='backward'||path.dataset.gradientEdge==='true'&&path.dataset.clipped!=='true'));
  const layer=document.createElementNS(SVG_NS,'g');layer.classList.add('flow-particles');layer.setAttribute('aria-hidden','true');svg.append(layer);
  svg.dataset.playingStage=stage;
  const moving=[];
  for(const path of paths){
    const repeat=Number(path.dataset.flowRepeat??1),length=path.getTotalLength();
    for(let i=0;i<repeat;i++){
      const dot=document.createElementNS(SVG_NS,'circle');dot.setAttribute('r','3.4');dot.setAttribute('fill',path.getAttribute('stroke'));dot.setAttribute('stroke','white');dot.setAttribute('stroke-width','1.3');layer.append(dot);
      moving.push({dot,path,length,delay:Number(path.dataset.flowOrder??0)*70+i*24});
    }
    const ink=path.cloneNode();ink.removeAttribute('data-motion-key');ink.classList.add('flow-ink');ink.setAttribute('stroke-width','2.6');ink.style.strokeDasharray=length;ink.style.strokeDashoffset=length;layer.prepend(ink);
    play(ink,[{strokeDashoffset:length,opacity:0},{strokeDashoffset:0,opacity:.75,offset:.75},{strokeDashoffset:0,opacity:0}],{duration,delay:Number(path.dataset.flowOrder??0)*70});
  }
  const start=performance.now();let frame;
  const tick=now=>{for(const {dot,path,length,delay} of moving){const t=Math.max(0,Math.min(1,(now-start-delay)/(duration-160)));const p=path.getPointAtLength(length*(direction==='backward'?1-ease(t):ease(t)));dot.setAttribute('cx',p.x);dot.setAttribute('cy',p.y);dot.style.opacity=t<=0||t>=1?'0':'1';}frame=requestAnimationFrame(tick);};
  frame=requestAnimationFrame(tick);
  const completed=await waitForMotion(duration+Math.max(0,...moving.map(x=>x.delay)),signal);
  cancelAnimationFrame(frame);layer.remove();delete svg.dataset.playingStage;
  return completed;
}

let lastDetailOrigin=null,closingDetail=null;
export function openDetail(dialog,mutate,origin){
  closingDetail=null;
  const wasOpen=dialog.open,oldHeight=wasOpen?dialog.getBoundingClientRect().height:0;
  const oldBody=dialog.querySelector('#dialog-body');
  const oldContent=wasOpen?oldBody.cloneNode(true):null;
  dialog.getAnimations().forEach(animation=>animation.cancel());mutate();
  if(!wasOpen)dialog.showModal();
  const box=dialog.getBoundingClientRect();lastDetailOrigin=origin??lastDetailOrigin;
  if(!wasOpen){
    const point=lastDetailOrigin?`${lastDetailOrigin.left+lastDetailOrigin.width/2-box.left}px ${lastDetailOrigin.top+lastDetailOrigin.height/2-box.top}px`:'50% 50%';
    dialog.style.transformOrigin=point;
    play(dialog,[{opacity:0,transform:'scale(.94)'},{opacity:1,transform:'scale(1)'}],{duration:timings.detail});
    play(oldBody,[{opacity:0,transform:'translateY(9px)'},{opacity:1,transform:'none'}],{duration:260,delay:60});
  }else{
    dialog.querySelectorAll('.detail-exit').forEach(node=>node.remove());
    oldContent.removeAttribute('id');oldContent.setAttribute('aria-hidden','true');oldContent.classList.add('detail-exit');oldBody.before(oldContent);
    play(oldContent,[{opacity:.8,transform:'translateY(0)'},{opacity:0,transform:'translateY(-8px)'}],{duration:150}).then(()=>oldContent.remove());
    play(oldBody,[{opacity:0,transform:'translateY(10px)'},{opacity:1,transform:'none'}],{duration:260,delay:80});
    play(dialog.querySelector('h2'),[{opacity:.35,transform:'translateY(3px)'},{opacity:1,transform:'none'}],{duration:220});
    if(Math.abs(oldHeight-box.height)>1)play(dialog,[{height:oldHeight+'px'},{height:box.height+'px'}],{duration:320});
  }
}
export async function closeDetail(dialog){
  if(!dialog.open)return;
  const request={};closingDetail=request;
  await play(dialog,[{opacity:1,transform:'scale(1)'},{opacity:0,transform:'scale(.97)'}],{duration:180});
  if(closingDetail===request){dialog.close();closingDetail=null;}
}

let surfaceTransition=null;
export function switchSurface(update){
  if(motionPreference.matches){update();return Promise.resolve();}
  surfaceTransition?.skipTransition();
  const transition=document.startViewTransition(update);surfaceTransition=transition;
  transition.finished.then(()=>{if(surfaceTransition===transition)surfaceTransition=null;});
  return transition.updateCallbackDone;
}
