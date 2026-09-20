// Guest adapter for the shared strike lifecycle. Landmarks are projected into
// the same mirrored 2.5D interaction space; only the compact event crosses the network.
import {fistScore,sweptEllipsoid} from '../physics.js';
import {StrikeTracker} from '../strike-system.js';

const PALM=[0,5,9,13,17],TARGET=[0,0,-.415],RADII=[.28,.25,.09];
const clamp=value=>Math.max(-1,Math.min(1,value));

export class PunchDetector{
  constructor({minSpeed=.9,hookSpeed=1.2,cooldownMs=450,aspect=16/9,diagnostics=false}={}){
    Object.assign(this,{minSpeed,hookSpeed,cooldownMs,aspect,diagnostics,guards:new Map()});
    this.tracker=new StrikeTracker({startSpeed:minSpeed,minInward:.12,minFist:.5,maxGapMs:250,rearmMs:cooldownMs,rearmDistance:.05,maxRecoverMs:cooldownMs*2});
    this.hands=this.tracker.hands;
  }
  update(key,landmarks,timeMs){
    if(!landmarks?.length||!Number.isFinite(timeMs))return null;
    const cx=PALM.reduce((sum,index)=>sum+landmarks[index].x,0)/PALM.length,cy=PALM.reduce((sum,index)=>sum+landmarks[index].y,0)/PALM.length;
    const scale=Math.hypot((landmarks[9].x-landmarks[0].x)*this.aspect,landmarks[9].y-landmarks[0].y);
    const previous=this.tracker.hands.get(key);
    if(!this.guards.has(key)||previous&&timeMs-previous.time>250)this.guards.set(key,scale);
    const guard=Math.max(this.guards.get(key),.02),ratio=scale/guard;
    const position=[(.5-cx)*this.aspect*.85,(.5-cy)*.65,-(.30+(ratio-1)*.45)];
    const motionTarget=ratio>1.08?[position[0],position[1],TARGET[2]]:[0,position[1],position[2]];
    const event=this.tracker.update({hand:key,position,target:motionTarget,timestamp:timeMs,closed:fistScore(landmarks),confidence:1},({from,to,radius})=>{
      const t=sweptEllipsoid(from,to,TARGET,RADII,radius);if(t===null)return null;
      const point=from.map((value,index)=>value+(to[index]-value)*t),travel=to.map((value,index)=>value-from[index]),magnitude=Math.hypot(...travel)||1;
      return {point,direction:travel.map(value=>value/magnitude),region:'face'};
    });
    if(!event)return null;
    const hook=event.mode==='hook',u=clamp((.5-cx)*2.2),v=clamp((.5-cy)*1.6);
    const compact={u,v,lateral:hook?clamp(event.direction[0]):0,speed:+Math.min(4,event.speed).toFixed(2),side:cx<.5?'right':'left',kind:hook?'hook':'straight'};
    if(!this.diagnostics)return compact;
    return {...compact,screenX:1-cx,screenY:cy,direction:{x:event.direction[0],y:event.direction[1],z:event.direction[2]}};
  }
  forget(key){this.tracker.reset(key);this.guards.delete(key);}
}
