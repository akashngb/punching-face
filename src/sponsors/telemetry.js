// Turns raw contacts into what a coach and a scoreboard need: who, where, how hard, how often.
// Pure and clock-injected, so Node tests cover it. Coordinates are head-local metres with the
// face toward +z; "L"/"R" are from the puncher's point of view, which is how a coach talks.

export function zoneOf(point){
  const [x,y]=point,side=x<0?'L':'R',centre=Math.abs(x)<.022;
  if(y>.045)return 'forehead';
  if(y>.015)return centre?'brow':'eye-'+side;
  if(y>-.03)return centre?'nose':'cheek-'+side;
  if(y>-.06)return Math.abs(x)<.03?'mouth':'jaw-'+side;
  return centre?'chin':'jaw-'+side;
}

export class RoundStats{
  constructor({windowMs=30000,comboMs=1500,comboSize=3}={}){Object.assign(this,{windowMs,comboMs,comboSize,events:[],totals:new Map()});}
  // Returns the triggers this punch caused, e.g. ['combo'] or ['personal-best'].
  add({id='host',name='You',speed=0,point=[0,0,0],side=null,time}){
    const event={id,name,speed,zone:zoneOf(point),side:side||(point[0]<0?'left':'right'),time};
    const total=this.totals.get(id)||{name,count:0,max:0};const triggers=[];
    if(speed>total.max+.15&&total.count>=3)triggers.push('personal-best');
    total.name=name;total.count++;total.max=Math.max(total.max,speed);this.totals.set(id,total);
    this.events.push(event);this.trim(time);
    const recent=this.events.filter(e=>e.id===id&&time-e.time<=this.comboMs);
    if(recent.length===this.comboSize)triggers.push('combo');
    this.last=event;return triggers;
  }
  trim(now){while(this.events.length&&now-this.events[0].time>this.windowMs)this.events.shift();}
  // The JSON the coach relay turns into a telemetry paragraph. Small on purpose: numbers, no media.
  snapshot(now,trigger){
    this.trim(now);const people=new Map();
    for(const e of this.events){
      const p=people.get(e.id)||{name:e.name,count:0,sum:0,max:0,left:0,right:0,zones:{}};
      p.count++;p.sum+=e.speed;p.max=Math.max(p.max,e.speed);p[e.side]++;p.zones[e.zone]=(p.zones[e.zone]||0)+1;people.set(e.id,p);
    }
    const participants=[...people.values()].map(({sum,...p})=>({...p,avg:+(sum/p.count).toFixed(2),max:+p.max.toFixed(2)})).sort((a,b)=>b.count-a.count);
    return {participants,last:this.last?{name:this.last.name,zone:this.last.zone,speed:+this.last.speed.toFixed(2)}:null,trigger:trigger||null};
  }
  scoreboard(){return [...this.totals.entries()].map(([id,t])=>({id,name:t.name,count:t.count,max:+t.max.toFixed(1)})).sort((a,b)=>b.count-a.count);}
}
