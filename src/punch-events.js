// Punch event extraction — the decision core of the target-perspective pipeline.
//
// See docs/target-cv-pipeline.md for the full architecture. The short version: every quantity the
// app needs (impact point, contact velocity, force direction, punch type) is a property of a whole
// trajectory, so this module never decides anything from a single frame. Observations accumulate
// in per-slot buffers; a retrospective extractor finds apexes of the fitted range signal a few
// frames behind real time; each accepted apex is solved once, from the full window, into one
// event. Duplicates and retraction-firing are excluded by the shape of that process, not by
// stacked refractory timers:
//
//   - a punch is approach -> apex -> retreat, and the event is emitted at the apex, exactly once,
//     with the retreat acting as the *confirmation* of the punch rather than a motion that has to
//     be filtered out;
//   - emitted events consume their evidence, so the same samples cannot support a second event;
//   - candidate arbitration merges other views of the same strike two ways: overlapping image
//     support (one fist under two identities at the same spot), and temporal exclusivity — blur
//     can displace the image position arbitrarily far between two sightings of one fist, but it
//     cannot make one fist appear twice at once, so identities that never coexisted on screen are
//     one fist unless each independently observed a complete punch.
//
// Everything here is pure data-in/data-out: no DOM, no THREE, no camera plumbing. Units: metres,
// m/s, milliseconds. Camera frame: x right (+u), y up, z = distance in front of the lens
// (positive). Head frame (what events report): the scene's — x is the puncher's screen-right,
// +z faces the puncher; the single x negation between the two lives in `toHead` and nowhere else.

const clamp=(value,min,max)=>Math.min(max,Math.max(min,value));
const finite3=value=>Array.isArray(value)&&value.length===3&&value.every(Number.isFinite);
const hyp3=(x,y,z)=>Math.hypot(x,y,z);

// Camera frame -> head frame. The target camera's image x runs opposite the scene's (getUserMedia
// is unmirrored; the puncher's right hand sits at low u and must land at scene +x). This is a fact
// of the setup, applied in exactly one place.
export const toHead=([x,y,z])=>[-x,y,z];

/**
 * Where the punch lands: the FIRST point of its measured trajectory that touches the face — never
 * where the fist happened to stop. `points` is the in-contact-range slice of the fitted lateral
 * path, head frame, time-ascending. Returns [x, y, z] on the head surface, or null for a miss.
 *
 * The caller has already gated the path on RANGE (only the part of the flight within ~a fist of
 * its closest approach is eligible — a jab converging from a low guard crosses the silhouette
 * laterally half a metre out, where nothing can touch), so what remains is pure first-contact:
 * walk the path forward in time and take the first point inside the silhouette. A hook with
 * follow-through therefore lands on the cheek it CAME IN through, not wherever past the
 * centreline its fist decelerated — an earlier version that solved from the stopping point put
 * hooks on the wrong side of the face. Crossings are interpolated onto the true silhouette; a
 * path that skims within `graze` of it snaps on (the fist is ~10 cm wide); the silhouette is
 * inflated a little for the inside test, and points in the margin clamp so markers stay on the
 * head. The entry's z is the front hemisphere at that spot — the camera measures (x, y) superbly
 * and depth terribly, so depth never decides anything here.
 *
 * `arrival` (a head-frame lateral direction) handles TRUNCATED evidence: motion blur eats the
 * entry-side flight of real hooks, so the measured path often BEGINS at or past the centreline —
 * there is nothing to walk, and reporting the first measured point puts the marker on the
 * follow-through side. Physically the face would have stopped the fist where its trajectory
 * first crossed the silhouette, so when the path was never observed outside the silhouette and
 * the caller vouches that this is a lateral punch, the entry is reconstructed by walking
 * BACKWARD from the first measured point along the arrival direction to the silhouette. The
 * caller only passes `arrival` for punches classified lateral with truncated coverage; a
 * fully-observed path takes the genuine crossing and never reconstructs.
 */
export function firstContact(points,radii,{inflate=1.12,graze=.05,arrival=null}={}){
  const [rx,ry,rz]=radii;
  const frontZ=(x,y)=>rz*Math.sqrt(Math.max(0,1-(x*x)/(rx*rx)-(y*y)/(ry*ry)));
  const clampToSilhouette=(x,y)=>{
    const q=(x*x)/(rx*rx)+(y*y)/(ry*ry);
    if(q<=1)return [x,y];
    const s=1/Math.sqrt(q);return [x*s,y*s];
  };
  const finish=(x,y)=>{const [cx,cy]=clampToSilhouette(x,y);return [cx,cy,frontZ(cx,cy)];};
  const backwardEntry=point=>{
    const length=Math.hypot(arrival[0],arrival[1]);
    if(!(length>1e-6))return null;
    const dx=arrival[0]/length,dy=arrival[1]/length;
    const a=(dx*dx)/(rx*rx)+(dy*dy)/(ry*ry);
    const b=2*((point[0]*dx)/(rx*rx)+(point[1]*dy)/(ry*ry));
    const c=(point[0]*point[0])/(rx*rx)+(point[1]*point[1])/(ry*ry)-1;
    const disc=b*b-4*a*c;
    if(!(a>1e-12)||disc<0)return null;
    const s=(-b-Math.sqrt(disc))/(2*a);
    if(s>=0)return null;              // no crossing behind the point along the arrival line
    return finish(point[0]+dx*s,point[1]+dy*s);
  };
  let previous=null,nearest=null,nearestGap=Infinity,first=true;
  for(const point of points){
    if(!point||!Number.isFinite(point[0])||!Number.isFinite(point[1]))continue;
    const q=(point[0]*point[0])/(rx*rx*inflate*inflate)+(point[1]*point[1])/(ry*ry*inflate*inflate);
    if(q<=1){
      // The very first valid point is already inside: the entry-side flight was never observed.
      // With an arrival direction vouched for, reconstruct where the face would have stopped it.
      if(first&&arrival){
        const reconstructed=backwardEntry(point);
        if(reconstructed)return reconstructed;
      }
      // Entered the silhouette. If the previous point was outside, place the entry where the
      // segment crosses the TRUE silhouette rather than however deep this sample already sits.
      if(previous){
        const dx=point[0]-previous[0],dy=point[1]-previous[1];
        const a=(dx*dx)/(rx*rx)+(dy*dy)/(ry*ry);
        const b=2*((previous[0]*dx)/(rx*rx)+(previous[1]*dy)/(ry*ry));
        const c=(previous[0]*previous[0])/(rx*rx)+(previous[1]*previous[1])/(ry*ry)-1;
        const disc=b*b-4*a*c;
        if(a>1e-12&&disc>=0){
          const t=(-b-Math.sqrt(disc))/(2*a);
          if(t>=0&&t<=1)return finish(previous[0]+dx*t,previous[1]+dy*t);
        }
      }
      return finish(point[0],point[1]);
    }
    // Track the closest miss for the graze snap.
    const gap=(Math.sqrt(q)-1)*Math.min(rx,ry)*inflate;
    if(gap<nearestGap){nearestGap=gap;nearest=point;}
    previous=point;first=false;
  }
  if(nearest&&nearestGap<=graze)return finish(nearest[0],nearest[1]);
  return null;
}

// --- local weighted fits ---------------------------------------------------------------------
// A quadratic in time, weighted least squares. Derivatives of punch trajectories come from these
// fits rather than from a causal filter: there is no lag/overshoot trade to tune, a single noisy
// frame cannot cross a threshold, and the extractor runs behind real time anyway so the future
// half of the window exists. Falls back to linear, then constant, as samples thin out.
function quadFit(times,values,weights,t0){
  let n=0;for(const w of weights)if(w>0)n++;
  const sums=new Float64Array(5),m=new Float64Array(3);
  for(let i=0;i<times.length;i++){
    const w=weights[i];if(!(w>0))continue;
    const dt=(times[i]-t0)/1000,y=values[i];
    let p=w;
    sums[0]+=p;m[0]+=p*y;p*=dt;
    sums[1]+=p;m[1]+=p*y;p*=dt;
    sums[2]+=p;m[2]+=p*y;p*=dt;
    sums[3]+=p;p*=dt;
    sums[4]+=p;
  }
  if(n>=3){
    const A=[sums[0],sums[1],sums[2],sums[1],sums[2],sums[3],sums[2],sums[3],sums[4]];
    const det=A[0]*(A[4]*A[8]-A[5]*A[7])-A[1]*(A[3]*A[8]-A[5]*A[6])+A[2]*(A[3]*A[7]-A[4]*A[6]);
    if(Number.isFinite(det)&&Math.abs(det)>1e-12){
      const inv=[
        (A[4]*A[8]-A[5]*A[7])/det,(A[2]*A[7]-A[1]*A[8])/det,(A[1]*A[5]-A[2]*A[4])/det,
        (A[5]*A[6]-A[3]*A[8])/det,(A[0]*A[8]-A[2]*A[6])/det,(A[2]*A[3]-A[0]*A[5])/det,
        (A[3]*A[7]-A[4]*A[6])/det,(A[1]*A[6]-A[0]*A[7])/det,(A[0]*A[4]-A[1]*A[3])/det,
      ];
      const a=inv[0]*m[0]+inv[1]*m[1]+inv[2]*m[2];
      const b=inv[3]*m[0]+inv[4]*m[1]+inv[5]*m[2];
      const c=inv[6]*m[0]+inv[7]*m[1]+inv[8]*m[2];
      if(Number.isFinite(a)&&Number.isFinite(b)&&Number.isFinite(c))return {value:a,slope:b,curve:c};
    }
  }
  if(n>=2){
    const det=sums[0]*sums[2]-sums[1]*sums[1];
    if(Number.isFinite(det)&&Math.abs(det)>1e-12){
      const a=(sums[2]*m[0]-sums[1]*m[1])/det,b=(sums[0]*m[1]-sums[1]*m[0])/det;
      if(Number.isFinite(a)&&Number.isFinite(b))return {value:a,slope:b,curve:0};
    }
  }
  if(n>=1&&sums[0]>0)return {value:m[0]/sums[0],slope:0,curve:0};
  return null;
}

const tricube=x=>{const t=1-Math.abs(x)**3;return t>0?t*t*t:0;};

// Weighted LINEAR slope of a scalar series at t0 — for measuring speeds on sparse samples. The
// quadratic localFit is right for trajectories (it follows curvature) but wrong for sparse noisy
// speed reads: three points 100 ms apart fit a quadratic EXACTLY, so 1 cm of depth noise on the
// middle point becomes metres-per-second of phantom slope. A line through the same window is
// stable by construction and reads the windowed mean speed — an honest, slightly conservative
// figure that a gate can be calibrated against.
export function linearSlope(samples,pick,t0,halfMs){
  let sw=0,st=0,sy=0,stt=0,sty=0,n=0;
  for(const s of samples){
    const dt=s.t-t0;
    if(Math.abs(dt)>halfMs)continue;
    const w=s.w*tricube(dt/halfMs),x=dt/1000,y=pick(s);
    if(!(w>0))continue;
    sw+=w;st+=w*x;sy+=w*y;stt+=w*x*x;sty+=w*x*y;n++;
  }
  if(n<2)return null;
  const det=sw*stt-st*st;
  if(!Number.isFinite(det)||Math.abs(det)<1e-12)return null;
  return (sw*sty-st*sy)/det;
}

// Fitted value + derivative of a scalar series at t0, over a +/-halfMs tricube window.
export function localFit(samples,pick,t0,halfMs){
  const times=[],values=[],weights=[];
  for(const s of samples){
    const dt=s.t-t0;
    if(Math.abs(dt)>halfMs)continue;
    times.push(s.t);values.push(pick(s));weights.push(s.w*tricube(dt/halfMs));
  }
  return quadFit(times,values,weights,t0);
}

// Position and velocity of the trajectory at one instant, each axis from its own local weighted
// quadratic. Local fits rather than one window-wide polynomial: a punch's speed envelope swings
// from metres-per-second to zero and back inside the terminal window, and a single quadratic is
// too stiff to carry the rotating velocity of a hook through that — it reports the chord. A local
// fit at each probe time follows the arc.
function stateAt(samples,t,halfMs){
  const axes=[0,1,2].map(k=>localFit(samples,s=>s.p[k],t,halfMs));
  if(axes.some(a=>!a))return null;
  return {position:axes.map(a=>a.value),velocity:axes.map(a=>a.slope)};
}

// --- classification --------------------------------------------------------------------------
// Punch type from full-window evidence: the terminal travel direction, where the knuckles face
// (averaged over the window — orientation is steadier than instantaneous travel), how much the
// velocity direction rotated through the terminal window (hooks arc; jabs do not), where the
// approach began, and the punch's net IMAGE sweep. The sweep is the blur-proof feature: a hook
// crosses a third of the frame laterally even when every depth estimate is garbage and the
// terminal tangent degraded to a chord, so it is what keeps a censored hook a hook. All head
// frame; sweep in head-frame image units (+du = head +x, +dv = up).
export function classifyMode({direction,knuckleNormal=null,entry=null,curvature=0,sweep=null,handAxis=null,range=null},detail=null){
  const verdict=(mode,why)=>{if(detail)detail.why=why;return mode;};
  // WHERE THE FIST POINTS, measured in the image, in fist-widths. This is the jab/uppercut
  // discriminator and it is checked first, because the two punches differ most obviously in
  // exactly this: a jab arrives with its knuckles square at the camera, so the wrist->knuckle
  // axis points along the view axis and FORESHORTENS to almost nothing; an uppercut arrives with
  // its knuckles up, so the same axis stands tall up the frame. Both readings are pure image
  // geometry — immune to the depth compression that made the 3D knuckle normal tilt a jab's
  // axis "up" and read it as an uppercut (and, the other way, buried real uppercuts whose
  // travel was truncated). Measured on real hand geometry: a fist aimed at the camera reads
  // 0.0-0.25 fist-widths, a true uppercut 0.4-0.9 (the absolute figure shrinks with perspective
  // as the fist nears the lens, which is why the bands are set from the two CLEAR cases). The
  // middle is left UNDECIDED on purpose — an ambiguous orientation defers to travel evidence
  // rather than inventing a type from it.
  //
  // WHICH FACE of the hand shows is the other half of the shape (`facing`: knuckle-row span
  // relative to hand scale). A guard fist is ALSO vertical — same axis as an uppercut — but it
  // is seen EDGE-ON: back of the fist to the side, so the knuckle row points away from the
  // camera and its apparent span collapses (measured: guard .11, even 20° off edge .40, every
  // face-on punch 1.0). An uppercut and a hook both show the BACK of the fist; guard shows its
  // edge. So no orientation verdict from an edge-on hand — idling fists in guard were firing
  // "uppercut" purely on their vertical axis. And a face-on fist with its knuckles to the SIDE
  // is a hook's hand shape: corroborating image evidence when blur ate the sweep.
  const facing=handAxis?.facing??1;
  const faceOn=facing>=.55,edgeOn=!!handAxis&&facing<.55;
  const axisLen=handAxis?Math.hypot(handAxis.du,handAxis.dv):0;
  const axisAimed=handAxis&&axisLen<.32;                       // knuckles at the camera
  const axisUp=handAxis&&faceOn&&handAxis.dv>.40&&handAxis.dv>Math.abs(handAxis.du)*1.3;
  const axisDown=handAxis&&faceOn&&handAxis.dv<-.40&&-handAxis.dv>Math.abs(handAxis.du)*1.3;
  const axisSide=handAxis&&faceOn&&Math.abs(handAxis.du)>.55&&Math.abs(handAxis.du)>Math.abs(handAxis.dv)*1.3;
  const d=direction,n=finite3(knuckleNormal)?knuckleNormal:null;
  const fromBelow=entry&&Number.isFinite(entry[1])&&entry[1]<-.16;
  const su=Math.abs(sweep?.du??0),sv=Math.abs(sweep?.dv??0);
  const sweepHook=su>.16&&su>sv*1.3;
  const sweepUp=(sweep?.dv??0)>.12&&sv>su;
  const lat=Math.abs(d[0]);
  // The image plane is what the target camera measures BEST, so where the measured sweep and the
  // fitted 3D direction disagree, the sweep wins. A hook translates across the image; an
  // uppercut climbs it; a jab barely moves — it looms. The fitted lateral, by contrast, is
  // depth-starved exactly at contact: when the rigid fit dies and depth goes inertial, the
  // velocity's z collapses and the terminal direction degrades to pure lateral/vertical noise —
  // which read straight jabs as hooks and uppercuts. Hence: no hook verdict without image
  // support, and upward image motion outranks every orientation prior. The bar sits at .06:
  // depth-starved lateral noise moves the image a few hundredths at most, while a truncated hook
  // still sweeps ≥ ~.09 even when the apex rewind has collapsed its window (a flat-range hook
  // "arrives at depth" almost immediately while its lateral sweep is still going). An off-centre
  // jab's perspective drift can exceed the bar, but its fitted lateral stays small, so no hook
  // branch takes it anyway.
  const hookEvidence=!sweep||su>=.06||axisSide;
  // Knuckles square at the camera is a straight punch, never an uppercut — an uppercut's
  // knuckles point UP at contact. The veto only needs the fist to be measured near face-on with
  // no vertical tilt; every genuine uppercut carrier requires or shows n.y well above this.
  // Knuckles pointing more FORWARD than up is a straight punch, never an uppercut. The old veto
  // band (n.y < .35) left a gap: ordinary jab form angles the fist up 30-50°, so n.y lands at
  // .45-.75 — unvetoed, and close enough to the orientation carrier to tip a jab into 'uppercut'
  // on noise. An uppercut at contact has its knuckles genuinely UP: n.y must at least exceed
  // |n.z|, which no face-on fist does however tilted.
  const knAtCamera=axisAimed||(!handAxis&&!!n&&n[1]<Math.abs(n[2]));
  // Measured orientation decides the vertical question outright, before any travel evidence is
  // consulted: an uppercut cannot arrive with its knuckles aimed at the camera, and a straight
  // punch cannot arrive with them standing up the frame. Overwhelming lateral travel still wins
  // first — a rising hook is a hook — and the axis must not itself be lying sideways.
  // ...but orientation may not overrule clean travel. Near the lens the fist fills the frame and
  // perspective is extreme: the wrist is much farther from the camera than the knuckles, so it
  // projects low in the image and the measured axis "stands up" even for a dead-straight jab
  // (observed live at 10 cm range, logged as LEFT UPPERCUT while the knuckles faced the camera).
  // So when the punch is measurably driving straight in, orientation needs corroboration from
  // some independent sign of a rise — entry from below, or an upward image sweep — before it can
  // call an uppercut. A real uppercut supplies one of those even when blur truncates its travel.
  // ONE invariant governs every uppercut verdict below: the punch must show IMAGE-PLANE evidence
  // that it rose — an upward sweep across the frame, or an entry from below the head. Nothing
  // depth-derived may stand in for it. Three separate live misclassifications all had the same
  // shape: a straight jab at close range, where the fitted direction, the 3D knuckle normal and
  // (through perspective) even the image hand axis all tilt "up", while the punch demonstrably
  // never rose in the frame. The fist at 14 cm fills the view and its wrist projects far below
  // its knuckles; a fragmented punch's censored direction degrades the same way. Image rise is
  // the one signal that stays honest through all of it — and a genuine uppercut always has it,
  // because rising up the frame is what an uppercut IS. The hand axis strengthens the verdict
  // but can never substitute for it.
  //
  // Orientation may still carry a truncated uppercut — but only from far enough away to be
  // believed. Inside ~25 cm the fist subtends a huge angle, its wrist sits far behind its
  // knuckles, and every orientation reading tilts upward whatever the punch is doing; that is
  // precisely where the live misfires happened (10-14 cm). Beyond it, orientation is sound and a
  // top-only uppercut whose rise happened below the frame can still be recognised.
  const rose=!sweep||sweepUp||fromBelow;
  const closeRange=Number.isFinite(range)&&range<.25;
  const orientationUp=handAxis?axisUp:(!!n&&n[1]>.75);
  const upOk=rose||(!closeRange&&orientationUp);
  if(axisUp&&upOk&&!(lat>.60&&sweepHook))return verdict('uppercut','fist-points-up');
  if(axisDown&&Math.abs(d[0])<.60&&(sweep?.dv??0)<-.08)return verdict('overhand','fist-points-down');
  if(!knAtCamera&&!edgeOn&&sweepUp&&sv>.15&&(sv>su*1.5||lat<.50)&&d[1]>-.25)return verdict('uppercut','image-rise');
  // Overwhelming lateral travel is a hook before any other prior gets a look — a rising hook must
  // not fall into an uppercut branch on its upward tilt.
  if(lat>.60&&d[1]<.60&&hookEvidence)return verdict('hook','lateral-travel');
  // The hook's hand shape: back of the fist to the camera, knuckles to the SIDE. With lateral
  // travel behind it, the shape alone carries the verdict — the case where the apex rewind
  // collapsed the sweep and hooks were falling through to 'jab'.
  if(axisSide&&lat>.30&&!sweepUp)return verdict('hook','fist-points-side');
  if(n){
    // Moderate knuckles-up plus rising travel is an uppercut only when the IMAGE saw some rise
    // too: ordinary jab form angles the fist upward and depth noise tilts the fitted direction,
    // and the pair of them read straight jabs as uppercuts.
    if(!knAtCamera&&!edgeOn&&upOk&&n[1]>.45&&d[1]>.25&&lat<.55&&!sweepHook)return verdict('uppercut','knuckles+rise');
    // Orientation may carry a truncated uppercut on its own: the rise is low, fast, blurred and
    // often below the frame, so the landmarker locks on only at the top where the upward velocity
    // is already spent — the measured direction reads near-pure -z and every travel-gated branch
    // fails, while the fist's ORIENTATION is measured at the apex, exactly where tracking is
    // good. But only EMPHATIC knuckles-up qualifies: ordinary jab form angles the fist upward
    // too (n.y ~.5-.7 for a fist punched at 45°), and at the old .55 threshold real jabs read as
    // uppercuts. At .75 only a genuinely vertical fist carries the class alone, and the image
    // must at least not contradict.
    if(upOk&&!edgeOn&&n[1]>.75&&d[1]>-.15&&lat<.55&&!sweepHook)return verdict('uppercut','knuckles-vertical');
    if(n[1]<-.50&&d[1]<-.20)return verdict('overhand','knuckles-down');
    if(Math.abs(n[0])>.50&&lat>.30&&hookEvidence)return verdict('hook','knuckles-side');
  }
  if(!knAtCamera&&!edgeOn&&upOk&&d[1]>.45&&d[1]>lat*.9&&!sweepHook)return verdict('uppercut','rising-travel');
  // The entered-from-below prior needs corroboration beyond a low guard: rising travel that is
  // not strongly lateral, and an image sweep that is actually vertical.
  if(!knAtCamera&&!edgeOn&&fromBelow&&d[1]>.30&&lat<.50&&(sweepUp||sv>=su))return verdict('uppercut','from-below');
  if(lat>.45&&hookEvidence)return verdict('hook','lateral');
  if(lat>.30&&(curvature>.2||sweepHook)&&hookEvidence)return verdict('hook','arc');
  if(sweepHook&&lat>.22)return verdict('hook','image-sweep');
  return verdict('jab','default-straight');
}

/**
 * Which of the puncher's hands threw this. Physically grounded geometry outvotes MediaPipe's
 * per-frame guess:
 *
 *   - rest lineage (strongest): where this identity RESTED before it flew. Fists guard on their
 *     own side — the puncher's right at low u — and, unlike every within-window measurement,
 *     the guard was observed while tracking was good, however badly the punch itself blurred.
 *   - elimination: a fist visibly resting somewhere else right now is not the fist that just
 *     landed. Cast only when the striker has no rest history of its own (mid-flight birth), so a
 *     duplicate identity of the striker parked at its guard cannot vote against it.
 *   - wrist trail: the forearm always exits toward its own shoulder, and the puncher's right
 *     shoulder sits at camera -x — so a windowed mean of the METRIC lateral wrist-minus-knuckles
 *     offset says which arm this is on every sample, whatever the punch type and however late the
 *     track picked it up. Metric, not image-space: the wrist is deeper than the knuckles and
 *     perspective drags deeper points toward the image centre, which read as the wrong hand for
 *     any laterally-offset fist. The strongest vote when the arm shows any lateral geometry.
 *   - metric chirality: the fitted 3D hand's own handedness (positive = right in the fit frame),
 *     the one property no rotation can disguise; averaged over the window it survives the head-on
 *     ambiguity that makes the label flicker.
 *   - entry side: only meaningful when the approach genuinely entered from an outer edge — a hook
 *     first detected mid-arc near frame centre used to cast this vote arbitrarily, which is
 *     exactly how right hooks got logged as left. The caller passes null for centre entries.
 *   - travel azimuth: a right hook drives toward the puncher's left (head-frame -x); weighted up
 *     when the punch is strongly lateral.
 *   - the landmarker's label as the weakest vote. This vendored bundle names the PHYSICAL hand
 *     on an unmirrored feed (no selfie assumption — verified live); a mirrored feed's labels
 *     are flipped at the shell boundary before they reach here.
 */
export function resolveHand({entryU=null,labels=[],direction=null,wristDx=null,chirality=null,restU=null,otherRestU=null}){
  let right=0,left=0;
  const cast=(isRight,weight)=>{if(isRight)right+=weight;else left+=weight;};
  if(Number.isFinite(restU)&&Math.abs(restU-.5)>.06)
    cast(restU<.5,1.4*Math.min(1,Math.abs(restU-.5)/.18));
  if(Number.isFinite(otherRestU)&&Math.abs(otherRestU-.5)>.06)
    cast(otherRestU>=.5,1.2*Math.min(1,Math.abs(otherRestU-.5)/.18));
  if(Number.isFinite(wristDx)&&Math.abs(wristDx)>.015)
    cast(wristDx<0,1.0*Math.min(1,Math.abs(wristDx)/.06));
  if(Number.isFinite(chirality)&&Math.abs(chirality)>.04)
    cast(chirality>0,.7*Math.min(1,Math.abs(chirality)/.12));
  if(Number.isFinite(entryU)){
    if(entryU<.42)cast(true,.8);else if(entryU>.58)cast(false,.8);
  }
  let labelScore=0,labelCount=0;
  for(const {label,score} of labels){
    // The vendored landmarker (tasks-vision 0.10.32 bundle) names the PHYSICAL hand on an
    // unmirrored feed — it does NOT apply MediaPipe's documented selfie assumption. Verified
    // live: under the selfie interpretation every overlay letter came out swapped. Mirrored
    // feeds are already label-flipped at the shell boundary before reaching here.
    if(label==='Right')labelScore+=score??.5;
    else if(label==='Left')labelScore-=score??.5;
    labelCount++;
  }
  if(labelCount&&labelScore)cast(labelScore>0,.6*Math.min(1,Math.abs(labelScore)/labelCount*2));
  if(finite3(direction)&&Math.abs(direction[0])>.3)
    cast(direction[0]<0,Math.abs(direction[0])>.6?.8:.5);
  const total=right+left;
  if(!total)return {hand:'right',confidence:.5,votes:{right,left}};
  return right>=left
    ?{hand:'right',confidence:right/total,votes:{right,left}}
    :{hand:'left',confidence:left/total,votes:{right,left}};
}

// --- tunables --------------------------------------------------------------------------------
// The public knobs are physical quantities of a punch; the internals are properties of the camera
// and of human motion, not of any filter.
export const DEFAULTS={
  startClosing:.8,    // m/s — closing speed for the instant-fire gate and "born fast" evidence
  stopClosing:.3,     // m/s — below this at the end of the data, a censored track is not mid-punch
  minPeak:1.0,        // m/s — peak measured closing speed a punch must reach. Measured as a
                      // windowed LINEAR slope over rigid samples (see #extract), which reads the
                      // punch's sustained closing rather than an instantaneous spike — honest and
                      // stable, but ~20-30% under the true peak, so the gate sits at 1.0 where the
                      // old spiky estimate sat at 1.2. Lazy reaches measure 0.2-0.5; real soft
                      // jabs 1.1-1.8.
  minTravel:.08,      // m   — range it must actually close
  contactDepth:.12,   // m   — a punch reaching this close fires immediately, without waiting for
                      // apex. Deliberately deep: a committed jab crosses 20 cm ~50-100 ms before
                      // full extension (and near-camera depth noise dips the fitted range under an
                      // early line even sooner), so a shallow trigger reported mid-flight
                      // positions and consumed the real apex. At 12 cm the crossing IS the landing.
  strikeRange:.45,    // m   — a punch ARRIVES: its apex must come within striking range of the
                      // head. Approach shape alone is not sufficient evidence — any inward
                      // lateral guard shift closes range geometrically (r=|p| shrinks by the
                      // lateral slack r−z), which fired phantom "hooks" whose apex sat 57 cm out.
                      // No jab, hook or uppercut terminates half a metre from the face.
  rearmGap:.07,       // m   — withdrawal required after a fire before the next punch can register
  maxDepth:1.6,       // m   — beyond this the "hand" is not a punch in progress
  minFist:.12,        // closure the hand must show at some point in the approach (lenient: a fist
                      // seen knuckles-on hides its own fingers and reads low)
  graceMs:160,        // ms  — samples older than this leave the trajectory censored
};
// A hook starts outside a ~60deg webcam's view and an uppercut below it; both are visible only for
// the decelerating tail of their flight. A track born against a frame border (or born already
// fast) is judged on these softened gates, because arriving from off-camera already travelling is
// itself the evidence of a punch.
export const EDGE_MARGIN=.14,EDGE_PEAK=.35,EDGE_REACH=.035;   // EDGE_PEAK rides the same
// recalibration as minPeak: the windowed-linear speed measure reads ~25% under the old spiky
// quadratic, and a truncated hook's decelerating tail measured 0.41 against the old 0.45.
// Born already closing this fast means the punch predates the track (a 20 fps hook can materialise
// inside the edge margin between two samples). Set well above guard jitter and the odd twitch —
// a genuinely mid-flight hook measures 2+ m/s at its first sample pair.
export const BORN_FAST=1.4;
const FIT_HALF_MS=90;       // half-window of the local range fit
const CONFIRM_RISE=.02;     // m of fitted range regained after the minimum = the retreat confirms
const CONFIRM_HOLD_MS=70;   // and the retreat must be SUSTAINED this long. A single fitted sample
                            // above the confirm line is one bounce of near-camera depth noise —
                            // which fired the event mid-flight, reported a point along the punch's
                            // travel, and consumed the real apex so the actual landing never
                            // registered. A real retraction clears both conditions ~2 frames later.
const WINDUP_MS=450;        // how far back the start of a punch is looked for
const IMAGE_GATE=.20,IMAGE_GATE_PER_SECOND=.9;   // association gate, growing with the sample gap
const SAME_REGION_MS=280;   // two apexes closer than this with overlapping support are one punch
const MERGE_GATE=.22;       // image distance that counts as overlapping support
const SAME_FIST_MS=400;     // a candidate this soon after a fire, whose identity never coexisted
                            // with the one that fired, is the same fist re-tracked after blur —
                            // unless both sightings independently observed a complete punch
const COEXIST_MS=60;        // landmark-span overlap that proves two identities are two objects
const REST_SPEED=.45;       // image speed (u/s) below which a fist counts as resting
const REST_RUN=8;           // consecutive slow samples before rest accrues — longer than an apex
                            // hold, much shorter than a guard
const REST_MIN_W=4;         // rest evidence required before lineage may vote on handedness
const REST_ALPHA=.12;       // EMA rate of the learned rest position
const RIGID_SUPPORT=.4;     // fraction of a candidate's claimed travel that must be witnessed by
                            // rigid-fit samples. Span-depth and blob expansion can CARRY an
                            // already-observed approach through blur, but they can never
                            // constitute one: a fist held still at the lens had its span-depth
                            // oscillate ±8 cm as fingers clipped the frame, and every fabricated
                            // dip fired an "instant" punch at superhuman speed.
const RIGID_CONFIRM_MS=150; // rigid samples this soon after the apex still witness the approach —
                            // reacquired landmarks at the start of the retraction reveal how deep
                            // the fist actually got, even when the apex itself was blob-carried
const IMMINENT_MS=120;      // a censored punch was cut mid-flight; it arrives if its trailing
                            // closing speed would have carried it into striking range this soon
const HEAD_FLOOR=1.15;      // × headRadii[1] below centre — arrivals below this are chest/shoulder
                            // height, and there is no head there to hit; ignored outright
const TERMINAL_MS=220,TERMINAL_AFTER_MS=50;      // solver window around the apex
const CONTACT_RANGE=.10;    // m of range above the closest approach within which the fist can
                            // actually touch the face — the contact-eligible slice of the flight
const APEX_TOLERANCE=.025;  // m — the contact is the FIRST arrival within this of the deepest
                            // range, not the deepest sample itself (see the rewind in #extract)
const DIR_AVG_MS=60;        // direction averages the last qualifying probes over this span
// Direction is read at the LATEST probe whose fitted velocity is still trustworthy — an absolute
// speed floor, because reliability is about signal-to-noise of that one local fit, not about where
// the punch's own peak happened to fall. The relative backstop only matters for the slow tail of
// an edge-entry punch.
const DIR_FLOOR_SPEED=.35,DIR_FLOOR_FRACTION=.15;
const CENSOR_EXTRAP_MS=60;  // how far a blur-censored apex is carried past the last sample
const BLOB_BRIDGE_MS=260;   // how long motion blobs may extend a landmark-dead trajectory
const BLOB_GATE=.24;        // image gate for a blob to count as the same object
const BUFFER_MS=1400,MAX_SLOTS=4,EVENT_MEMORY_MS=1000;

let nextSlotId=0;

// Two identities are two OBJECTS only if they were ever seen at the same time: fragments of one
// fist take turns (blur kills the first before reacquisition births the second), while two real
// fists overlap on screen. Landmark spans only — blob-bridged samples keep a dead identity's
// buffer warm without proving anyone saw the object itself.
function coexisted(a,b){
  if(a===b)return true;
  if(a.firstObsT===null||b.firstObsT===null)return false;
  return Math.min(a.lastObsT,b.lastObsT)-Math.max(a.firstObsT,b.firstObsT)>=COEXIST_MS;
}

class Slot{
  constructor(u,v,time){
    this.id=`s${++nextSlotId}`;
    this.samples=[];             // {t,u,v,p:[x,y,z],r,w,closure,kn,label,score,quality}
    this.consumedUntil=-1e9;
    this.rearm=null;             // {min} — fired; wait for r to regain rearmGap
    this.lastSeen=time;this.u=u;this.v=v;this.du=0;this.dv=0;
    this.pu=u;this.pv=v;this.pdu=0;this.pdv=0;this.pt=time;   // blob-informed predicted position
    this.firstObsT=null;this.lastObsT=null;                   // landmark span, for coexistence
    this.restU=null;this.restW=0;this.slowRun=0;              // guard lineage (see push)
    this.bornAtEdge=u<EDGE_MARGIN||u>1-EDGE_MARGIN||v<EDGE_MARGIN||v>1-EDGE_MARGIN;
    this.bornFast=false;
    this.lastFireT=null;         // windup for a later punch never reaches back across a fire
    this.labels=new Map();       // score-weighted handedness votes, for the shell's estimator keying
    this.bridged=0;              // synthetic blob samples appended since the last real sample
  }
  get label(){
    let best='hand',score=-1;
    for(const [name,tally] of this.labels)if(tally>score){best=name;score=tally;}
    return best;
  }
  // Where this identity should be at time t. Blob samples advance this state (position always,
  // velocity when the blob carried a measured flow), so the prediction follows a fist through the
  // blur that killed its landmarks instead of extrapolating the stale pre-blur velocity.
  predictAt(t){
    const dt=clamp((t-this.pt)/1000,0,.5);
    return [this.pu+this.pdu*dt,this.pv+this.pdv*dt];
  }
}

export class PunchExtractor{
  constructor(options={}){
    this.options={...DEFAULTS,...options};
    this.slots=new Map();
    this.blobs=[];               // global motion-blob buffer: {t,u,v,mass,spread,flow,expand}
    this.recent=[];              // emitted events: {t,u,v}
    this.stats={impacts:0,stale:0,merged:0,refractory:0,ignored:0,rejected:''};
    this.debug={phase:'idle',r:null,closing:0,peak:0,travel:0,minRange:null,slot:'—'};
  }
  reset(){this.slots.clear();this.blobs.length=0;this.recent.length=0;}
  configure(options){Object.assign(this.options,options);}

  // Association: detections claim the nearest slot whose predicted position they fall within;
  // the gate grows with the time gap so a dropped frame does not sever identity. Leftovers open
  // new slots — identity mistakes are contained by arbitration, never by a per-frame decision.
  assign(detections,timestamp){
    // Real capture timestamps never run backward; a large rewind means a new session (worker
    // restart, test harness re-drive). Stale future-stamped state must not adopt the new stream.
    for(const slot of this.slots.values())if(slot.lastSeen-timestamp>400){this.reset();break;}
    const open=[...this.slots.values()],pairs=[];
    for(const d of detections)for(const slot of open){
      // Position from the blob-informed prediction (it followed the arc through any blur gap);
      // gate growth from LANDMARK silence — uncertainty about identity grows while nobody has
      // actually seen the hand, however confidently the motion stream tracked something.
      const dt=clamp((timestamp-slot.lastSeen)/1000,0,.5);
      const [pu,pv]=slot.predictAt(timestamp);
      const gap=Math.hypot(d.u-pu,d.v-pv);
      if(gap<=IMAGE_GATE+IMAGE_GATE_PER_SECOND*dt)pairs.push({d,slot,gap});
    }
    pairs.sort((a,b)=>a.gap-b.gap);
    const taken=new Set(),claimed=new Set(),out=new Map();
    for(const pair of pairs){
      if(taken.has(pair.d)||claimed.has(pair.slot))continue;
      taken.add(pair.d);claimed.add(pair.slot);out.set(pair.d,pair.slot);
    }
    for(const d of detections)if(!out.has(d)){
      if(this.slots.size>=MAX_SLOTS){
        let stalest=null;
        for(const slot of this.slots.values())if(!stalest||slot.lastSeen<stalest.lastSeen)stalest=slot;
        if(stalest)this.slots.delete(stalest.id);
      }
      const slot=new Slot(d.u,d.v,timestamp);
      this.slots.set(slot.id,slot);out.set(d,slot);
    }
    return out;
  }

  /**
   * A hand was SEEN at (u,v) but could not be measured in 3D — no pose solve, and no recent fit
   * to carry depth from. Identity is updated (association prediction, liveness, rest lineage)
   * and nothing enters the trajectory.
   *
   * This separation is the whole point: being tracked and being measured are different things.
   * The old 'loose' tier guessed depth from apparent knuckle width in exactly this situation,
   * and since a punch lasts ~200 ms while "no fit for 400 ms" only happens to a fist parked at
   * the lens, it never once carried a real punch — it only fabricated approaches out of fingers
   * clipping the frame. Seeing a hand is worth remembering; inventing its depth is not.
   */
  observe(slot,{t,u,v}){
    if(!Number.isFinite(t)||!Number.isFinite(u)||!Number.isFinite(v))return;
    this.#track(slot,t,u,v);
  }

  // Identity bookkeeping shared by measured samples and see-only observations.
  #track(slot,t,u,v){
    const dt=(t-slot.lastSeen)/1000;
    if(dt>1e-3&&dt<.5){
      slot.du=(u-slot.u)/dt;slot.dv=(v-slot.v)/dt;
      // Guard lineage: a fist SUSTAINEDLY slow in the image is a fist at rest, and where it
      // rests says which hand it is (the puncher's right guards at low u) — evidence that
      // survives however badly the punch itself later blurs. The run requirement keeps the
      // apex hold out: a punch pauses at full extension for a few frames, a guard rests for many.
      if(Math.hypot(slot.du,slot.dv)<REST_SPEED){
        if(++slot.slowRun>=REST_RUN){
          slot.restU=slot.restU===null?u:slot.restU+(u-slot.restU)*REST_ALPHA;
          slot.restW=Math.min(slot.restW+1,60);
        }
      }else slot.slowRun=0;
    }
    slot.u=u;slot.v=v;slot.lastSeen=t;slot.bridged=0;
    slot.pu=u;slot.pv=v;slot.pdu=slot.du;slot.pdv=slot.dv;slot.pt=t;
    if(slot.firstObsT===null)slot.firstObsT=t;
    slot.lastObsT=Math.max(slot.lastObsT??t,t);
  }

  // One fused observation. `p` is the camera-frame metric position; `quality` says how it was
  // measured: 'rigid' from the 6-DOF pose solve (the only true measurement), 'span' from a
  // failed solve with depth carried inertially from the last rigid fit, 'blob' from the motion
  // stream. Only 'rigid' may speak about depth — see `#solve`.
  push(slot,obs){
    const {t,u,v,p,closure=1,kn=null,label=null,score=1,quality='rigid',wristDx=null,chirality=null,axis=null}=obs;
    if(!finite3(p)||!Number.isFinite(t))return;
    const r=hyp3(...p);
    if(r>this.options.maxDepth){this.stats.rejected='too far to be a punch';return;}
    // Re-arm withdrawal must be OBSERVED, by rigid-fit samples. The fitted range blends span and
    // blob evidence, and fabricated depth (a fist held at the lens, span oscillating as fingers
    // clip the frame; a body swaying behind a still fist) used to swing r far enough to clear the
    // re-arm — after which the next fabricated dip fired again, about once a second, forever.
    if(quality==='rigid'&&slot.rearm){
      slot.rearm.min=Math.min(slot.rearm.min,r);
      if(r>=slot.rearm.min+this.options.rearmGap)slot.rearm.cleared=true;
    }
    if(quality!=='blob')this.#track(slot,t,u,v);
    else{
      // The motion stream steers the slot's PREDICTED position through landmark silence. Blur
      // destroys landmarks mid-arc, and extrapolating the stale pre-blur velocity made
      // reacquired landmarks miss the association gate and fork a second identity — the root of
      // most duplicate events. Blob centroids are arm-biased, so they steer prediction only;
      // landmark liveness (lastSeen) and the slot's measured position stay landmark-owned.
      slot.pu=u;slot.pv=v;slot.pt=t;
      if(Number.isFinite(obs.du))slot.pdu=obs.du;
      if(Number.isFinite(obs.dv))slot.pdv=obs.dv;
    }
    if(label)slot.labels.set(label,(slot.labels.get(label)??0)+(score??.5));
    const w=quality==='rigid'?1:quality==='span'?.3:.12;
    const samples=slot.samples,sample={t,u,v,p:[...p],r,w,closure,kn:finite3(kn)?[...kn]:null,label,score,quality,wristDx,chirality,axis};
    // Keep time order even if two in-flight frames resolved out of order.
    let i=samples.length;while(i>0&&samples[i-1].t>t)i--;
    samples.splice(i,0,sample);
    // Born fast: the track's first TWO consecutive sample pairs both moving hard means the punch
    // predates the track — the same evidence as an edge birth, from speed instead of position.
    // Full 3D speed, not range rate: a hook picked up mid-arc moves 3-5 m/s while closing range
    // slowly, and judging it on range rate kept the relaxed reach gate away from exactly the
    // punches that needed it. One pair is not enough: a single step of depth noise reads as
    // ~1 m/s on its own.
    if(samples.length===2||samples.length===3){
      const a=samples[samples.length-2],b=samples[samples.length-1],step=(b.t-a.t)/1000;
      const fast=step>1e-3&&hyp3(b.p[0]-a.p[0],b.p[1]-a.p[1],b.p[2]-a.p[2])/step>=BORN_FAST;
      if(samples.length===2)slot.firstPairFast=fast;
      else if(fast&&slot.firstPairFast)slot.bornFast=true;
    }
    while(samples.length&&t-samples[0].t>BUFFER_MS)samples.shift();
  }
  pushBlob(blob){
    if(!blob||!Number.isFinite(blob.t))return;
    this.blobs.push(blob);
    while(this.blobs.length&&blob.t-this.blobs[0].t>BUFFER_MS)this.blobs.shift();
  }

  // The motion stream carries a landmark-dead trajectory: while blobs continue along the slot's
  // predicted path, low-weight samples keep the buffer alive, with depth integrated from the
  // blob's expansion rate (scale grows as 1/z). Blur destroys landmarks at exactly the moment of
  // contact; it *feeds* this stream.
  #bridge(slot,now){
    const last=slot.samples[slot.samples.length-1];
    if(!last||now-slot.lastSeen<=40)return;
    const from=Math.max(slot.lastSeen,last.t);
    for(const blob of this.blobs){
      if(blob.t<=from||blob.t>slot.lastSeen+BLOB_BRIDGE_MS)continue;
      // Gated against the blob-informed prediction, so successive blobs chain along a curving
      // arc instead of falling out of a straight-line extrapolation from the last landmark.
      const [pu,pv]=slot.predictAt(blob.t);
      const gap=Math.hypot(blob.u-pu,blob.v-pv);
      if(gap>BLOB_GATE)continue;
      const prev=slot.samples[slot.samples.length-1];
      const step=clamp((blob.t-prev.t)/1000,0,.2);
      // Depth integrated from the blob's expansion rate: apparent scale grows as 1/z, so a blob
      // expanding at `expand` per second is closing depth at the same relative rate.
      const z=clamp(prev.p[2]*Math.exp(-(blob.expand??0)*step),.05,this.options.maxDepth);
      // Lateral straight from the pinhole model at that depth. The centroid is the whole moving
      // mass (fist + forearm), biased toward the arm — hence the low sample weight.
      const x=(blob.u*2-1)*z*(blob.kx??.77);
      const y=(1-blob.v*2)*z*(blob.ky??.58);
      this.push(slot,{t:blob.t,u:blob.u,v:blob.v,p:[x,y,z],quality:'blob',du:blob.du,dv:blob.dv});
      slot.bridged++;
    }
  }

  // Fitted range + closing speed at each sample of the unconsumed region.
  #series(slot){
    const samples=slot.samples,region=[];
    for(const s of samples)if(s.t>slot.consumedUntil)region.push(s);
    if(region.length<3)return null;
    const fitted=region.map(s=>{
      const fit=localFit(samples,x=>x.r,s.t,FIT_HALF_MS);
      return {t:s.t,r:fit?.value??s.r,closing:fit?-fit.slope:0,s};
    });
    return fitted;
  }

  // The extractor proper: find the latest unconsumed apex of the fitted range signal, check that
  // an approach preceded it and a retreat (or censoring) followed, gate on physical quantities
  // measured over the whole window, and hand a candidate to arbitration. Pulling the arm back
  // cannot fire here: retraction is rising range, and a candidate needs a falling segment ending
  // in a confirmed minimum.
  #extract(slot,now){
    const o=this.options;
    const fitted=this.#series(slot);
    if(!fitted){
      this.#debugFor(slot,null);
      return null;
    }
    const latest=fitted[fitted.length-1];
    // Re-arm after a fire: the fist must withdraw rearmGap before anything new can register —
    // which is what makes a double-tap without withdrawal one punch. Clearing the re-arm also
    // consumes everything seen so far, because whatever minimum formed while it was pending (an
    // instant fire's own later apex, most importantly) is part of the punch that already fired.
    if(slot.rearm){
      // Cleared only by rigid-observed withdrawal (tracked in push): a fist that never visibly
      // pulls back stays spent, however wildly span or blob depth swings around it.
      if(slot.rearm.cleared){
        slot.rearm=null;
        slot.consumedUntil=Math.max(slot.consumedUntil,latest.t);
        return null;
      }
      this.#debugFor(slot,{phase:'spent',latest});return null;
    }
    let m=0;
    for(let i=1;i<fitted.length;i++)if(fitted[i].r<fitted[m].r)m=i;
    const apex=fitted[m];
    // Censoring judges the trailing window's PEAK closing, not the last sample's: a hook that
    // blurred out in its tangential phase was closing hard 100 ms earlier — that fragment is a
    // punch losing its landing to blur, not a hand drifting to a stop.
    let trailingClosing=0;
    for(const f of fitted)if(f.t>=latest.t-150&&f.closing>trailingClosing)trailingClosing=f.closing;
    // Censoring runs from the last EVIDENCE, blob bridge included: while the motion stream is
    // still carrying the trajectory, the punch is not lost — it is heading for a measurable
    // reversal (the bracketed path) or for landmark reacquisition into this same slot. Firing on
    // landmark silence alone decided the event ~160 ms after the blur started, from half the
    // punch, and the better-solved continuation then had to be deduplicated away.
    const censorable=now-latest.t>o.graceMs&&trailingClosing>=o.stopClosing;
    // The retreat that confirms an apex must be sustained: at least two fitted samples above the
    // confirm line, held for CONFIRM_HOLD_MS, with the trajectory still up there NOW. A noise
    // bounce fails all three within a tick or two (the fist drives past the false minimum and the
    // argmin simply moves); a real retraction satisfies them ~70 ms after the landing.
    let risen=0;
    for(const f of fitted)if(f.t>apex.t+15&&f.r>=apex.r+CONFIRM_RISE)risen++;
    const bracketed=risen>=2&&latest.t>=apex.t+CONFIRM_HOLD_MS&&latest.r>=apex.r+CONFIRM_RISE*.75;
    const instant=latest.r<=o.contactDepth&&latest.closing>=o.startClosing;
    let kind=null,at=null;
    if(instant){kind='instant';at=latest;}
    else if(bracketed){kind='apex';at=apex;}
    else if(censorable){kind='censored';at=latest;}
    else{
      this.#debugFor(slot,{phase:latest.closing>=o.startClosing?'closing':'idle',latest,apex});
      return null;
    }
    // The punch's real extent: range closed from the recent maximum, not from wherever the fitted
    // speed happened to cross a gate. The windup may look back across a gate-fail consumption
    // boundary — travel belongs to the punch even when a rejected wobble sits mid-approach (which
    // is why raw samples are consulted too) — but never across a previous fire: a punch thrown out
    // of half-retraction measures from there, not from the last punch's start.
    const windupFloor=Math.max(at.t-WINDUP_MS,slot.lastFireT??-1e12);
    const windup=fitted.filter(f=>f.t>=windupFloor&&f.t<=at.t);
    let startRange=at.r,peak=0;
    for(const f of windup)if(f.r>startRange)startRange=f.r;
    // PEAK CLOSING is measured on rigid samples alone whenever they can carry a fit. Blending
    // frozen-depth fallbacks into the slope corrupts it both ways: interleaved, they flatten it
    // (a real jab measured 1.13 m/s against the 1.2 gate and was rejected — 'why does this not
    // register as any punch at all'); as a plateau ending in one rigid catch-up, they read as a
    // staircase step that inflated a soft 1.5 m/s jab to 3.4. The blended series keeps owning
    // apex timing, bracketing and censoring, where span/blob continuity is the whole point —
    // but speed is a measurement, and only measurements may set it.
    const rigidWindup=slot.samples.filter(s=>s.quality==='rigid'&&s.t>=windupFloor&&s.t<=at.t+RIGID_CONFIRM_MS);
    if(rigidWindup.length>=3){
      for(const f of windup){
        const slope=linearSlope(rigidWindup,x=>x.r,f.t,FIT_HALF_MS*1.4);
        if(slope!==null&&-slope>peak)peak=-slope;
      }
    }else for(const f of windup)if(f.closing>peak)peak=f.closing;
    for(const s of slot.samples){
      if(s.t<windupFloor||s.t>at.t)continue;
      // Blob depth is expansion-integrated off an arm-biased centroid; it must not inflate the
      // claimed travel it cannot witness.
      if(s.quality!=='blob'&&s.r>startRange)startRange=s.r;
    }
    const travel=startRange-at.r;
    // Truncated evidence softens both gates. Born against a border: the flight happened mostly
    // off-camera. Born already fast mid-frame (full 3D speed): tracking picked the punch up
    // mid-flight, so its peak CLOSING is measured only over the decelerating tail — an uppercut
    // whose rise happened below the frame reads ~0.95 m/s at its top and was being rejected
    // against the full 1.2 gate. The birth speed itself is the evidence the punch was fast.
    const edge=slot.bornAtEdge||slot.bornFast;
    const peakGate=edge?Math.min(EDGE_PEAK,o.minPeak):o.minPeak;
    const reachGate=edge?Math.min(EDGE_REACH,o.minTravel):o.minTravel;
    let closureMax=0;
    for(const f of windup)if(f.s.closure>closureMax)closureMax=f.s.closure;
    this.#debugFor(slot,{phase:'closing',latest,apex:at,peak,travel});
    const reject=reason=>{
      // Nothing was scored, so no lockout: consume only through this minimum so a slow-building
      // punch that keeps driving in can still fire at its real apex.
      slot.consumedUntil=Math.max(slot.consumedUntil,at.t);
      if(reason)this.stats.rejected=reason;
      return null;
    };
    // Idling is not a rejection: minima that no approach ever drove into (noise wiggles at guard,
    // the tail of a retraction) are consumed silently, so the readout keeps naming the gate that
    // actually turned a real punch away.
    if(kind!=='instant'&&peak<o.stopClosing)return reject(null);
    if(kind!=='instant'&&peak<peakGate)return reject(`peak ${peak.toFixed(2)} < ${peakGate.toFixed(2)}${edge?' (edge)':''}`);
    if(travel<reachGate)return reject(`reach ${(travel*100).toFixed(0)}cm < ${(reachGate*100).toFixed(0)}cm${edge?' (edge)':''}`);
    // The approach must be MEASURED, not manufactured. The range a candidate claims to have
    // closed has to be substantially witnessed by MEASURED samples: rigid fits, and anchored
    // 'span' fallbacks — whose lateral is honest and whose depth is frozen, so they can only
    // UNDERSTATE closure (a blurred hook closes range mostly laterally, and rigid-only
    // witnessing starved exactly those). Blob expansion may CARRY an approach through blur (the
    // censored path), but it can never constitute one. Witnesses
    // shortly AFTER the apex count too: landmarks reacquired at the start of the retraction
    // reveal how deep the fist actually got when the apex itself was blob-carried.
    let rigidMax=-Infinity,rigidMin=Infinity;
    for(const s of slot.samples){
      if(s.t<windupFloor||s.t>at.t+RIGID_CONFIRM_MS)continue;
      if(s.quality!=='rigid'&&s.quality!=='span')continue;
      if(s.r>rigidMax)rigidMax=s.r;
      if(s.r<rigidMin)rigidMin=s.r;
    }
    const rigidTravel=rigidMax>rigidMin?rigidMax-rigidMin:0;
    // Two ways to be witnessed: most of the claimed travel was rigid-observed, or the
    // rigid-observed part alone is a full punch-scale approach (a fist whose first frames of
    // template convergence — or whose blurred terminal — degraded to fallback samples still
    // closed an unsoftened minTravel on rigid evidence).
    if(rigidTravel<travel*RIGID_SUPPORT&&rigidTravel<o.minTravel)
      return reject(`approach unverified (${(rigidTravel*100).toFixed(0)} of ${(travel*100).toFixed(0)}cm rigid)`);
    if(closureMax<o.minFist)return reject(`open hand (closure ${closureMax.toFixed(2)})`);
    // A punch ARRIVES. Approach shape alone is not sufficient: any inward lateral guard shift
    // closes range geometrically (by the lateral slack r−z, 6-11 cm from a normal guard — right
    // in the travel gates' band) and brackets itself on the shift back, but no jab, hook or
    // uppercut terminates half a metre from the face. A censored candidate was cut mid-flight,
    // so it is judged on imminent arrival: where its trailing closing speed was about to put it.
    const arrive=kind==='censored'?at.r-trailingClosing*(IMMINENT_MS/1000):at.r;
    if(arrive>o.strikeRange)
      return reject(`stopped ${(at.r*100).toFixed(0)}cm from the head (> ${(o.strikeRange*100).toFixed(0)}cm)`);
    if(kind==='apex'){
      // The contact is the FIRST arrival into the deepest zone, not the deepest sample. A hook's
      // retraction can re-cross the closest region and dip the range BELOW the forward crossing;
      // anchoring on the global minimum then hangs the terminal window, the direction — everything
      // downstream — on the pull-back instead of the punch, which reversed the reported travel and
      // put the entry on the wrong cheek. Gates above were judged on the true minimum; the contact
      // instant rewinds to the first sample within tolerance of it.
      for(const f of fitted){
        if(f.t>at.t)break;
        if(f.t>=windupFloor&&f.r<=at.r+APEX_TOLERANCE){at=f;break;}
      }
    }
    if(kind==='censored'){
      // A censored track with a live neighbour continuing the same motion is the same fist under a
      // new identity; hand off instead of firing a partial view of the punch.
      for(const other of this.slots.values()){
        if(other===slot||now-other.lastSeen>o.graceMs)continue;
        if(Math.hypot(other.u-slot.u,other.v-slot.v)<=IMAGE_GATE){
          slot.consumedUntil=Math.max(slot.consumedUntil,at.t);
          this.stats.merged++;
          return null;
        }
      }
    }
    return {slot,kind,at,fitted,startRange,travel,peak,closureMax,windupFloor,censored:kind==='censored'};
  }

  // One event per physical punch, across identities. Another view of a just-emitted strike is
  // recognised two ways. (1) Image support: a candidate within SAME_REGION_MS and MERGE_GATE of
  // the event is the same fist seen twice at the same spot (duplicate detections). (2) Temporal
  // exclusivity: blur can displace the image position arbitrarily far between two sightings of
  // one fist — a hook crosses a third of the frame while its landmarks are dead — so no image
  // gate can catch a sequential fragment; but blur cannot make one fist appear twice AT ONCE.
  // Two real fists coexist on screen; fragments take turns. A candidate whose identity never
  // coexisted with the identity that fired is therefore the same fist re-tracked — with one
  // exemption: when BOTH sightings independently observed a complete punch (approach -> apex ->
  // sustained retreat), they are two punches thrown through a tracking dropout (a real double),
  // while a censored or instant fire means the evidence stopped mid-punch and whatever continues
  // it belongs to it.
  #suppressed(candidate,now){
    for(const e of this.recent){
      const apart=Math.abs(candidate.at.t-e.t),age=now-e.emittedAt;
      if((apart<=SAME_REGION_MS||age<=SAME_REGION_MS)
        &&Math.hypot(candidate.at.s.u-e.u,candidate.at.s.v-e.v)<=MERGE_GATE)return true;
      if((apart<=SAME_FIST_MS||age<=SAME_FIST_MS)
        &&e.slot&&!(e.complete&&candidate.kind==='apex')
        &&!coexisted(candidate.slot,e.slot))return true;
    }
    return false;
  }

  #consume(candidate,confirmT){
    const slot=candidate.slot;
    slot.consumedUntil=Math.max(slot.consumedUntil,confirmT);
    slot.rearm={min:candidate.at.r};
    slot.lastFireT=candidate.at.t;
  }

  // Solve the event from its window: one quadratic per axis around the apex gives position and
  // velocity as smooth functions of time; every reported number reads off that same fit, so
  // location, velocity and direction can no longer disagree about which instant they describe.
  #solve(candidate,{gain,headRadii,now}){
    const {slot,at,kind}=candidate;
    // TRAVEL — the direction the punch was going, which decides its type and its impact vector —
    // is solved on MEASURED samples only. A fallback sample's depth is frozen (span) or absent,
    // so a velocity fitted through one reads dz ≈ 0; for a straight punch, whose entire velocity
    // IS dz, the normalised direction then collapses onto whatever lateral and vertical noise is
    // left over, and "mostly up" reads as an uppercut. Jabs were being logged as uppercuts and
    // hooks for exactly this reason. Position may still use every sample (image position is
    // honest in all tiers); velocity may not. When the terminal window holds too few rigid
    // samples to fit, the direction falls through to the chord below rather than inventing one.
    const solid=slot.samples.filter(s=>s.quality==='rigid');
    const trail=solid.length>=3?solid:slot.samples;
    const tApex=kind==='censored'?at.t+CENSOR_EXTRAP_MS:at.t;
    const apexP=stateAt(slot.samples,tApex,FIT_HALF_MS)?.position??at.s.p;
    // Where the terminal approach began — anchors both the measured sweep (which drives the entry
    // solve) and the chord fallback for direction.
    const tStart=Math.max(slot.samples[0]?.t??at.t,candidate.windupFloor,at.t-TERMINAL_MS);
    const startP=stateAt(trail,tStart,FIT_HALF_MS)?.position??candidate.fitted[0].s.p;
    const chord=(()=>{
      const d=[0,1,2].map(k=>apexP[k]-startP[k]),len=hyp3(...d);
      return len>1e-4?d.map(v=>v/len):[0,0,-1];
    })();
    // Direction: the fitted velocity where the fist still carried real speed. At the apex of a
    // shadow punch the fist has stopped, so the tangent there is ill-conditioned; the floor walks
    // back to the latest moment of the arc that was still travelling. Each probe is its own local
    // fit, so the direction follows the arc all the way in. Two hard lessons are encoded here:
    // a probe must be CLOSING on the head — local fits near the apex blend in the retraction, and
    // a single such probe once reported one identical jab in eight as travelling OUT of the face,
    // which put its marker on the back of the skull — and the direction is averaged over the last
    // qualifying probes rather than read from one, so per-probe noise cannot steer the report.
    const probes=[];
    for(let t=at.t-TERMINAL_MS;t<=tApex;t+=10){
      const state=stateAt(trail,t,70);
      if(!state)continue;
      const range=hyp3(...state.position);
      const closing=range>1e-4?-(state.velocity[0]*state.position[0]+state.velocity[1]*state.position[1]+state.velocity[2]*state.position[2])/range:0;
      probes.push({t,position:state.position,velocity:state.velocity,speed:hyp3(...state.velocity),closing});
    }
    const vPeak=probes.reduce((most,p)=>Math.max(most,p.speed),0);
    const floor=Math.min(DIR_FLOOR_SPEED,Math.max(vPeak*DIR_FLOOR_FRACTION,1e-3));
    const qualifying=probes.filter(p=>p.speed>=floor&&p.closing>=Math.max(.05,p.speed*.15));
    let dir=null,speedAt=0,vDir=null,vFirst=qualifying[0]??null;
    if(qualifying.length){
      const latest=qualifying[qualifying.length-1];
      const mean=[0,0,0];let count=0;
      for(const p of qualifying){
        if(p.t<latest.t-DIR_AVG_MS)continue;
        for(let k=0;k<3;k++)mean[k]+=p.velocity[k];count++;
      }
      const speed=hyp3(...mean)/Math.max(count,1);
      if(speed>1e-3){dir=mean.map(v=>v/(speed*count));speedAt=speed;vDir={t:latest.t,velocity:mean.map(v=>v/count),speed};}
    }
    if(!dir)dir=chord;
    // Curvature: how much the travel direction rotated while the fist actually carried speed —
    // measured between the first and last probes above the DIR_FLOOR, because at the apex itself
    // the fitted speed is ~0 and the tangent there says nothing. Hooks arc; jabs read near zero.
    let curvature=0;
    if(vDir&&vFirst&&vDir.t>vFirst.t&&vFirst.speed>.2&&vDir.speed>.2){
      curvature=Math.acos(clamp(vFirst.velocity.reduce((s,v,k)=>s+v*vDir.velocity[k],0)/(vFirst.speed*vDir.speed),-1,1));
    }
    // Knuckle normal: averaged over the terminal window's valid samples — orientation is steadier
    // than any single frame of it.
    const kn=[0,0,0];let knW=0;
    for(const s of slot.samples){
      if(s.t<at.t-TERMINAL_MS||s.t>at.t+TERMINAL_AFTER_MS||!s.kn)continue;
      for(let k=0;k<3;k++)kn[k]+=s.kn[k]*s.w;knW+=s.w;
    }
    const knLen=knW?hyp3(...kn):0;
    const knuckleNormal=knLen>1e-6?toHead(kn.map(v=>v/knLen)):null;
    // The IMAGE-plane hand axis over the same window — the jab/uppercut discriminator, averaged
    // so a single frame's landmark jitter cannot decide a punch's type. Already head-frame
    // (+du = head +x, +dv = up) and expressed in fist-widths, so it is directly comparable
    // between punches at any distance.
    let axisU=0,axisV=0,axisF=0,axisW=0;
    for(const s of slot.samples){
      if(s.t<at.t-TERMINAL_MS||s.t>at.t+TERMINAL_AFTER_MS||!s.axis)continue;
      axisU+=s.axis.du*s.w;axisV+=s.axis.dv*s.w;axisF+=(s.axis.facing??1)*s.w;axisW+=s.w;
    }
    const handAxis=axisW?{du:axisU/axisW,dv:axisV/axisW,facing:axisF/axisW}:null;
    // Labels, confidence, wrist trail and chirality over the window.
    const labels=[];let confidence=0,confW=0,closure=0;
    let wristSum=0,wristW=0,chirSum=0,chirW=0;
    for(const s of slot.samples){
      if(s.t<at.t-WINDUP_MS||s.t>at.t+TERMINAL_AFTER_MS)continue;
      if(s.label)labels.push({label:s.label,score:s.score});
      if(s.quality!=='blob'){confidence+=(s.score??1)*s.w;confW+=s.w;}
      if(s.closure>closure)closure=s.closure;
      if(Number.isFinite(s.wristDx)){wristSum+=s.wristDx*s.w;wristW+=s.w;}
      if(Number.isFinite(s.chirality)&&s.chirality!==0){chirSum+=s.chirality*s.w;chirW+=s.w;}
    }
    // Head frame. Lateral position is the calibration-free half of the measurement; absolute depth
    // is deliberately discarded for aiming — the entry walks the apex's lateral offset along the
    // punch's MEASURED sweep, and the surface it crosses decides where the punch landed: cheek for
    // a hook, under the chin for an uppercut, the front for a jab.
    let direction=toHead(dir);
    const norm=hyp3(...direction);
    for(let k=0;k<3;k++)direction[k]/=norm||1;
    // Invariant: a landed punch never travels meaningfully BACK toward the puncher — a direction
    // with a strongly positive z is a retraction-contaminated fit, the artifact that once put jab
    // markers on the back of the skull. Only z is tested: a follow-through hook's terminal
    // direction legitimately stops closing on the head centre while staying fully valid laterally,
    // and an earlier centre-closing form of this invariant wrongly collapsed such hooks to the
    // fallback and classified them as jabs. Fall back to the terminal chord, which for a straight
    // punch is stable by construction.
    const apexRange=hyp3(...apexP)||1;
    if(direction[2]>.25){
      const chordHead=toHead(chord);
      direction=chordHead[2]<=.25?chordHead:[0,0,-1];
    }
    const aim=[-apexP[0]*gain,apexP[1]*gain,0];
    const sweepMetric=[-(apexP[0]-startP[0])*gain,(apexP[1]-startP[1])*gain];
    const speed=Math.max(candidate.peak,speedAt);
    // The approach's start, bounded by the windup so a long-idle buffer cannot masquerade as an
    // entry. Image sweep is measured from here to the apex — in head-frame image units, so +du is
    // head +x and +dv is up. Anchored on RAW samples, like the travel measurement above: a
    // gate-fail rejection mid-approach consumes fitted coverage, and a sweep measured from the
    // leftovers undercounted exactly the truncated hooks whose sweep is their classification.
    let firstS=null;
    for(const s of slot.samples){
      if(s.t<candidate.windupFloor||s.t>at.t||s.quality==='blob')continue;
      firstS=s;break;
    }
    const first={s:firstS??candidate.fitted[0].s};
    const entryHead=toHead(first.s.p).map(v=>v*gain);
    // Looming masquerades as sweep: an approaching fist's image position diverges radially AWAY
    // from the frame centre (u−.5 ∝ x/z), so a jab thrown above the low-slung laptop camera
    // "rises" in the image and an off-centre jab "sweeps" — with zero lateral motion — which
    // classified dead-straight high jabs as uppercuts. Real punch travel is never outward-radial:
    // hooks cross the frame, uppercuts climb TOWARD the centre from below. Remove the
    // outward-radial component before classification; near the centre divergence is negligible
    // and no meaningful radial direction exists, so the correction is skipped.
    let su=at.s.u-first.s.u,sv=at.s.v-first.s.v;
    const mu=(first.s.u+at.s.u)/2-.5,mv=(first.s.v+at.s.v)/2-.5,away=Math.hypot(mu,mv);
    if(away>=.08){
      const proj=(su*mu+sv*mv)/away;
      if(proj>0){su-=proj*mu/away;sv-=proj*mv/away;}
    }
    const sweep={du:-su,dv:-sv};
    // The entry-side vote is only cast when the entry is trustworthy: a slow (guard-anchored)
    // birth, or a genuine outer-edge entry. A hook first detected mid-arc near frame centre used
    // to cast it arbitrarily — which is exactly how right hooks got logged as left.
    const entryTrusted=!candidate.slot.bornFast||first.s.u<.34||first.s.u>.66;
    // Hand lineage. The strongest handedness evidence is not in the punch at all: (a) where this
    // identity RESTED before it flew — the puncher's right fist guards at low u — and (b)
    // elimination: a fist visibly resting somewhere else right now is not the fist that just
    // landed, so the punch belongs to the other hand. (b) only fills in when (a) is absent (a
    // mid-flight birth with no guard history), because a duplicate identity of the striker parked
    // at the same guard spot would otherwise vote against it.
    const restU=slot.restW>=REST_MIN_W?slot.restU:null;
    let otherRestU=null;
    if(restU===null)for(const other of this.slots.values()){
      if(other===slot||other.restW<REST_MIN_W||other.slowRun<REST_RUN)continue;
      if(now-other.lastSeen>250)continue;                                  // resting NOW, on camera
      if(Math.hypot(other.u-at.s.u,other.v-at.s.v)<=MERGE_GATE)continue;   // another view of the striker
      otherRestU=other.restU;break;
    }
    const {hand,confidence:handConfidence,votes:handVotes}=resolveHand({
      entryU:entryTrusted?first.s.u:null,labels,direction,
      wristDx:wristW?wristSum/wristW:null,
      chirality:chirW?chirSum/chirW:null,
      restU,otherRestU,
    });
    // `why` names the branch that decided the type — the live log shows it, so a misclassified
    // punch reports which evidence convinced the classifier instead of having to be guessed at.
    const detail={why:''};
    const mode=classifyMode({direction,knuckleNormal,entry:entryHead,curvature,sweep,handAxis,range:at.r},detail);
    // Impact = the FIRST point of the measured trajectory that touches the face: walk the fitted
    // path forward in time, keep only the slice within CONTACT_RANGE of the closest approach (a
    // jab crosses the silhouette laterally half a metre out, where nothing can touch), and take
    // the first point inside the silhouette. A follow-through hook thereby lands on the cheek it
    // came in through, not wherever past the centreline its fist decelerated. When blur ate the
    // entry-side flight entirely — the path was never observed outside the silhouette — a punch
    // classified lateral with truncated coverage passes its arrival direction so firstContact can
    // reconstruct where the face would have stopped it; a fully-observed path, straight punches,
    // and guard-to-apex tracks never reconstruct.
    const lateralPunch=mode==='hook'||mode==='uppercut'||mode==='overhand';
    // The reconstruction arrival is mode-aware: an uppercut classified from its orientation may
    // carry no measured vertical in its direction at all — but the classification itself asserts
    // the punch came from below, so the entry walks down to the chin (and an overhand down from
    // above). Hooks keep their measured lateral, which is what classified them.
    const arrival=mode==='uppercut'?[direction[0],Math.max(direction[1],.6)]
      :mode==='overhand'?[direction[0],Math.min(direction[1],-.6)]
      :[direction[0],direction[1]];
    // Truncation is measured, not inferred from birth flags: how much APPROACH was observed
    // before the punch entered contact range? A guard-tracked punch closes 15-30 cm on camera
    // before contact and its measured path speaks for itself; a blur-truncated hook was first
    // seen already at contact range and shows essentially none.
    const truncated=candidate.startRange-(apexRange+CONTACT_RANGE)<.12;
    const path=[];
    for(const probe of probes){
      if(hyp3(...probe.position)>apexRange+CONTACT_RANGE)continue;
      path.push([-probe.position[0]*gain,probe.position[1]*gain]);
    }
    path.push([aim[0],aim[1]]);   // the stopping point itself is always contact-eligible
    const point=firstContact(path,headRadii,
      {arrival:lateralPunch&&truncated?arrival:null});
    return {
      type:'target-impact',
      point:point??aim,aim,missed:!point,
      apexPoint:[-apexP[0]*gain,apexP[1]*gain,Math.abs(apexP[2])*gain],
      direction,velocity:direction.map(v=>v*speed),
      closing:candidate.peak,speed,
      lateral:{x:apexP[0],y:apexP[1]},depth:Math.abs(apexP[2]),range:at.r,
      travel:candidate.travel,closure:closure||candidate.closureMax,
      confidence:confW?clamp(confidence/confW,0,1):.5,
      knuckleNormal,handAxis,curvature,sweep,sweepMetric,
      mode,why:detail.why,hand,handConfidence,handVotes,lineage:{restU,otherRestU},
      solidSamples:solid.length,
      timestamp:candidate.at.t,stale:candidate.censored,instant:kind==='instant',
      slot:slot.id,label:slot.label,
      samples:slot.samples.filter(s=>s.t>=at.t-WINDUP_MS&&s.t<=at.t+TERMINAL_AFTER_MS).length,
      bridged:slot.samples.some(s=>s.quality==='blob'&&s.t>=at.t-TERMINAL_MS),
    };
  }

  #debugFor(slot,state){
    // The most recently active slot owns the readout.
    if(this.debug.owner&&this.debug.owner!==slot.id&&(this.debug.at??0)>slot.lastSeen)return;
    if(!state){Object.assign(this.debug,{phase:'idle',r:null,closing:0,owner:slot.id,slot:slot.label,at:slot.lastSeen});return;}
    const {phase,latest,apex,peak=this.debug.peak,travel=this.debug.travel}=state;
    Object.assign(this.debug,{
      phase,r:latest?.r??null,closing:latest?.closing??0,peak,travel,
      minRange:apex?.r??null,owner:slot.id,slot:slot.label,at:slot.lastSeen,
    });
  }

  /**
   * Run extraction over every slot. Returns at most one event; further candidates re-derive on the
   * next tick from their own unconsumed evidence. `gain` maps measured lateral metres onto the
   * loaded head's width; `headRadii` is the ellipsoid stand-in the aim ray is solved against.
   */
  tick(now,{gain=1,headRadii=[.15,.14,.09]}={}){
    for(const slot of this.slots.values())this.#bridge(slot,now);
    const candidates=[];
    for(const slot of this.slots.values()){
      const c=this.#extract(slot,now);
      if(c)candidates.push(c);
    }
    candidates.sort((a,b)=>a.at.r-b.at.r);
    let emitted=null;
    for(const candidate of candidates){
      if(this.#suppressed(candidate,now)){
        this.#consume(candidate,candidate.at.t);
        this.stats.refractory++;
        continue;
      }
      if(emitted){
        // Two candidates in one tick: overlapping support, or two identities that never coexisted
        // (same complete-punch exemption as #suppressed) — either way one strike; a genuinely
        // distinct punch survives to the next tick untouched.
        if(Math.hypot(candidate.at.s.u-emitted.at.s.u,candidate.at.s.v-emitted.at.s.v)<=MERGE_GATE
          ||(!(emitted.kind==='apex'&&candidate.kind==='apex')&&!coexisted(candidate.slot,emitted.slot))){
          this.#consume(candidate,candidate.at.t);
          this.stats.merged++;
        }
        continue;
      }
      const event=this.#solve(candidate,{gain,headRadii,now});
      const confirm=candidate.fitted.find(f=>f.t>candidate.at.t+15&&f.r>=candidate.at.r+CONFIRM_RISE);
      this.#consume(candidate,Math.max(candidate.at.t,confirm?.t??now));
      this.recent.push({t:candidate.at.t,u:candidate.at.s.u,v:candidate.at.s.v,emittedAt:now,
        slot:candidate.slot,complete:candidate.kind==='apex'});
      // A strike spends every nearby mid-approach view of itself, whichever identity it wears.
      for(const other of this.slots.values()){
        if(other===candidate.slot)continue;
        if(Math.hypot(other.u-candidate.at.s.u,other.v-candidate.at.s.v)<=MERGE_GATE){
          other.consumedUntil=Math.max(other.consumedUntil,candidate.at.t);
          other.rearm={min:candidate.at.r};
        }
      }
      // Below the head there is nothing to hit: a strike arriving at chest/shoulder height is
      // ignored outright — no hit, no miss, no log line. It still consumed its evidence and
      // entered the arbitration memory above, so its fragments cannot resurface as events. The
      // check runs AFTER classification steered the entry (an uppercut travels THROUGH chest
      // height but arrives at the chin, so only its classification proves it was not a body shot).
      if((event.missed?event.aim[1]:event.point[1])<-headRadii[1]*HEAD_FLOOR){
        this.stats.ignored++;
        continue;
      }
      if(candidate.censored)this.stats.stale++;
      this.stats.rejected='';
      this.stats.impacts++;
      emitted=candidate;emitted.event=event;
    }
    while(this.recent.length&&now-this.recent[0].emittedAt>EVENT_MEMORY_MS)this.recent.shift();
    for(const [id,slot] of this.slots)if(now-slot.lastSeen>this.options.graceMs*6)this.slots.delete(id);
    return emitted?.event??null;
  }
}
