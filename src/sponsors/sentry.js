// Sentry for the browser, loaded only when a DSN is configured. Everything else calls `obs`,
// whose methods are no-ops until then, so no feature depends on Sentry being present.
//
// Privacy: the webcam <video> and every image are blocked from Session Replay, the 3D canvas is
// never recorded (no canvas integration), inputs are masked, and no request bodies are attached.
// What Sentry sees is timing, status text, counts and errors: never a face.
let sdk=null;
export const obs={
  enabled:false,
  span(name,attributes,work){return sdk?sdk.startSpan({name,op:name.split('.')[0],attributes},work):work({setAttribute(){},setStatus(){}});},
  log(message,attributes={}){sdk?.logger?.info(message,attributes);},
  warn(message,attributes={}){sdk?.logger?.warn(message,attributes);},
  error(error,context){if(sdk)sdk.captureException(error,context?{extra:context}:undefined);else console.error(error);},
  crumb(category,message,data){sdk?.addBreadcrumb({category,message,data,level:'info'});},
  tag(key,value){sdk?.setTag(key,value);},
};

export async function initSentry({dsn,environment}){
  if(!dsn||sdk)return obs.enabled;
  const Sentry=await import('@sentry/browser');
  Sentry.init({
    dsn,environment,sendDefaultPii:false,enableLogs:true,
    integrations:[
      Sentry.browserTracingIntegration(),
      Sentry.replayIntegration({maskAllText:false,maskAllInputs:true,blockAllMedia:true}),
    ],
    // Every trace is kept: this is a demo-sized app and the traces are the point.
    tracesSampleRate:1,replaysSessionSampleRate:1,replaysOnErrorSampleRate:1,
    // Continue traces into the three local Python services (5174/5175 via the Vite proxy, 5176 direct).
    tracePropagationTargets:[/^\/(api|physics)\//,/^http:\/\/(127\.0\.0\.1|localhost):5176\//],
    beforeBreadcrumb:crumb=>crumb.category==='console'?null:crumb,
  });
  sdk=Sentry;obs.enabled=true;
  return true;
}

// Physics health as logs: the step loop runs at 30 Hz, far too hot to trace per call, so report a
// rolling summary instead. `metrics` comes from window.__contactLab.state.physicsMetrics.
export function reportPhysics(readState,everyMs=5000){
  let samples=[];const collect=setInterval(()=>{const m=readState()?.physicsMetrics;if(m&&Number.isFinite(m.stepMs))samples.push(m.stepMs);},250);
  const report=setInterval(()=>{
    if(!samples.length||!obs.enabled)return void(samples=[]);
    const sorted=samples.slice().sort((a,b)=>a-b),p=q=>sorted[Math.min(sorted.length-1,Math.floor(sorted.length*q))];
    obs.log('physics.step',{p50_ms:+p(.5).toFixed(1),p95_ms:+p(.95).toFixed(1),max_ms:+sorted[sorted.length-1].toFixed(1),samples:sorted.length});samples=[];
  },everyMs);
  return()=>{clearInterval(collect);clearInterval(report);};
}
