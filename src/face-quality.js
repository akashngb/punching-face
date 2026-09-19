export const FACE_OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
export function faceQuality(result,width,height,previous){
  if(result.faceLandmarks?.length!==1)return {ok:false,message:'Show one face, with even lighting and no obstructions.'};
  const lm=result.faceLandmarks[0],matrix=result.facialTransformationMatrixes?.[0]?.data;
  if(lm.length<468||!matrix||Array.from(matrix).some(x=>!Number.isFinite(x)))return {ok:false,message:'Hold still while the face pose is detected.'};
  const oval=FACE_OVAL.map(i=>lm[i]),xs=oval.map(p=>p.x),ys=oval.map(p=>p.y);
  if(Math.min(...xs)<.03||Math.max(...xs)>.97||Math.min(...ys)<.03||Math.max(...ys)>.97)return {ok:false,message:'Keep the forehead, chin and both cheeks inside the frame.'};
  if((Math.max(...ys)-Math.min(...ys))*height<180)return {ok:false,message:'Move closer: the face needs more image detail.'};
  const yaw=Math.atan2(matrix[8],matrix[10])*180/Math.PI,pitch=Math.atan2(-matrix[9],Math.hypot(matrix[8],matrix[10]))*180/Math.PI;
  if(Math.abs(yaw)>65||Math.abs(pitch)>30)return {ok:false,message:'Turn back slightly so the face landmarks remain visible.'};
  const eye=Math.hypot((lm[33].x-lm[263].x)*width,(lm[33].y-lm[263].y)*height),mouth=Math.hypot((lm[13].x-lm[14].x)*width,(lm[13].y-lm[14].y)*height);
  if(mouth/Math.max(eye,1)>.12)return {ok:false,message:'Close your mouth and keep a neutral expression throughout the scan.'};
  if(previous&&Math.hypot(yaw-previous.yaw,pitch-previous.pitch)<1.2)return {ok:false,message:'Turn slowly to a new angle; repeated views are skipped.'};
  return {ok:true,yaw,pitch,landmarks:lm.slice(0,468).map(p=>({x:p.x,y:p.y})),oval};
}
export function captureCoverage(frames){const yaw=frames.map(f=>f.yaw).filter(Number.isFinite);return {front:yaw.some(x=>Math.abs(x)<10),left:yaw.some(x=>x<=-25),right:yaw.some(x=>x>=25),headOnly:frames.length-yaw.length,count:frames.length};}
