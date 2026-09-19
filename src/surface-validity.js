// A projected triangle-area constraint. It protects the rendered skin from
// folds through itself locally; it is not a volume or self-collision solver.
export class SurfaceValidity {
  constructor(tissue){
    this.tissue=tissue;const data=[],{indices,map,vertices}=tissue;
    for(let i=0;i<indices.length;i+=3){
      const ai=map[indices[i]],bi=map[indices[i+1]],ci=map[indices[i+2]],a=vertices[ai].p,b=vertices[bi].p,c=vertices[ci].p;
      const ux=b[0]-a[0],uy=b[1]-a[1],uz=b[2]-a[2],vx=c[0]-a[0],vy=c[1]-a[1],vz=c[2]-a[2];
      const nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx,area=Math.hypot(nx,ny,nz);
      if(area>1e-12)data.push(ai*3,bi*3,ci*3,nx/area,ny/area,nz/area,area,ux,uy,uz,vx,vy,vz);
    }
    this.triangles=new Float64Array(data);this.positions=new Float64Array(vertices.length*3);this.delta=new Float64Array(this.positions.length);
  }
  constrainRelative(field,base,limit){
    this.constrain(field,.25,false,base,limit);
    const nodes=this.tissue.vertices,t=this.triangles,d=this.delta,p=this.positions;
    let alpha=1;
    const rootBound=(A,B,C)=>{
      if(C<0)return; // A pre-existing boundary is not a reason to erase old dents.
      if(Math.abs(A)<1e-18){if(B<0)alpha=Math.min(alpha,-C/B*.999);}
      else {const disc=B*B-4*A*C;if(disc>=0){const r=Math.sqrt(disc),a=(-B-r)/(2*A),b=(-B+r)/(2*A);if(a>1e-10)alpha=Math.min(alpha,a*.999);if(b>1e-10)alpha=Math.min(alpha,b*.999);}}
    };
    for(let i=0;i<nodes.length;i++){
      const v=nodes[i],copy=v.copies[0]*3;
      let dot=0,oldSquared=0;for(let j=0;j<3;j++){d[i*3+j]=field[copy+j]-base[copy+j];dot+=d[i*3+j]*base[copy+j];oldSquared+=base[copy+j]**2;}
      // Local smoothing may redistribute a dent. Do not permit it to pull a
      // previously displaced point back toward rest on another clay contact.
      if(dot<0&&oldSquared>1e-14)for(let j=0;j<3;j++)d[i*3+j]-=base[copy+j]*dot/oldSquared;
      let A=0,B=0;for(let j=0;j<3;j++){p[i*3+j]=v.p[j]+base[copy+j];A+=d[i*3+j]**2;B-=2*base[copy+j]*d[i*3+j];}
      const slack=Math.max(0,limit*limit-oldSquared),bd=-B*.5;
      if(A>1e-20){
        const radical=Math.sqrt(Math.max(0,bd*bd+A*slack));
        const root=bd>=0?(slack/(radical+bd||1)):(-bd+radical)/A;
        const localAlpha=Math.min(1,Math.max(0,root*.999));
        for(let j=0;j<3;j++)d[i*3+j]*=localAlpha;
      }
    }
    for(let k=0;k<t.length;k+=13){
      const a=t[k],b=t[k+1],c=t[k+2],nx=t[k+3],ny=t[k+4],nz=t[k+5],area=t[k+6];
      const ux=p[b]-p[a],uy=p[b+1]-p[a+1],uz=p[b+2]-p[a+2],vx=p[c]-p[a],vy=p[c+1]-p[a+1],vz=p[c+2]-p[a+2];
      const dx=d[b]-d[a],dy=d[b+1]-d[a+1],dz=d[b+2]-d[a+2],ex=d[c]-d[a],ey=d[c+1]-d[a+1],ez=d[c+2]-d[a+2];
      const A=(dy*ez-dz*ey)*nx+(dz*ex-dx*ez)*ny+(dx*ey-dy*ex)*nz;
      const B=(uy*ez-uz*ey+dy*vz-dz*vy)*nx+(uz*ex-ux*ez+dz*vx-dx*vz)*ny+(ux*ey-uy*ex+dx*vy-dy*vx)*nz;
      const C=(uy*vz-uz*vy)*nx+(uz*vx-ux*vz)*ny+(ux*vy-uy*vx)*nz-area*.075;
      rootBound(A,B,C);
    }
    alpha=Math.max(0,alpha);
    for(let i=0;i<nodes.length;i++)for(const copy of nodes[i].copies)for(let j=0;j<3;j++)field[copy*3+j]=base[copy*3+j]+d[i*3+j]*alpha;
    return alpha;
  }
  constrain(field,minimum=.18,protectPath=true,baseline=null,limit=.065){
    const nodes=this.tissue.vertices,p=this.positions,t=this.triangles,d=this.delta;
    for(let i=0;i<nodes.length;i++){const v=nodes[i],copy=v.copies[0]*3;for(let j=0;j<3;j++)p[i*3+j]=v.p[j]+field[copy+j];}
    // Most triangles require no correction. Use flat scalar loops here: this
    // runs for dense photo models at contact time, before the render animation.
    for(let pass=0;pass<(baseline?50:10);pass++){
      let changed=0;
      for(let k=0;k<t.length;k+=13){
        const a=t[k],b=t[k+1],c=t[k+2],nx=t[k+3],ny=t[k+4],nz=t[k+5],area=t[k+6];
        const ux=p[b]-p[a],uy=p[b+1]-p[a+1],uz=p[b+2]-p[a+2],vx=p[c]-p[a],vy=p[c+1]-p[a+1],vz=p[c+2]-p[a+2];
        const signed=(uy*vz-uz*vy)*nx+(uz*vx-ux*vz)*ny+(ux*vy-uy*vx)*nz;
        if(signed>=area*minimum)continue;
        const bx=vy*nz-vz*ny,by=vz*nx-vx*nz,bz=vx*ny-vy*nx,cx=ny*uz-nz*uy,cy=nz*ux-nx*uz,cz=nx*uy-ny*ux,ax=-bx-cx,ay=-by-cy,az=-bz-cz;
        const denom=ax*ax+ay*ay+az*az+bx*bx+by*by+bz*bz+cx*cx+cy*cy+cz*cz;if(denom<1e-20)continue;
        const step=(area*(minimum+.02)-signed)/denom;
        p[a]+=ax*step;p[a+1]+=ay*step;p[a+2]+=az*step;p[b]+=bx*step;p[b+1]+=by*step;p[b+2]+=bz*step;p[c]+=cx*step;p[c+1]+=cy*step;p[c+2]+=cz*step;changed++;
      }
      if(baseline){
        for(let i=0;i<nodes.length;i++){
          const node=nodes[i],copy=node.copies[0]*3;let dot=0,oldSquared=0;
          for(let j=0;j<3;j++){dot+=(p[i*3+j]-node.p[j]-baseline[copy+j])*baseline[copy+j];oldSquared+=baseline[copy+j]**2;}
          if(dot<0&&oldSquared>1e-14)for(let j=0;j<3;j++)p[i*3+j]-=baseline[copy+j]*dot/oldSquared;
          const length=Math.hypot(p[i*3]-node.p[0],p[i*3+1]-node.p[1],p[i*3+2]-node.p[2]);
          if(length>limit)for(let j=0;j<3;j++)p[i*3+j]=node.p[j]+(p[i*3+j]-node.p[j])*limit/length;
        }
      }
      if(!changed)break;
    }
    for(let i=0;i<nodes.length;i++)for(let j=0;j<3;j++)d[i*3+j]=p[i*3+j]-nodes[i].p[j];
    if(!protectPath){for(let i=0;i<nodes.length;i++)for(const copy of nodes[i].copies)for(let j=0;j<3;j++)field[copy*3+j]=d[i*3+j];return 1;}
    // Signed area during rest-to-impact interpolation is quadratic. Limit the
    // rare unresolved sliver before its first positive threshold crossing.
    let alpha=1;
    for(let k=0;k<t.length;k+=13){
      const a=t[k],b=t[k+1],c=t[k+2],nx=t[k+3],ny=t[k+4],nz=t[k+5],area=t[k+6],ux=t[k+7],uy=t[k+8],uz=t[k+9],vx=t[k+10],vy=t[k+11],vz=t[k+12];
      const dx=d[b]-d[a],dy=d[b+1]-d[a+1],dz=d[b+2]-d[a+2],ex=d[c]-d[a],ey=d[c+1]-d[a+1],ez=d[c+2]-d[a+2];
      const A=(dy*ez-dz*ey)*nx+(dz*ex-dx*ez)*ny+(dx*ey-dy*ex)*nz;
      const B=(uy*ez-uz*ey+dy*vz-dz*vy)*nx+(uz*ex-ux*ez+dz*vx-dx*vz)*ny+(ux*ey-uy*ex+dx*vy-dy*vx)*nz,C=area*(1-minimum*.5);
      if(Math.abs(A)<1e-16){if(B<0)alpha=Math.min(alpha,-C/B*.999);}
      else {const D=B*B-4*A*C;if(D>=0){const root=Math.sqrt(D),r1=(-B-root)/(2*A),r2=(-B+root)/(2*A);if(r1>0)alpha=Math.min(alpha,r1*.999);if(r2>0)alpha=Math.min(alpha,r2*.999);}}
    }
    alpha=Math.max(0,alpha);
    for(let i=0;i<nodes.length;i++)for(const copy of nodes[i].copies)for(let j=0;j<3;j++)field[copy*3+j]=d[i*3+j]*alpha;
    return alpha;
  }
}
