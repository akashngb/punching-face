"""Continuous missing-head appearance without spherical UV poles.

Unseen hair is synthesized from a bounded photo swatch in Cartesian space;
Astra supplies hairstyle parameters, not photographic evidence.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates,distance_transform_edt

def smooth(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)

def rear_reference(path):
    rear=np.asarray(Image.open(path).convert('RGB'))/255.
    fg=((np.ptp(rear,axis=2)>.10)|(rear.max(2)<.44))&((rear.max(2)<.85)|(rear.min(2)<.55))
    rows=np.where(fg.sum(1)>rear.shape[1]*.03)[0];cols=np.where(fg.sum(0)>rear.shape[0]*.03)[0]
    if not len(rows) or not len(cols):return None
    rear=rear[rows.min():rows.max()+1,cols.min():cols.max()+1].copy();fg=fg[rows.min():rows.max()+1,cols.min():cols.max()+1]
    _,near=distance_transform_edt(~fg,return_indices=True);rear[~fg]=rear[near[0][~fg],near[1][~fg]]
    return rear

def sample(image,u,v):
    return np.stack([map_coordinates(image[:,:,i],[v*(image.shape[0]-1),u*(image.shape[1]-1)],order=1,mode='nearest') for i in range(3)],1)

def missing_head_material(x,n,p,spec,skin,rear=None,swatch=None):
    hair=(spec or {}).get('hair',{});height=p[10,1]-p[152,1];lo=p.min(0);hi=p.max(0)
    base=np.array(hair.get('colorSrgb',[.075,.064,.049]))
    # The swatch stays strictly inside the rear hair mass. Cartesian mirrored
    # repeat has finite derivatives at the crown, unlike atan2/spherical UVs.
    if swatch is None and rear is not None:
        h,w=rear.shape[:2];patch=rear[int(h*.12):int(h*.42),int(w*.28):int(w*.72)].copy()
        dark=(patch.mean(2)<.38)&(patch.max(2)<.62)
        if dark.any():
            _,nearest=distance_transform_edt(~dark,return_indices=True);patch[~dark]=patch[nearest[0][~dark],nearest[1][~dark]];swatch=patch
    scale=.070;angle=np.radians(hair.get('flowDegrees',15));c,s=np.cos(angle),np.sin(angle)
    def tile(a,b):
        a,b=(c*a-s*b)/scale,(s*a+c*b)/scale
        a=1-np.abs((a+.37)%2-1);b=1-np.abs((b+.21)%2-1)
        if swatch is not None:return sample(swatch,a,b)
        wavelength=hair.get('waveLengthMm',12)*.001
        strand=.8+.2*np.sin((a*scale+.004*np.sin(b*12))*2*np.pi/wavelength)
        fine=.92+.08*np.sin(a*780+b*13)
        return base[None]*strand[:,None]*fine[:,None]
    weights=np.maximum(np.abs(n),.001)**4;weights/=weights.sum(1)[:,None]
    hair_rgb=tile(x[:,2],x[:,1])*weights[:,0,None]+tile(x[:,0],x[:,2])*weights[:,1,None]+tile(x[:,0],x[:,1])*weights[:,2,None]
    # Smooth semantic hairline around temples and nape, never a skin wedge at
    # a texture pole. At the crown the weight is uniformly one.
    azimuth=np.abs(np.arctan2(x[:,0],x[:,2]+.09))
    threshold=np.interp(azimuth,[0,.6,1.45,2.15,np.pi],[p[10,1]+.005,p[10,1]-.015,p[10,1]-height*hair.get('sideHairlineFraction',.3),p[10,1]-height*hair.get('rearHairlineFraction',.85),p[10,1]-height*hair.get('rearHairlineFraction',.85)])
    scalp=smooth((x[:,1]-threshold)/.014)
    if hair.get('present') is False:scalp*=0
    inferred=np.tile(skin,(len(x),1))*(1-scalp[:,None])+hair_rgb*scalp[:,None]
    if rear is not None:
        # Orthographic posterior projection, blended out before it becomes
        # tangential. No cylindrical wrap, no pole, no pinched center texel.
        u=np.clip(.5-x[:,0]/max(hi[0]-lo[0],.1),.035,.965)
        v=np.clip((hi[1]-x[:,1])/max(hi[1]-p[152,1],.2),.03,.96)
        posterior=sample(rear,u,v)
        # Keep any reference-skin pixels off the model's inferred hair area.
        is_skin=smooth((posterior.mean(1)-.24)/.16)*scalp
        posterior=posterior*(1-is_skin[:,None])+hair_rgb*is_skin[:,None]
        blend=smooth((-n[:,2]-.25)/.55)*(1-smooth((n[:,1]-.25)/.45))
        inferred=inferred*(1-blend[:,None])+posterior*blend[:,None]
    return inferred,scalp
