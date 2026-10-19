"""Paper-local paths, explicit uncertainty budgets and bounded transfers."""
from dataclasses import dataclass
import numpy as np
from fivefinger.trajectory import make_path, PenPath


@dataclass
class PaperFrame:
    center: np.ndarray
    rotation: np.ndarray
    size: np.ndarray

    @classmethod
    def from_detection(cls, detection):
        if not detection['pose_ready']: raise ValueError('Paper pose is uncertain')
        corners=np.asarray(detection['corners_m'],float)
        edges=np.roll(corners,-1,axis=0)-corners
        lengths=np.linalg.norm(edges,axis=1)
        x=edges[np.argmin(lengths)];x=x/np.linalg.norm(x)
        # Blank paper has a 180-degree ambiguity. Convention: short axis points base +X.
        if x[0]<0: x=-x
        z=np.cross(edges[0],edges[1]);z/=np.linalg.norm(z)
        if z[2]<0:z=-z
        y=np.cross(z,x)
        return cls(corners.mean(0),np.column_stack((x,y,z)),np.array([lengths.min(),lengths.max()]))

    def to_base(self, points):return np.asarray(points)@self.rotation.T+self.center

    def to_local(self, points):return (np.asarray(points)-self.center)@self.rotation


def poem_path(text,paper, speed=.045, travel_speed=.1, uncertainty_m=.002):
    if len(text) not in (20,28,40):raise ValueError('Supported rule categories: 20,28,40 characters')
    columns=7 if len(text)==28 else 5
    rows=len(text)//columns
    margin=.012+3*uncertainty_m;gap=.007
    size=min(.032,(paper.size[0]-2*margin-(columns-1)*gap)/columns,
             (paper.size[1]-2*margin-(rows-1)*gap)/rows)
    if size<.020:raise ValueError('Paper too small or uncertainty too large')
    width=columns*size+(columns-1)*gap;height=rows*size+(rows-1)*gap
    local=make_path(text,origin_xy=(-width/2,height/2-size),character_size=size,gap=gap,
                    paper_z=0,clearance=.009,speed=speed,travel_speed=travel_speed,columns=columns)
    if (np.abs(local.xyz[:,:2])>paper.size/2-margin).any():raise ValueError('Stroke outside measured paper margin')
    return PenPath(paper.to_base(local.xyz),local.pen_down,local.time_s),dict(columns=columns,rows=rows,
        character_size_m=size,path_seconds=float(local.time_s[-1]),margin_m=margin,
        writing_quality_validated=False,official_score=None)


def eye_target(center, radius_m, localization_uncertainty_m, predicted_ink_radius_m):
    """Do not equate any detected circle with the semantic eye. Caller must verify template."""
    if radius_m<=0 or predicted_ink_radius_m<=radius_m/3:
        raise ValueError('Ink radius does not satisfy the stated validity rule')
    if 3*localization_uncertainty_m+predicted_ink_radius_m>radius_m:
        raise ValueError('Uncertainty too high to guarantee ink inside eye boundary')
    point=np.asarray(center,float)
    return np.array([point+[0,0,.04],point,point+[0,0,.04]])


def transfer_waypoints(start,finish,table_z=0.,obstacles=(),clearance=.05):
    """Conservative lift-traverse-lower TOOL path. Full robot swept collision check still required."""
    a,b=np.asarray(start,float),np.asarray(finish,float)
    if not np.isfinite(np.r_[a,b]).all() or min(a[2],b[2])<table_z:raise ValueError('Target below table')
    height=max(a[2],b[2],table_z+clearance)
    for low,high in obstacles:
        low,high=np.asarray(low),np.asarray(high)
        if np.all(np.maximum(a[:2],b[:2])>=low[:2]-clearance) and np.all(np.minimum(a[:2],b[:2])<=high[:2]+clearance):
            height=max(height,float(high[2])+clearance)
    return np.array([a,[a[0],a[1],height],[b[0],b[1],height],b])
