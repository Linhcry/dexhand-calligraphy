"""Record unintended robot/environment contacts at each physical step.

Brush contacts and robot self contacts are excluded. This is observation only:
no state, actuator command, force, or collision parameter is modified.
"""


def surface_trace_passes(trace,saved_states,declared_clear):
    """Missing or incomplete evidence cannot pass the surface-contact gate."""
    return bool(trace and saved_states>0 and declared_clear is True
        and trace.get('robot_non_tool_surface_contact_free') is True
        and trace.get('pairs')==[] and trace.get('physics_steps_checked')==5*saved_states
        and trace.get('rate_hz')==500.)


def robot_group(name):
    if name.startswith('collision_'):return 'arm_and_palm'
    if name.startswith('col_r_f_link'):return 'fingers'
    return None


class RobotSurfaceMonitor:
    def __init__(self,model):
        self.model=model;self.names=[model.geom(i).name or '' for i in range(model.ngeom)]
        self.groups=[robot_group(n) for n in self.names];self.pen=model.body('brush').id
        self.steps=0;self.pairs={}

    def update(self,data,stage):
        self.steps+=1;per_pair={}
        for c in data.contact:
            a,b=int(c.geom1),int(c.geom2)
            if bool(self.groups[a])==bool(self.groups[b]):continue
            robot,other=(a,b) if self.groups[a] else (b,a)
            if int(self.model.geom_bodyid[other])==self.pen or c.dist>=0:continue
            key=(self.groups[robot],self.names[robot],self.names[other])
            per_pair[key]=min(per_pair.get(key,0.),float(c.dist))
        for key,distance in per_pair.items():
            if key not in self.pairs:
                self.pairs[key]={'group':key[0],'robot_geom':key[1],'environment_geom':key[2],
                    'first_time_s':float(data.time),'first_stage':str(stage),'overlapping_steps':0,
                    'maximum_penetration_mm':0.}
            row=self.pairs[key];row['overlapping_steps']+=1
            row['maximum_penetration_mm']=max(row['maximum_penetration_mm'],-distance*1000)
            row['last_time_s']=float(data.time);row['last_stage']=str(stage)

    def report(self):
        return {'physics_steps_checked':self.steps,'rate_hz':1/float(self.model.opt.timestep),
            'robot_non_tool_surface_contact_free':not bool(self.pairs),'pairs':list(self.pairs.values()),
            'scope':'Collision-geometry overlap (distance < 0), arm/palm collision_* and finger col_r_f_link* versus non-robot bodies, excluding the brush. No forces inferred. Tiny overlap is a soft-contact model quantity.',
            'hardware_ready':False}
