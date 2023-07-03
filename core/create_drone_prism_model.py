import itertools
import math
import numpy as np

template_header = """
mdp

const int MAXX = 15;
const int MAXY = 15;
const int MAXZ = 15;


formula attarget2 = x > MAXX - 3;
formula attarget = x > MAXX - 2 & y > MAXY - 2 & z > MAXZ - 2;
formula crash = (xarea2 & yarea4 & zarea1) | (xarea1 & yarea3 & zarea2) | (xarea4 & yarea2 & zarea2) | (xarea3 & yarea0 & zarea4);
formula valid = (1 <= x & x <= MAXX-1 & 1 <= y & y <= MAXY-1 & 1 <= z & z <= MAXZ-1) & !crash & ok=1; 
"""


control_module = """
module control
    c : [0..1] init 0;
    [move] c = 0 -> 1: (c'=1);
    [env] c = 1 -> 1: (c'=0);
endmodule
"""

mod_head = """
module mod
    ok : [0..1] init 1;
    cond : [0..0] init 0;    
    wdir : [0..0] init 0;
        
    x : [0..MAXX] init 2;
    y : [0..MAXY] init 2;
    z : [0..MAXZ] init 2;
    
    phase : [0..1] init 0;
        
    [move] x < MAXX -> 1: (x'=x+1);
    [move] x > 0 -> 1: (x'=x-1);
    [move] y < MAXY -> 1: (y'=y+1);
    [move] y > 0 -> 1: (y'=y-1);
    [move] z < MAXX -> 1: (z'=z+1);
    [move] z > 0 -> 1: (z'=z-1);
    [move] attarget -> 1: (phase'=1);

 
    [env] !valid -> 1: (ok'=0) & (x'=0) & (y'=0) & (z'=0) & (phase'=0);
"""

mod_foot="""    
endmodule
"""

reward_head="""
rewards "time"  // flight time
"""
reward_foot="""
endrewards
"""

labels="""

"""

xsplits = [2,4,7,9,11]
ysplits = [3,5,7,10,13]
zsplits = [2,4,8,10,12]

x_area_splits = [0] + xsplits + ["MAXX"]
y_area_splits = [0] + ysplits + ["MAXY"]
z_area_splits = [0] + zsplits + ["MAXZ"]

def _areadefinitions(dir, splits):
    return "\n".join(["formula {}area{} = {} <= {} & {} < {};".format(dir, i, 
                                              l, dir, dir, h) for i, (l, h) in
               enumerate(zip(splits[:-1], splits[1:]))])

x_area_definitions = _areadefinitions("x", x_area_splits)
y_area_definitions = _areadefinitions("y", y_area_splits)
z_area_definitions = _areadefinitions("z", z_area_splits)
area_definitions = "\n".join([x_area_definitions, y_area_definitions, 
                              z_area_definitions])

conditions = [0]
winddirs = [0]
xzones = list(range(len(xsplits)+1))
yzones = list(range(len(ysplits)+1))
zzones = list(range(len(zsplits)+1))

def get_update(dir,eff):
    if dir % 10 == 1:
        # straight
        affected_var, other_var = ("x","y") if dir in [0,4] else ("y","x")
        main_var_sym = "+" if dir < 4 else "-"
        sec_var_op = (["","-1","+1"][eff%3]) if dir in [0,6] else (["","+1","-1"][eff%3])
        return "({}' = {}{}{})".format(affected_var, affected_var, 
                                       main_var_sym, math.ceil(1)) + \
               " & " + "({}' = {}{})".format(other_var,other_var,sec_var_op)
    else:
        # diagonals
        if eff == 0:
            return "(x'=x) & (y'=y)"
        if eff == 1:
            return ("(x'=x+1)")
        if eff == 2:
            return ("(x'=x+2)")
        if eff == 3:
            return ("(x'=x-1)")
        if eff == 4:
            return ("(x'=x-2)")
        if eff == 5:
            return ("(y'=y+1)")
        if eff == 6:
            return ("(y'=y+2)")
        if eff == 7:
            return ("(y'=y-1)")
        if eff == 8:
            return ("(y'=y-2)")
        if eff == 9:
            return ("(y'=y+1) & (x'=x+1)")
        if eff == 10:
            return ("(y'=y-1) & (x'=x-1)")
        if eff == 11:
            return ("(y'=y+1) & (x'=x-1)")
        if eff == 12:
            return ("(y'=y-1) & (x'=x+1)")
        assert False

num_effs=13

def create_command(c,w,x,y,z):
    guard = "cond = {} & wdir = {} & xarea{} & yarea{} & zarea{} & valid".format(c, w, x, y, z)
    updates = [get_update(w, i) for i in range(num_effs)]
    probs = ["pc{}w{}x{}y{}z{}eff{}".format(c, w, x, y, z, i) 
             for i in range(num_effs)]
    weighted_updates = ["{}: {}".format(p, u) for p, u in zip(probs, updates)]
    update_str = " + ".join(weighted_updates)
    return "\t[env] {} -> {};".format(guard, update_str)

commands = [create_command(c,w,x,y,z) for (c,w,x,y,z) in 
            itertools.product(conditions, winddirs, xzones, yzones, zzones)]

def parameter_groups():
    groups = []
    
    for  (c,w,x,y,z) in itertools.product(conditions, winddirs, xzones, 
                                          yzones, zzones):
        groups.append(["pc{}w{}x{}y{}z{}eff{}".format(c, w, x, y, z, i) 
                       for i in range(num_effs)])
    
    return groups

def parameter_definitions(groups,init):
    if init:
        return "\n".join(["const double {} = 1/{};".format(par, len(group)) 
                          for group in groups for par in group])
    else:
        return "\n".join(["const double {};".format(par) for group in groups 
                                                         for par in group])

def parameter_dirichlet_instantiations(groups,weather):
    instantiations=dict()
    for group in groups:
        
        #if there is eff in the parameter group, create dirichlet weights 
        # non-uniformly
        if weather == "uniform":
            array = np.random.random_sample((len(group),))
        elif weather == "x-neg-bias":
            if any("eff" in group_item for group_item in group):
                array = np.random.random_sample((len(group),))
                array[0]=1*array[0]
                array[1]=1*array[1]
                array[2]=1*array[2]
                array[3]=2*array[3]
                array[4]=2*array[4]
                array[5]=1*array[5]
                array[6]=1*array[6]
                array[7]=1*array[7]
                array[8]=1*array[8]
                array[9]=1*array[9]
                array[10]=2*array[10]
                array[11]=2*array[11]
                array[12]=1*array[12]
                
            #create uniformly
            else:
                array = np.random.random_sample((len(group),))
        elif weather == "y-pos-bias":
            if any("eff" in group_item for group_item in group):
                array = np.random.random_sample((len(group),))
                array[0]=1*array[0]
                array[1]=1*array[1]
                array[2]=1*array[2]
                array[3]=1*array[3]
                array[4]=1*array[4]
                array[5]=1*array[5]
                array[6]=1*array[6]
                array[7]=2*array[7]
                array[8]=2*array[8]
                array[9]=1*array[9]
                array[10]=2*array[10]
                array[11]=1*array[11]
                array[12]=2*array[12]
                
            #create uniformly
            else:
                array = np.random.random_sample((len(group),))
        else:
            raise RuntimeError("Specificed weather is not compatible")

        instantiations[tuple(group)]=array
    return instantiations

def parameter_dirichlet_samples(parameter_instantiations):
    instantiations_sample=dict()
    for group in parameter_instantiations:

        array = parameter_instantiations[tuple(group)]
        if any("RewardEnvw" in group_item for group_item in group):
            sample=10*np.random.dirichlet((array), 1)
        else:
            sample=np.random.dirichlet((array), 1)

        instantiations_sample[tuple(group)]=sample

    return instantiations_sample

if __name__ == '__main__':
    
    groups = parameter_groups()
    weather = "uniform"
    
    parameter_defs = parameter_definitions(groups,False)
    parameter_instantiations=parameter_dirichlet_instantiations(groups,weather)
    parameter_samples=parameter_dirichlet_samples(parameter_instantiations)
    
    with open("drone_model.nm",'w') as file:
        file.write(template_header)
        file.write(area_definitions)
        file.write("\n\n")
        file.write(parameter_defs)
        file.write(control_module)
        file.write(mod_head)
        file.write("\n".join(commands))
        file.write(mod_foot)
