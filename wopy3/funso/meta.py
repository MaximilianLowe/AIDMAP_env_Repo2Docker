import numpy as np
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any
from collections import OrderedDict

from ..forward.forward_calculators import masspoint_calc_FTG_2,pillar_to_pointmass,pillar_to_pointmass_chebby



class Geometry:
    """
    3-D: Shape = (nz,ny,nx)
    2-D: Shape = (nz,nx)
    1-D: Shape = (nz)
    The first dimension always corresponds to depth,
    arr[0] is always a depth-slice (or line in 2-D)
    
    The numbers correspond to the number of cells, and the axes are the
    lllcorner of each cell.
    # TODO: Is this logical?
    """
    def __init__(self,shape,spacing,origin=None):
        assert len(shape) == len(spacing)
        self.dim = len(shape)
        self.shape = shape
        self.spacing = spacing
        
        if origin is None:
            self.origin = np.zeros((len(self.shape),))
        else:
            self.origin = origin
    
    def get_axes(self):
        axis = []
        for i in range(self.dim):
            axis.append(np.arange(0,self.shape[i])*self.spacing[i] + self.origin[i])
        return axis

def above_interface(layered_model,interface_ix,invert=False,horz_mask=None):
    if horz_mask is None:
        horz_mask = (layered_model.geometry.dim-1) * (slice(None),)
        manual_slice = (slice(None),) + (None,)*(layered_model.geometry.dim-1)
    else:
        assert horz_mask.shape == layered_model.geometry.shape[1:]
        manual_slice = (slice(None),None,)

    geometry = layered_model.geometry
    depth_axis = geometry.get_axes()[0]
    interface = layered_model.interfaces[interface_ix][horz_mask]
    if invert:
        return depth_axis[manual_slice] >= interface
    else:
        return depth_axis[manual_slice] < interface

class Voxel:
    def __init__(self,geometry,const=False):
        self.geometry = geometry
        self.value = np.zeros(geometry.shape)
        self.const = const
        self.depends_on = set()
        self.change_mask = np.zeros(geometry.shape,dtype=bool)
        self.affects = set()
        self.updated = False
        self.free_variables = 0
    
    def update(self,x=None):
        """
        A constant voxel can serve as input and thus should be able to take x as argument,
        even though all subclasses are intermediates and thus don't take x as argument
        this is a bit inconsistent ...
        """
        self.updated=True
        
    def refresh(self):
        self.change_mask[:] = False
    
    def __hash__(self):
        return id(self)

class ElementwiseVoxel(Voxel):
    def __init__(self,input_voxel,operation,dtype=np.float64,*args):
        super().__init__(input_voxel.geometry)
        self.input_voxel = input_voxel
        self.operation = operation
        self.depends_on.add(input_voxel)
        self.value = np.zeros(self.geometry.shape,dtype=dtype)
        self.args = args
    
    def update(self):
        super().update()
        mask = self.input_voxel.change_mask
        self.value[mask] = self.operation(self.input_voxel.value[mask],*self.args)
        self.change_mask[mask] = True
    
    # TODO: I have no idea how efficient this is
    def create_categorical_assigner(input_voxel,categories,dtype=np.float64):
        def internal_func(x):
            y = np.ones(x.shape)*np.nan
            for c in categories:
                y[x==c] = categories[c]
            return y
        return ElementwiseVoxel(input_voxel,internal_func,dtype=dtype)
        

class OverlayVoxel(Voxel):
    """
    Should be moved to intermediate.py later on!
    """
    
    def __init__(self,background_cube,fore_domain,fore_cube):
        super().__init__(background_cube.geometry)
        self.background_cube = background_cube
        self.fore_domain = fore_domain
        self.fore_cube = fore_cube
        self.depends_on.update([background_cube,fore_cube,fore_domain])
        
        self.value = np.zeros(self.geometry.shape)
        self.value[:] = self.background_cube.value
        self.value[self.fore_domain.value] = self.fore_cube.value[self.fore_domain.value]
        
    def update(self):
        super().update()
        back_change_mask = self.background_cube.change_mask
        fore_change_mask = self.fore_cube.change_mask
        domain_change_mask = self.fore_domain.change_mask
        
        self.change_mask[domain_change_mask] = True
        self.change_mask[(~domain_change_mask) & fore_change_mask] = True
        self.change_mask[(~domain_change_mask) & back_change_mask] = True
        
        # TODO: This is redundant, but safe, could be optimized in the future
        self.value[:] = self.background_cube.value
        self.value[self.fore_domain.value] = self.fore_cube.value[self.fore_domain.value]
                
        
class NLayerModel:
    def __init__(self,geometry,n_layer=1,fixed_interfaces=None):
        """
        fixed_interfaces: dict: key is int, value is 2d array
        """
        
        self.geometry = geometry
        self.n_layer = n_layer
        self.interfaces = np.zeros((n_layer+1,)+geometry.shape[1:])
        self.affects = set()
        self.depends_on = set()
        self.updated = False
    
        self.is_fixed_interface = np.zeros((n_layer+1,),dtype=bool)
        if fixed_interfaces is None:
            fixed_interfaces = [np.zeros((n_layer+1,),dtype=bool)]
        else:
            for k in fixed_interfaces:
                self.interfaces[k] = fixed_interfaces[k]
                self.is_fixed_interface[k] = True
        
        self.interface_variables = np.prod(geometry.shape[1:]) * (n_layer + 1 - self.is_fixed_interface.sum())
       
        self.free_variables = self.interface_variables
        self.x_sub_old = np.zeros((self.free_variables,))
        self.value = np.zeros(self.geometry.shape,dtype=int)
        
    def calc_value(self,horz_mask=None):
        if horz_mask is None:
            above =np.zeros(((self.n_layer+1,)+self.geometry.shape),dtype=bool)
        else:
            above = np.zeros((self.n_layer+1,self.geometry.shape[0],horz_mask.sum()),dtype=bool)
        
        for i in range(self.n_layer+1):
            above[i,...] = above_interface(self,i,False,horz_mask)
        in_layer = np.argmax(above[1:] * ~above[:-1],0)
        valid =  ~(np.all(above,0) + np.all(~above,0))
        
        subbi = np.zeros(valid.shape,dtype=int)
        subbi[valid] = in_layer[valid]
        subbi[np.all(above,0)] = -1
        subbi[np.all(~above,0)] = -2
                
        if horz_mask is None:
            self.value = subbi
        else:
            self.value[:,horz_mask] = subbi
            
    
    def update(self,x):
        x_sub = x      
        changed = ~(x_sub == self.x_sub_old)
        changed_interface = changed[:self.interface_variables].reshape((-1,)+self.geometry.shape[1:])
        changed_loc = (changed_interface.sum(0) > 0)
        changed_vol = changed_loc * np.ones(self.geometry.shape,dtype=bool)
        # This seems wasteful, but I dont know how expensive a reshape actually is.
        new_interface = x_sub[:self.interface_variables].reshape((-1,)+self.geometry.shape[1:])
        for i,j in enumerate(np.where(~self.is_fixed_interface)[0]):
            self.interfaces[j,changed_loc] = new_interface[i,changed_loc]
    
        self.calc_value(changed_loc)
        self.change_mask = changed_vol
        
        self.x_sub_old[:] = x_sub
        self.updated = True

    
    def to_vector(self):
        a = self.interfaces[~self.is_fixed_interface,...].flatten()
        return a
    
    def __hash__(self):
        return id(self)
    
        
class Gravity:
    def __init__(self,density,stations,params):
        self.stations = stations
        self.density = density
        self.geometry = density.geometry
        axes = self.geometry.get_axes()
        self.depends_on = set([density])
        self.change_mask = np.zeros((stations.shape[0]),dtype=bool)
        self.affects = set()
        
        assert self.geometry.dim == 3
        hsplit = params.get("hsplit",1)
        vsplit = params.get("vsplit",1)
        self.ref_pillar = params.get("ref_pillar",np.zeros((self.geometry.shape[0])))
        self.cheb_order = params.get("cheb_order",0)
        self.hsplit = hsplit
        self.vsplit = vsplit
        
        
        self.template_pillar = pillar_to_pointmass(0.,0.,axes[0],
                        self.geometry.spacing[2],self.geometry.spacing[1],self.geometry.spacing[0],
                        hsplit=self.hsplit,vsplit=self.vsplit)
        self.pillar_grav = np.zeros((stations.shape[0],)+self.geometry.shape[1:])
        
    def calc_pillar(self,row_ix,col_ix):
        lon,lat = self.geometry.get_axes()[2][col_ix],self.geometry.get_axes()[1][row_ix]
        axes = self.geometry.get_axes()

        if self.cheb_order > 0:
            pillar = pillar_to_pointmass_chebby(lon,lat,axes[0],
                        self.geometry.spacing[2],self.geometry.spacing[1],self.geometry.spacing[0],
                        self.cheb_order,hsplit=self.hsplit,vsplit=self.vsplit,
                        density=self.density.value[:,row_ix,col_ix]- self.ref_pillar)      
            pillar_mass = pillar[3]

        else:
            pillar = pillar_to_pointmass(lon,lat,axes[0],
                        self.geometry.spacing[2],self.geometry.spacing[1],self.geometry.spacing[0],
                        hsplit=self.hsplit,vsplit=self.vsplit)
            pillar_mass = pillar[3] * (self.density.value[:,row_ix,col_ix]- self.ref_pillar)[:,None,None,None] 
        temp =  masspoint_calc_FTG_2(pillar[0],pillar[1],pillar[2],pillar_mass,
            self.stations[:,0],self.stations[:,1],self.stations[:,2],calc_mode='grav')
        self.pillar_grav[:,row_ix,col_ix] = temp.sum((0,1,2,3))[:,0]
        
    def update(self):
        mask = self.density.change_mask
        horz_mask = np.any(mask,axis=0)
        for i,j in np.transpose(np.where(horz_mask)):
            self.calc_pillar(i,j)
        self.updated = True
        self.change_mask[:] = True
    @property
    def value(self):
        return self.pillar_grav.sum((1,2))

    def __hash__(self):
        return id(self)

        
class Project:
    def __init__(self,observables):
        self.observables = observables
        self.misfits = OrderedDict()
        self.priors = OrderedDict()
        self.proposals = OrderedDict()
        self.prop_weights = OrderedDict()
        self.objects = set()
        # I'm using an orderedDict basically as an ordered Set, because
        # the order matters.
        self.leaves = OrderedDict()
        queue = list()
        queue.extend(self.observables)
        while queue:
            obj = queue.pop()
            self.objects.add(obj)
            if not obj.depends_on:
                self.leaves[obj] = None
            for d in obj.depends_on:
                queue.append(d)
                d.affects.add(obj)
        
        assert np.all([not obs.affects for obs in self.observables])
        
        self.slices = dict()
        counter = 0
        for leaf in self.leaves:
            self.slices[leaf] = slice(counter,counter+leaf.free_variables)
            counter += leaf.free_variables
            
    
    def refresh_all(self):
        for obj in self.objects:
            obj.change_mask[:] = False
            obj.updated = False
    
    def update(self,x):
        
        @dataclass(order=True)
        class PrioritizedItem:
            priority: int
            item: Any=field(compare=False)
        
        self.refresh_all()
        queue = PriorityQueue()
        
        for leaf in self.leaves:
            leaf.update(x[self.slices[leaf]])
            for obj in leaf.affects:
                queue.put(PrioritizedItem(0,obj))
        
        while not queue.empty():
            temp = queue.get()
            priority = temp.priority
            obj = temp.item
            if(obj.updated):
                #print(obj,' is already updated')
                continue
            #print('project is trying to update ',obj,' with priority ',priority)
            if all((d.updated for d in obj.depends_on)):
                #print('updating ',obj,' with priority ',priority)
                obj.update() #Note that the intermediates do not receive any x as input
                #print('done!')
                for a in obj.affects:
                    queue.put(PrioritizedItem(priority+1,a))
            else:
                #print(obj,' is not ready for updating')
                queue.put(PrioritizedItem(priority+1,obj))
    
    def forward(self,x):
        self.update(x)
        y = [obs.value for obs in self.misfits]
        return y
    
    def add_misfit(self,output_element,misfit):
        self.misfits[output_element] = misfit
    
    def misfit(self,y):
        logp = 0.
        for i,obs in enumerate(self.misfits):
            logp += self.misfits[obs](y[i])
        return logp
    
    def add_prior(self,input_element,prior):
        self.priors[input_element] = prior
    
    def prior(self,x):
        logp = 0.
        for obj in self.priors:
            logp += self.priors[obj](x[self.slices[obj]])
        return logp
    
    def add_proposal(self,input_element,proposal,prop_weight=None):
        self.proposals[input_element] = proposal
        if prop_weight is None:
            self.prop_weights[input_element] = (self.slices[input_element].stop - self.slices[input_element].start)
        else:
            self.prop_weights[input_element] = prop_weight
        
    def proposal(self):
        key = np.random.choice(list(self.proposals.keys()),p=np.array(list(self.prop_weights.values()))/sum(self.prop_weights.values()))
        changed,dx = self.proposals[key]()
        return self.slices[key].start + changed,dx
        