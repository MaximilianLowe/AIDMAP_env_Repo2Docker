import numpy as np
from dataclasses import dataclass,field
import equivalent_layer_model as equi
import equivalent_layer_model_cartesian as equi_cart
import pyshtools as pysh
import time
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

@dataclass
class ESInput:
    survey_ix:np.ndarray
    lonlatz: np.ndarray
    TFA:np.ndarray
    dgrf_info:np.ndarray
    year : np.ndarray
    err: np.ndarray
    igrf_NED_stat : np.ndarray = field(init=False)
    N_data : int = field(init=False)
    
    def __post_init__(self):
        self.igrf_NED_stat = np.squeeze(equi.igrf_comp(self.dgrf_info[:,2],self.dgrf_info[:,1],self.dgrf_info[:,0]))
        self.N_data = self.lonlatz.shape[0]
        if self.igrf_NED_stat.ndim == 1:
            self.igrf_NED_stat = self.igrf_NED_stat[None,:]

@dataclass
class ESBlockSources:
    dipoles: np.ndarray
    igrf_NED_dip: np.ndarray
    block_definition: equi_cart.CartesianBlockDefinition
    spacing: float
    block_row: np.ndarray
    block_col: np.ndarray


@dataclass
class ESSettings:
    aux_spacing: float
    aux_height: float
    height_factor: float
    block_factor: float
    lambda_d: float
    igrf_years: np.ndarray
    aux_year: float
    fill_threshold: float
    strategy: str
    corners: np.ndarray
    far_field_outer : int = 1
    blank_center : bool = False
    lambda_s : float = 0
    lambda_factor_override : float = None
    corner_buffer : float = 0

    def __post_init__(self):
        self.corners = np.array(self.corners)
        self.igrf_years = np.array(self.igrf_years)

    def to_json_dict(self):
        returnor = dict()
        for k,it in self.__dict__.items():
            if isinstance(it,np.ndarray):
                returnor[k] = it.tolist()
            else:
                returnor[k] = it
        return returnor

@dataclass
class ESResults:
    dipoles: np.ndarray
    equivalent_sources: np.ndarray
    predicted: np.ndarray
    self_predicted: np.ndarray
    spacing:float

class ESProblem:
    def __init__(self,settings:ESSettings,input_data : ESInput,worldmap):
        self.settings = settings
        self.input_data = input_data
        self.worldmap = worldmap
        self.auxiliary,self.igrf_NED_aux,self.aux_shape,_,_ = self._create_dipoles(self.settings.aux_spacing,-self.settings.aux_height,year_override=self.settings.aux_year)

        self.stations_epochs = np.argmin( np.abs(self.settings.igrf_years[None,:] - self.aeromag_year[:,None]),axis=1)
        self.aux_epoch = np.argmin( np.abs(self.settings.igrf_years - self.settings.aux_year))
    
    @property
    def aeromag_year(self):
        return self.input_data.year

    @property
    def err(self):
        return self.input_data.err
    
    @property
    def igrf_NED_stat(self):
        return self.input_data.igrf_NED_stat
    
    @property
    def N_data(self):
        return self.input_data.N_data

    def _create_dipoles(self,dipole_spacing,dipole_depth,year_override=None):
        worldmap = self.worldmap
        corners = self.settings.corners
        ## TODO: There is something strange going on here with dipoles
        ## ending up outside of the corners. Am I mis-understanding 
        ## how arange works?
        xdp = np.arange(corners[0],corners[1]+dipole_spacing,dipole_spacing)
        ydp = np.arange(corners[2],corners[3]+dipole_spacing,dipole_spacing)
        xi,yi = np.meshgrid(xdp,ydp)
        lonp,latp = worldmap(xi,yi,inverse=True)

        dipoles = np.vstack((lonp.flatten(),latp.flatten(),-dipole_depth*np.ones(lonp.size))).T
        order = [1,2,0]

        if year_override is None:
        
            igrf_years = self.settings.igrf_years

            multi_year = np.zeros((len(igrf_years),lonp.flatten().shape[0],3))
            for i,y in enumerate(igrf_years):
                igrf_coeffs = pysh.SHMagCoeffs.from_array(pysh.shio.read_igrf('igrf13coeffs.txt',year=y),6371.2)
                igrf_NED_dip = igrf_coeffs.expand(lon=lonp.flatten(),lat=latp.flatten(),a=6371.2-0.001*dipole_depth)
                igrf_NED_dip[:,0] = - igrf_NED_dip[:,0]
                igrf_NED_dip[:,1] = - igrf_NED_dip[:,1]
                igrf_NED_dip = igrf_NED_dip[:,order]
                multi_year[i] = igrf_NED_dip
            return dipoles,multi_year,lonp.shape,xdp,ydp
        else:
            igrf_coeffs = pysh.SHMagCoeffs.from_array(pysh.shio.read_igrf('igrf13coeffs.txt',year=year_override),6371.2)
            igrf_NED_dip = igrf_coeffs.expand(lon=lonp.flatten(),lat=latp.flatten(),a=6371.2-0.001*dipole_depth)
            igrf_NED_dip[:,0] = - igrf_NED_dip[:,0]
            igrf_NED_dip[:,1] = - igrf_NED_dip[:,1]
            igrf_NED_dip = igrf_NED_dip[:,order]
            return dipoles,igrf_NED_dip,lonp.shape,xdp,ydp

    def _create_block_sources(self,dipole_spacing):
        if self.settings.corner_buffer < dipole_spacing:
            print('*** Warning! No corner_buffer is set, this will produce dipoles outside of the block definition')
        dipole_depth = dipole_spacing * self.settings.height_factor
        dipoles,igrf_NED_dip,_,xdp,ydp = self._create_dipoles(dipole_spacing,dipole_depth)
        block_size = dipole_spacing * self.settings.block_factor
        block_definition = equi_cart.CartesianBlockDefinition(self.settings.corners,block_size,buff=self.settings.corner_buffer)
        block_row,block_col = block_definition.assign_grid(xdp,ydp)

        return ESBlockSources(dipoles,igrf_NED_dip,block_definition,dipole_spacing,block_row,block_col)

    def eq_run_extend(self,dipole_spacing,year_shift=None,prior=None,lambda_factor_override=None):
        ## Station info
        lonlatz = self.input_data.lonlatz
        if year_shift is None:
            year_shift = np.zeros(self.N_data)
        if lambda_factor_override is None:
            lambda_factor_override = self.settings.lambda_factor_override
        ## TODO: This aeromag variable is a temp solution but I didn't want to touch single_block_inner right now
        aeromag = np.concatenate((self.input_data.lonlatz,self.input_data.TFA[:,None]),axis=1)
        ## Create dipoles
        es_sources = self._create_block_sources(dipole_spacing)
        dipoles = es_sources.dipoles
        igrf_NED_dip = es_sources.igrf_NED_dip
        block_definition = es_sources.block_definition

        block_stat = block_definition.assign(*self.worldmap(lonlatz[:,0],lonlatz[:,1]))
        block_dip = (es_sources.block_row.flatten(),es_sources.block_col.flatten())
        equivalent_sources = np.zeros((dipoles.shape[0]))
        self_predicted = np.zeros(self.N_data)

        for i,j in block_definition:
            in_block_dip = (block_dip[0] == i) & (block_dip[1] == j)
            in_block_stat = (block_stat[0] == i) & (block_stat[1] == j)
            
            fill_ratio = in_block_stat.sum() / in_block_dip.sum()
            if fill_ratio < self.settings.fill_threshold:
                continue
            
            in_big_block_dip = (np.abs(block_dip[0] - i ) <= 1) & (np.abs(block_dip[1] - j ) <= 1)
            in_big_block_stat = (np.abs(block_stat[0] - i ) <= 1) & (np.abs(block_stat[1] - j ) <= 1)

            if in_big_block_dip.sum() == 0 or in_big_block_stat.sum() == 0:
                continue
            

            if prior is None:
                sub_prior = None
            else:
                sub_prior = prior[in_big_block_dip].copy()
                if self.settings.blank_center:
                    sub_prior[(block_dip[0][in_big_block_dip] == i) & (block_dip[1][in_big_block_dip] == j) ] = 0

            if self.settings.lambda_s > 0:
                ## TODO: I have no fucking clue if the ordering is correct.
                satan = (np.abs(es_sources.block_row[:,0] - i ) <= 1)
                satan_2 = (np.abs(es_sources.block_col[0,:] - j ) <= 1)
                C_x,C_y = equi.build_1st_order_diffmatrix(satan_2.sum(),satan.sum())
                try:
                    assert (satan.sum()*satan_2.sum()) == in_big_block_dip.sum()
                except AssertionError:
                    print(satan.sum(),satan_2.sum(),in_big_block_dip.sum())
            else:
                C_x = None
                C_y = None

            ano_design_matrix,temp = single_block_inner(dipoles[in_big_block_dip],aeromag[in_big_block_stat],
                                                        igrf_NED_dip[:,in_big_block_dip],self.igrf_NED_stat[in_big_block_stat],
                                                        self.settings.lambda_d,self.settings.lambda_s,year_shift=year_shift[in_big_block_stat],
                                                        stations_epochs = self.stations_epochs[in_big_block_stat],err = self.err[in_big_block_stat],
                                                        prior = sub_prior,C_x=C_x,C_y=C_y,lambda_factor_override=lambda_factor_override)
            tomp = np.zeros(dipoles.shape[0])
            tomp[in_big_block_dip] += temp
            equivalent_sources[in_block_dip] += tomp[in_block_dip]
            tank = np.zeros(self.N_data)
            tank[in_big_block_stat] = ano_design_matrix.dot(temp)
            self_predicted[in_block_stat] = tank[in_block_stat]

        ## Auxilliary and measured prediction
        predicted = equi_cart.calculate_far_field(self.worldmap,block_definition,dipoles,self.input_data.lonlatz,
                                                igrf_NED_dip,self.igrf_NED_stat,equivalent_sources,
                                                inner=0,outer=1,stations_epochs=self.stations_epochs)
        return es_sources,ESResults(dipoles,equivalent_sources,predicted,self_predicted,dipole_spacing)
    
    def eq_run_sparse(self,dipole_spacing,year_shift=None):
        ## TODO: Include Prior
        ## Station info
        if year_shift is None:
            year_shift = np.zeros(self.N_data)
        ## Create dipoles
        es_sources = self._create_block_sources(dipole_spacing)
        dipoles = es_sources.dipoles
        igrf_NED_dip = es_sources.igrf_NED_dip
        block_definition = es_sources.block_definition

        ano_design_matrix,_ = get_ano_design_matrix_sparse(es_sources,self.input_data,self.worldmap,self.settings.block_factor*dipole_spacing,self.stations_epochs)
        lambda_factor = float(np.abs(ano_design_matrix).sum(0).mean(1))

        regularization_mat = 0
        ## TODO: Warning, the usage of lambda_s and lambda_factor is not consistent with SingleBlockInner
        ## ifg/ag-ebbing/regional-spherical-harmonic-analysis#7
        if self.settings.lambda_s > 0:
            C_x,C_y = equi.build_1st_order_diffmatrix(es_sources.block_row.shape[1],es_sources.block_row.shape[0])
            regularization_mat += (C_x.T.dot(C_x) + C_y.T.dot(C_y)) * (lambda_factor * self.settings.lambda_s) ** 2

        regularization_mat += (lambda_factor  * self.settings.lambda_d)**2 * sparse.eye(ano_design_matrix.shape[1])

        lhs = ano_design_matrix.T.dot(ano_design_matrix) + regularization_mat
        rhs = ano_design_matrix.T.dot(self.input_data.TFA - year_shift)
        sparse_returns = splinalg.lsqr(lhs,rhs,atol=0.01,btol=0.01)
        equivalent_sources =sparse_returns[0]
        ## Auxilliary and measured prediction
        predicted = ano_design_matrix.dot(equivalent_sources)
        return es_sources,ESResults(dipoles,equivalent_sources,predicted,None,dipole_spacing)

def single_block_inner(dipoles,aeromag,igrf_NED_dip,igrf_NED_stat,
                                    lambda_d,lambda_s,far_field = None,year_shift=None,stations_epochs=None,
                                    err = None,prior=None,C_x=None,C_y=None,lambda_factor_override=None):
    #TODO: There is an error in the combination of smoothing + prior!
    if far_field is None:
        far_field = np.zeros(aeromag.shape[0])
    if year_shift is None:
        year_shift = np.zeros(aeromag.shape[0])
    if err is None:
        err = np.ones(aeromag.shape[0])

    if igrf_NED_dip.ndim == 3:
        ano_design_matrix = equi.calculate_ano_design_matrix_multi_year(aeromag,dipoles,
                                                    igrf_NED_stat,igrf_NED_dip,stations_epochs)
    else:
        ano_design_matrix = equi.calculate_ano_design_matrix(aeromag,dipoles,
                                                    igrf_NED_stat,igrf_NED_dip)
        
    G = ano_design_matrix / err[:,None]

    if lambda_factor_override is None:
        lambda_factor = np.abs(G.T.dot(G)).sum(1).mean(0)
    else:
        lambda_factor = lambda_factor_override
    big_matrix = G.T.dot(G) + np.eye(G.shape[1]) * lambda_d * lambda_factor

    if lambda_s > 0:
        big_matrix += lambda_s * lambda_factor * (C_x.T.dot(C_x)  + C_y.T.dot(C_y))
    
    rhs = aeromag[:,3]-far_field-year_shift
    
    if not prior is None:
        rhs = rhs - ano_design_matrix.dot(prior)
        prior_x = prior
    else:
        prior_x = 0

    temp = np.linalg.solve(big_matrix,G.T.dot(rhs/err)) + prior_x
    
    return ano_design_matrix,temp

class ETA:
    """Slightly configurable object to report progress to command line
    Copied from wopy3.utils
    """
    def __init__(self,n,step=1.0,wait_time=0):
        self.n = n
        self.ix = 0
        self.wait_time = wait_time
        self.t_start = time.time()
        self.step = step
        self.last_message = 0
        self.last_time = self.t_start
        
    def __call__(self):
        t = time.time()
        self.ix = self.ix + 1
        if 100.0*(self.ix-self.last_message)/self.n >= self.step and (t-self.last_time) > self.wait_time:
            print('%.1f %% complete ETA %.2f s' % (((100.0*self.ix)/self.n),(t-self.t_start)/self.ix*(self.n-self.ix)))
            self.last_message = self.ix#
            self.last_time = t
            return True

def get_shift(es_input,predicted,far_field=0,shift_factor = 1,shift_lambda=0):
    shifts = np.zeros(len(np.unique(es_input.survey_ix)))
    shift_vec = np.zeros(es_input.lonlatz.shape[0])
    corrected = es_input.TFA -far_field
    for i,u in enumerate(np.unique(es_input.survey_ix)):
        sel  = es_input.survey_ix == u
        delta = (corrected[sel] - predicted[sel])
        shifts[i] = delta.sum() /(sel.sum()+shift_lambda) * shift_factor
        shift_vec[sel] = shifts[i]
    return shifts,shift_vec

def get_farfield(es_input:ESInput,es_sources:ESBlockSources,es_results:ESResults,es_problem,inner=2,outer=2,use_turbo=False,
                verbose=False):
    dipoles = es_sources.dipoles
    igrf_NED_dip = es_sources.igrf_NED_dip
    block_definition = es_sources.block_definition
    block_dip = (es_sources.block_row.flatten(),es_sources.block_col.flatten())
    if use_turbo:
        func_to_call = equi_cart.calculate_far_field_turbo
    else:
        func_to_call = equi_cart.calculate_far_field

    if verbose:
        timer = ETA(block_definition.block_centers()[0].shape[0]*block_definition.block_centers()[0].shape[1])
        callback = timer
    else:
        callback = lambda:None

    aeromag_farfield = func_to_call(es_problem.worldmap,block_definition,dipoles,es_input.lonlatz,
                                                igrf_NED_dip,es_input.igrf_NED_stat,
                                                es_results.equivalent_sources,
                                                inner=inner,outer=outer,stations_epochs=es_problem.stations_epochs,
                                                block_dip=block_dip,callback=callback)
    return aeromag_farfield

def misfit_stats(pred,obs):
    return np.sqrt(np.mean(np.square(pred-obs)))

def run_nouveau(inner_function,spacings,settings:ESSettings,es_input:ESInput,worldmap,repetitions:int,
                verbose=False,use_farfield=True,shift_factor=1,shift_lambda=0,previous=0,shift_vec=0,prior=None,
                setting_overrides=None):
    """Meta-function to run loop over several blocks
    Parameters
    ----------
    inner_function : function
        Signature of the function: f(ESProblem, spacing, year_shift, prior)
        -> es_sources,es_results
    setting_overrides : list of dict
        Values for settings which are supposed to be overriden for different spacings. This is
        mainly intended to include variable lambda_s or lambda_d
    """
    def say(s):
        if verbose:
            print(s)
    results = []
    shift_mem = []

    es_problem = ESProblem(settings,es_input,worldmap)

    sources = []
    farfields = []

    if setting_overrides is None:
        setting_overrides = [dict() for k in range(len(spacings))]

    for i,spacing in enumerate(spacings):
        say('Starting spacing %d m' % spacing)        
        for key in setting_overrides[i]:
            es_problem.settings.__dict__[key] = setting_overrides[i][key]
            say('Overriding setting %s with value %s'% (key,str(setting_overrides[i][key])))
        sub_results = []
        sub_farfields = []
        aeromag_farfield = np.zeros(es_input.N_data)
        shift_mem = []
        for k in range(repetitions):
            es_sources,es_results = inner_function(es_problem,spacing,year_shift=previous+shift_vec+aeromag_farfield,prior=prior)
            if k == 0:
                say('Block shape %d x %d'%(es_sources.block_definition.block_centers()[0].shape))                    
            # TODO: Calculate shift *before* updating aeromag farfield? (propbably makes no difference whether you do it before or after?)
            shifts,shift_vec = get_shift(es_input,es_results.predicted,aeromag_farfield+previous,shift_factor=shift_factor,shift_lambda=shift_lambda)
            shift_mem.append(shifts)
            # Update Farfield
            say('Calculating far field effects ...')
            aeromag_farfield = get_farfield(es_input,es_sources,es_results,es_problem,outer=settings.far_field_outer)
            say("Minimum and maximum shift %.1f %.1f" % (shifts.min(),shifts.max()))
            sub_results.append(es_results)
            sub_farfields.append(aeromag_farfield)
            say('RMS misfit %.1f' % misfit_stats(previous+shift_vec+aeromag_farfield+es_results.predicted,es_input.TFA))
        sources.append(es_sources)
        results.append(sub_results)
        farfields.append(sub_farfields)
        shift_mem.append(shifts)
        previous += sub_results[-1].predicted + sub_farfields[-1]
    return es_problem,sources,results,shift_mem,shift_vec,farfields

def block_analysis(aeromag,dipoles,block_definition,worldmap):
    station_count = block_definition.count(*worldmap(aeromag[:,0],aeromag[:,1]))
    dipole_count = block_definition.count(*worldmap(dipoles[:,0],dipoles[:,1]))

    return block_definition,station_count,dipole_count

def calc_aux(settings,sources,results,es_problem,outer=6,fixed_blockdefinition=-1):
    """Calculate auxiliary field
    Helper function copied from paper_runs.ipynb
    I'm not sure why they exist, but they are sometimes used...
    """
    worldmap = es_problem.worldmap
    spacings = [r[0].spacing for r in results]
    aux_field_far_fields = []
    for t,spacing in enumerate(spacings):
        es_sources = sources[t]
        es_results = results[t][-1]
        dipoles,igrf_NED_dip = es_sources.dipoles,es_sources.igrf_NED_dip
        if fixed_blockdefinition == -1:
            block_definition = equi_cart.CartesianBlockDefinition(settings.corners,settings.block_factor*spacing)
        else:
            block_definition = equi_cart.CartesianBlockDefinition(settings.corners,settings.block_factor*spacings[fixed_blockdefinition])
        sub = []
        for i in range(outer):
            aux_field_recalc = equi_cart.calculate_far_field(worldmap,block_definition,dipoles,es_problem.auxiliary,
                                                        igrf_NED_dip[es_problem.aux_epoch],es_problem.igrf_NED_aux,
                                                        es_results.equivalent_sources,
                                                        inner=i,outer=i)
            aux_field_recalc = aux_field_recalc.reshape(es_problem.aux_shape)
            sub.append(aux_field_recalc)
        aux_field_far_fields.append(sub)
    aux_field_far_fields = np.array(aux_field_far_fields)
    return aux_field_far_fields

def recalc(settings,sources,results,es_problem,outer=6):
    """Recalculate data at location of es_input
    Helper function copied from paper_runs.ipynb
    I'm not sure why they exist, but they are sometimes used...
    """
    worldmap = es_problem.worldmap
    ## Recalc aux_field
    spacings = [r[0].spacing for r in results]
    aux_field_far_fields = []
    for t,spacing in enumerate(spacings):
        es_sources = sources[t]
        es_results = results[t][-1]
        dipoles,igrf_NED_dip = es_sources.dipoles,es_sources.igrf_NED_dip
        block_definition = equi_cart.CartesianBlockDefinition(settings.corners,settings.block_factor*spacing)
        sub = []
        for i in range(outer):
            aux_field_recalc = equi_cart.calculate_far_field(worldmap,block_definition,dipoles,es_problem.input_data.lonlatz,
                                                        igrf_NED_dip[es_problem.aux_epoch],es_problem.igrf_NED_stat,
                                                        es_results.equivalent_sources,
                                                        inner=i,outer=i)
            sub.append(aux_field_recalc)
        aux_field_far_fields.append(sub)
    aux_field_far_fields = np.array(aux_field_far_fields)
    return aux_field_far_fields


def store_results(prefix,results,shift_mem,farfields):
    """Store all results which cannot be straightforwardly recovered from the settings
    """
    # results: equivalent_sources,predicted,self_predicted
    # shift_mem
    # farfields

    for i in range(len(results)):
        for j in range(len(results[i])):
            fname = '%s_eq.%d.%d.npy' % (prefix,i,j)
            np.save(fname,results[i][j].equivalent_sources)

            fname = '%s_pred.%d.%d.npy' % (prefix,i,j)
            np.save(fname,results[i][j].predicted)

            fname = '%s_spred.%d.%d.npy' % (prefix,i,j)
            np.save(fname,results[i][j].self_predicted)

    fname = '%s_shm.npy' % (prefix)
    np.save(fname,np.array(shift_mem))

    fname = '%s_far.npy' % (prefix)
    np.save(fname,np.array(farfields))

def load_results(prefix):
    fname = '%s_shm.npy' % (prefix)
    shift_mem = np.load(fname)

    fname = '%s_far.npy' % (prefix)
    farfields = np.load(fname)

    N,M = farfields.shape[0],farfields.shape[1]

    results = []

    for i in range(N):
        sub_results = []
        for j in range(M):
            fname = '%s_eq.%d.%d.npy' % (prefix,i,j)
            eq = np.load(fname)

            fname = '%s_pred.%d.%d.npy' % (prefix,i,j)
            pred = np.load(fname)

            fname = '%s_spred.%d.%d.npy' % (prefix,i,j)
            self_pred = np.load(fname)

            sub_results.append(ESResults(None,eq,pred,self_pred,None))
        results.append(sub_results)

    return results,shift_mem,farfields

def post_load_results(settings,spacings,results,es_input,worldmap):
    """Generate the data about sources to fill in results sources objects
    """
    es_problem = ESProblem(settings,es_input,worldmap)
    sources = []

    for i,spacing in enumerate(spacings):
        sources.append(es_problem._create_block_sources(spacing))
        for j in range(len(results[i])):
            results[i][j].spacing = spacing
            results[i][j].dipoles = sources[-1].dipoles
    
    return es_problem,sources,results

def get_pairwise_cross_distance(lon1,lat1,lon2,lat2):
    N1 = len(lon1)
    N2 = len(lon2)
    
    coslat1 = np.cos(lat1/180.0*np.pi)
    sinlat1 = np.sin(lat1/180.0*np.pi)

    coslat2 = np.cos(lat2/180.0*np.pi)
    sinlat2 = np.sin(lat2/180.0*np.pi)
    
    pd = np.zeros((N1,N2))
    for i in range(N1):
        dx = lon2 - lon1[i]
        cosdx = np.cos(dx/180.0*np.pi)
        pd[i,:] = coslat1[i] * coslat2 * cosdx + sinlat1[i] * sinlat2
    pd[pd>1] = 1
    pd = 180.0/np.pi*np.arccos(pd)
    return pd
    
def get_ano_design_matrix_sparse(sources : ESBlockSources,es_input : ESInput,worldmap,max_dist : float,stations_epochs : np.ndarray):
    """Construct a sparse forward matrix using blocks to speed up the distance calculations
    """
    ## TODO: At the moment, it is not ensured that the block size is sufficient for the given max_dist
    block_definition = sources.block_definition
    block_stat = block_definition.assign(*worldmap(es_input.lonlatz[:,0],es_input.lonlatz[:,1]))
    block_dip = (sources.block_row.flatten(),sources.block_col.flatten())
    # Construct init-arrays for coo-matrix
    data = []
    row_indices = []
    col_indices = []

    fullness = np.zeros(block_definition.block_centers()[0].shape)

    for i,j in block_definition:
        in_block_stat = (block_stat[0] == i) & (block_stat[1] == j)
        in_big_block_dip = (np.abs(block_dip[0] - i ) <= 1) & (np.abs(block_dip[1] - j ) <= 1)
        sub_matrix = equi.calculate_ano_design_matrix_multi_year(es_input.lonlatz[in_block_stat],sources.dipoles[in_big_block_dip],
                es_input.igrf_NED_stat[in_block_stat],sources.igrf_NED_dip[:,in_big_block_dip],stations_epochs[in_block_stat])
        jj,ii = np.meshgrid(np.where(in_big_block_dip)[0],np.where(in_block_stat)[0])

        pd = get_pairwise_cross_distance(es_input.lonlatz[in_block_stat][:,0],es_input.lonlatz[in_block_stat][:,1],sources.dipoles[in_big_block_dip,0],sources.dipoles[in_big_block_dip,1])
        in_radius = (pd*110e3 < max_dist)
        fullness[i,j] = in_radius.sum()/pd.size
        sub_matrix = sub_matrix[in_radius]
        ii = ii[in_radius]
        jj = jj[in_radius]

        data.extend(sub_matrix.flatten())
        row_indices.extend(ii.flatten())
        col_indices.extend(jj.flatten())

    design_matrix_sparse = sparse.csr_matrix(sparse.coo_matrix((data,(row_indices,col_indices)),shape=(es_input.N_data,sources.dipoles.shape[0])))
    return design_matrix_sparse,fullness

class PerformanceProfiler:
    def __init__(self):
        self.starts = dict()
        self.ends = dict()

    def start(self,code):
        self.starts[code] = time.time()

    def end(self,code):
        self.ends[code] = time.time()

    def get_times(self):
        times = dict()
        for k in self.starts:
            if k in self.ends:
                times[k] = self.ends[k]-self.starts[k]
        return times

    def merge(self,other):
        for k in other.starts:
            self.starts[k] = other.starts[k]
        for k in other.ends:
            self.ends[k] = other.ends[k]


class ESBlockRunner:
    @staticmethod
    def from_problem(problem:ESProblem,spacing:float,year_shift=None,prior=None):
       sources = problem._create_block_sources(spacing)
       new_obj = ESBlockRunner(problem.input_data,sources,problem.settings,problem.stations_epochs,problem.worldmap,year_shift=year_shift,prior=prior)   
       return new_obj
    
    def __init__(self,input_data,sources,settings,stations_epochs,worldmap,year_shift=None,prior=None):
        self.input_data = input_data
        self.sources = sources
        self.settings = settings
        self.stations_epochs = stations_epochs
        self.worldmap = worldmap
        if year_shift is None:  
            self.year_shift = np.zeros(self.input_data.lonlatz.shape[0])
        else:
            self.year_shift = year_shift
        if prior is None:
            self.prior = None
        else:
            self.prior = prior.copy()
        
        self.aeromag = np.concatenate((input_data.lonlatz,input_data.TFA[:,None]),axis=1)
        self.block_stat = sources.block_definition.assign(*self.worldmap(input_data.lonlatz[:,0],input_data.lonlatz[:,1]))
        self.block_dip =(sources.block_row.flatten(),sources.block_col.flatten())



    @property
    def dipoles(self):
        return self.sources.dipoles
    
    @property
    def igrf_NED_dip(self):
        return self.sources.igrf_NED_dip
    
    @property
    def aeromag_year(self):
        return self.input_data.year

    @property
    def err(self):
        return self.input_data.err
    
    @property
    def igrf_NED_stat(self):
        return self.input_data.igrf_NED_stat
    
    def get_dipoles_block(self,i,j,big=False):
        if big:
            sel = (np.abs(self.block_dip[0] - i ) <= 1) & (np.abs(self.block_dip[1] - j ) <= 1)
        else:
            sel = (self.block_dip[0] == i) & (self.block_dip[1] == j)

        return self.dipoles[sel]
    
    def get_sub_prior(self,i,j,big=True):
        if self.prior is None:
            sub_prior = None
        else:
            if big:
                sel = (np.abs(self.block_dip[0] - i ) <= 1) & (np.abs(self.block_dip[1] - j ) <= 1)
            else:
                sel = (self.block_dip[0] == i) & (self.block_dip[1] == j)
            sub_prior = self.prior[sel].copy()
            if self.settings.blank_center:
                sub_prior[(self.block_dip[0][sel] == i) & (self.block_dip[1][sel] == j) ] = 0
        return sub_prior

    def solve_block_extend_timing(self,i,j):
        """Solve for a single block and report timings for each step
        """
        profiler = PerformanceProfiler()
        
        block_dip = self.block_dip
        block_stat = self.block_stat
        
        in_big_block_dip = (np.abs(block_dip[0] - i ) <= 1) & (np.abs(block_dip[1] - j ) <= 1)
        in_big_block_stat = (np.abs(block_stat[0] - i ) <= 1) & (np.abs(block_stat[1] - j ) <= 1)

        sub_prior = self.get_sub_prior(i,j,True)
        profiler.start('A')
        ano_design_matrix = self.get_block_ano_design_matrix(i,j)
        profiler.end('A')
        
        satan = (np.abs(self.sources.block_row[:,0] - i ) <= 1)
        satan_2 = (np.abs(self.sources.block_col[0,:] - j ) <= 1)
        C_x,C_y = equi.build_1st_order_diffmatrix(satan_2.sum(),satan.sum())
        assert (satan.sum()*satan_2.sum()) == in_big_block_dip.sum()

        TFA = self.aeromag[in_big_block_stat,3] - self.year_shift[in_big_block_stat]
        err = self.err[in_big_block_stat]
        
        sol,sub_profiler = self.solve_matrix(ano_design_matrix,TFA,err,self.settings.lambda_d,self.settings.lambda_s,C_x,C_y,sub_prior,timing=True)
            
        profiler.merge(sub_profiler)

        in_center = (block_dip[0][in_big_block_dip] == i) & (block_dip[1][in_big_block_dip] == j)
        return sol,in_center,profiler,in_big_block_dip.sum(),in_big_block_stat.sum()


    def solve_block_extend(self,i,j):
        block_dip = self.block_dip
        block_stat = self.block_stat
        
        in_big_block_dip = (np.abs(block_dip[0] - i ) <= 1) & (np.abs(block_dip[1] - j ) <= 1)
        in_big_block_stat = (np.abs(block_stat[0] - i ) <= 1) & (np.abs(block_stat[1] - j ) <= 1)

        sub_prior = self.get_sub_prior(i,j,True)

        satan = (np.abs(self.sources.block_row[:,0] - i ) <= 1)
        satan_2 = (np.abs(self.sources.block_col[0,:] - j ) <= 1)
        C_x,C_y = equi.build_1st_order_diffmatrix(satan_2.sum(),satan.sum())
        assert (satan.sum()*satan_2.sum()) == in_big_block_dip.sum()


        ano_design_matrix,temp = single_block_inner(self.dipoles[in_big_block_dip],self.aeromag[in_big_block_stat],
                                                    self.igrf_NED_dip[:,in_big_block_dip],self.igrf_NED_stat[in_big_block_stat],
                                                    self.settings.lambda_d,self.settings.lambda_s,year_shift=self.year_shift[in_big_block_stat],
                                                    stations_epochs = self.stations_epochs[in_big_block_stat],err = self.err[in_big_block_stat],
                                                    prior = sub_prior,C_x=C_x,C_y=C_y,lambda_factor_override=self.settings.lambda_factor_override)
        in_center = (block_dip[0][in_big_block_dip] == i) & (block_dip[1][in_big_block_dip] == j)
        return temp,in_center
    
    def run_extend(self,update_prior=False,block_order = None):
        solution = np.zeros(self.dipoles.shape[0])
        if block_order is None:
            block_order = self.sources.block_definition
        for i,j in block_order:
            in_block_dip = (self.block_dip[0] == i) & (self.block_dip[1] == j)
            in_block_stat = (self.block_stat[0] == i) & (self.block_stat[1] == j)
            
            fill_ratio = in_block_stat.sum() / in_block_dip.sum()
            if in_block_dip.sum() == 0 or in_block_stat.sum() == 0:
                continue
            elif fill_ratio < self.settings.fill_threshold:
                continue
            sol,in_center = self.solve_block_extend(i,j)
            solution[in_block_dip] = sol[in_center]
            if update_prior:
                self.prior[in_block_dip] = sol[in_center]
        return solution

    def get_block_ano_design_matrix(self,i,j):
        block_dip = self.block_dip
        block_stat = self.block_stat
        
        in_big_block_dip = (np.abs(block_dip[0] - i ) <= 1) & (np.abs(block_dip[1] - j ) <= 1)
        in_big_block_stat = (np.abs(block_stat[0] - i ) <= 1) & (np.abs(block_stat[1] - j ) <= 1)

        aeromag = self.aeromag[in_big_block_stat]
        igrf_NED_stat = self.igrf_NED_stat[in_big_block_stat]
        stations_epochs = self.stations_epochs[in_big_block_stat]

        dipoles = self.dipoles[in_big_block_dip]        
        igrf_NED_dip = self.igrf_NED_dip[:,in_big_block_dip]
        
        
        if igrf_NED_dip.ndim == 3:
            ano_design_matrix = equi.calculate_ano_design_matrix_multi_year(aeromag,dipoles,
                                                    igrf_NED_stat,igrf_NED_dip,stations_epochs)
        else:
            ano_design_matrix = equi.calculate_ano_design_matrix(aeromag,dipoles,
                                                    igrf_NED_stat,igrf_NED_dip)
        return ano_design_matrix
    
    def get_cross_ano_design_matrix(self,i1,j1,i2,j2):
        """Calculate design matrix for stations and dipoles in different blocks
        i1,j1 -> Station blocks
        i2,j2 -> Dipole blocks
        """
        block_dip = self.block_dip
        block_stat = self.block_stat
        
        in_block_dip = (np.abs(block_dip[0] - i2 ) <= 0) & (np.abs(block_dip[1] - j2 ) <= 0)
        in_block_stat = (np.abs(block_stat[0] - i1 ) <= 0) & (np.abs(block_stat[1] - j1 ) <= 0)

        aeromag = self.aeromag[in_block_stat]
        igrf_NED_stat = self.igrf_NED_stat[in_block_stat]
        stations_epochs = self.stations_epochs[in_block_stat]

        dipoles = self.dipoles[in_block_dip]        
        igrf_NED_dip = self.igrf_NED_dip[:,in_block_dip]
        
        
        if igrf_NED_dip.ndim == 3:
            ano_design_matrix = equi.calculate_ano_design_matrix_multi_year(aeromag,dipoles,
                                                    igrf_NED_stat,igrf_NED_dip,stations_epochs)
        else:
            ano_design_matrix = equi.calculate_ano_design_matrix(aeromag,dipoles,
                                                    igrf_NED_stat,igrf_NED_dip)
        return ano_design_matrix

    def solve_matrix(self,ano_design_matrix,TFA,err,lambda_d,lambda_s,C_x=None,C_y=None,prior=None,timing=False):
        profiler = PerformanceProfiler()
        G = ano_design_matrix / err[:,None]

        if self.settings.lambda_factor_override is None:
            #lambda_factor = (np.abs(G).sum(1).mean(0))**2
            lambda_factor = np.abs(G.T.dot(G)).sum(1).mean(0)
        else:
            lambda_factor = self.settings.lambda_factor_override
        profiler.start('ATA')
        big_matrix = G.T.dot(G) + np.eye(G.shape[1]) * lambda_d * lambda_factor
        profiler.end('ATA')
        profiler.start('lambda_s')
        if lambda_s > 0:
            big_matrix += lambda_s * lambda_factor * (C_x.T.dot(C_x)  + C_y.T.dot(C_y))
        profiler.end('lambda_s')
        if not prior is None:
            rhs = TFA - ano_design_matrix.dot(prior)
            prior_x = prior
        else:
            rhs = TFA
            prior_x = 0
        profiler.start('rhs')
        actual_rhs = G.T.dot(rhs/err)
        profiler.end('rhs')
        profiler.start('solve')
        temp = np.linalg.solve(big_matrix,actual_rhs) + prior_x
        profiler.end('solve')
        
        if timing is True:
            return temp,profiler
        else:
            return temp

    
    def solve_block_extend_crossval(self,i,j,training_frac,lambda_vals,repetitions=1):
        """Solve inverse system for different values of lambda_d and lambda_s
        """
        block_dip = self.block_dip
        block_stat = self.block_stat
        
        in_big_block_dip = (np.abs(block_dip[0] - i ) <= 1) & (np.abs(block_dip[1] - j ) <= 1)
        in_big_block_stat = (np.abs(block_stat[0] - i ) <= 1) & (np.abs(block_stat[1] - j ) <= 1)

        in_center = (block_dip[0][in_big_block_dip] == i) & (block_dip[1][in_big_block_dip] == j)

        sub_prior = self.get_sub_prior(i,j,True)
        ano_design_matrix = self.get_block_ano_design_matrix(i,j)
        
        trainings = np.zeros((in_big_block_stat.sum(),repetitions),dtype=bool)
            
        for k in range(repetitions):
            temp = np.random.choice(in_big_block_stat.sum(),int(in_big_block_stat.sum()*training_frac),replace=False)
            trainings[temp,k] = True
        validations = ~trainings

        satan = (np.abs(self.sources.block_row[:,0] - i ) <= 1)
        satan_2 = (np.abs(self.sources.block_col[0,:] - j ) <= 1)
        C_x,C_y = equi.build_1st_order_diffmatrix(satan_2.sum(),satan.sum())
        assert (satan.sum()*satan_2.sum()) == in_big_block_dip.sum()

        TFA = self.aeromag[in_big_block_stat,3] - self.year_shift[in_big_block_stat]
        err = self.err[in_big_block_stat]
        
        lambda_factors = np.zeros(repetitions)
        
        cross_pred = np.zeros((len(lambda_vals),repetitions,validations[:,0].sum()))
        pred = np.zeros((len(lambda_vals),repetitions,trainings[:,0].sum()))
        sols = np.zeros((len(lambda_vals),repetitions,ano_design_matrix.shape[1]))
        for k in range(repetitions):
            training = trainings[:,k]
            validation = validations[:,k]
            G = ano_design_matrix[training] / err[training,None]
            lambda_factors[k] = np.abs(G.T.dot(G)).sum(1).mean(0)
            for i in range(len(lambda_vals)):
                lambda_d,lambda_s = lambda_vals[i]
                try:
                    sol = self.solve_matrix(ano_design_matrix[training],TFA[training],
                        err[training],lambda_d,lambda_s,C_x,C_y,sub_prior)
                    sols[i,k] = sol
                    cross_pred[i,k] = ano_design_matrix[validation].dot(sol) - TFA[validation]
                    pred[i,k] = ano_design_matrix[training].dot(sol) - TFA[training]
                except np.linalg.LinAlgError:
                    cross_pred[i,k] = np.inf
                    pred[i,k] = np.inf
                    sols[i,k] = np.nan

        return trainings,pred,cross_pred,lambda_factors,sols
    
    def calculate_far_field(self,i,j,equivalent_sources,outer,inner=2,verbose=False):
        """Calulate the far field effect from neighboring blocks onto stations
        """    
        block_definition = self.sources.block_definition
        in_block_stat = (np.abs(self.block_stat[0] - i ) <= 0) & (np.abs(self.block_stat[1] - j ) <= 0)
        far_field_stat = np.zeros(in_block_stat.sum())
        for i2,j2 in block_definition.neighbors(i,j,outer):
            di = np.abs(i2-i)
            dj = np.abs(j2-j)
            level = max(di,dj)
            if level < inner:
                continue

            A = self.get_cross_ano_design_matrix(i,j,i2,j2)
            in_block_dip = (np.abs(self.block_dip[0] - i2 ) <= 0) & (np.abs(self.block_dip[1] - j2 ) <= 0)
            far_field_stat+= A.dot(equivalent_sources.flatten()[in_block_dip])     
        return far_field_stat