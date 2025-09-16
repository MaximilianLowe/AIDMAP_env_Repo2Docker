import numpy as np
import numbers
import equivalent_layer_model as equi
import oo_implementation

HANDLED_FUNCTIONS = {}

def implements(numpy_function):
        def decorator(func):
            HANDLED_FUNCTIONS[numpy_function] = func
            return func
        return decorator

def has_shape(x):
    return hasattr(x,'shape')

class BlockMatrix:
    """Represents a block matrix
    """
    def __init__(self,ny,nx):
        self.internal = np.zeros((ny,nx),dtype=object)
        self.row_len = np.zeros(ny,dtype=int)
        self.col_len = np.zeros(nx,dtype=int)
        self.nonzero = np.zeros((ny,nx),dtype=bool)
        self.shape = ny,nx

    def __getitem__(self,index):
        return self.internal[index]
    
    def __setitem__(self,index,value):
        ## Check size
        i,j = index
        sub_shape = value.shape
        row_ok = (self.row_len[i] == 0) or (self.row_len[i]==sub_shape[0])
        col_ok = (self.col_len[j] == 0) or (self.col_len[j]==sub_shape[1])

        if row_ok and col_ok:
            self.internal[index]=value
            self.row_len[i]=sub_shape[0]
            self.col_len[j]=sub_shape[1]
            self.nonzero[i,j] = True
      

    def _dot_bv(self,bv):
        """Multiply with a block vector
        """
        Av = BlockVector(self.shape[0])
        for i in range(self.shape[0]):
            for j in np.where(self.nonzero[i,:])[0]:
                if self.nonzero[i,j]:
                    Av[i] += self[i,j].dot(bv[j])
        return Av
    
    def _dot_T_bv(self,bv):
        """Multiply transposed self with a block vector
        """
        Av = BlockVector(self.shape[1])
        for j in range(self.shape[1]):
            for i in np.where(self.nonzero[:,j])[0]:
                if self.nonzero[i,j]:
                    if isinstance(self[i,j],BlockMatrix):
                        Av[j] += self[i,j]._dot_T_bv(bv[i])
                    else:
                        Av[j] += self[i,j].T.dot(bv[i])
        return Av
    
    def dot(self,other):
        if isinstance(other,BlockVector):
            return self._dot_bv(other)
        else:
            raise NotImplementedError('Unknown type for dot',type(other))
    
    
    @implements(np.any)
    def any(self):
        return np.any(self.nonzero)

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle DiagonalArray objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
    
    @property
    def T(self):
        return BlockMatrixView(self,True)
    
class BlockMatrixView(BlockMatrix):
    def __init__(self,block_matrix,transpose=False):
        self.parent = block_matrix
        self.transpose = transpose
        self.shape = self.parent.shape
        self.nonzero = self.parent.nonzero
    
    def __setitem__(self,idx,value):
        raise ValueError('Views cannot set values')
    
    def __getitem__(self,index):
        i,j = index
        if self.transpose:
            if np.any(self.parent[j,i]):
                return self.parent[j,i].T
            else:
                return self.parent[j,i]
        else:
            return self.parent[i,j]
    
    def dot(self,other):
        if isinstance(other,BlockVector):
            return self.parent._dot_T_bv(other)
        else:
            return NotImplemented

class BlockVector:
    @staticmethod
    def from_raw(raw):
        shape = raw.shape
        block_vector = BlockVector(shape[0])
        block_vector.internal = raw
        ## Fill the length vectors
        for i in range(shape[0]):
            if np.any(raw[i]):
                ## This does not check for incompatible sizes
                block_vector.row_len[i]=raw[i].shape[0]
        return block_vector

    def __init__(self,n):
        self.internal = np.zeros((n),dtype=object)
        self.row_len = np.zeros(n,dtype=int)
        self.shape = (n,)
        self.nonzero = np.zeros(n,dtype=bool)
        
    def __getitem__(self,index):
        return self.internal[index]
    
    def __setitem__(self,index,value):
        ## Check size
        sub_shape = value.shape
        row_ok = (self.row_len[index] == 0) or (self.row_len[index]==sub_shape[0])
        
        if row_ok:
            self.internal[index]=value
            self.row_len[index]=sub_shape[0]
            self.nonzero[index] = True
    
    def assign(self,block_ix,data):
        for n in np.unique(block_ix):
            if n<0:
                continue
            self[n] = data[block_ix==n]
    
    def unblock(self,block_ix):
        data = np.zeros(len(block_ix))
        for n in np.unique(block_ix):
            data[block_ix==n] = self[n]
        return data

    def __add__(self,other):
        if isinstance(other,BlockVector):
            assert other.shape == self.shape
            result = BlockVector(self.shape[0])
            for i in range(self.shape[0]):
                if self.row_len[i]>0 or other.row_len[i]>0:
                    result[i] = self[i] + other[i]
        elif isinstance(other,numbers.Number):
            result = BlockVector(self.shape[0])
            for i in range(self.shape[0]):
                if self.row_len[i]>0:
                    result[i]= self[i] + other
        else:
            return NotImplemented
        return result

    def __radd__(self,other):
        return self+other

    def __mul__(self,other):
        if isinstance(other,numbers.Number):
            result = BlockVector(self.shape[0])
            for i in range(self.shape[0]):
                if self.row_len[i] > 0:
                    result[i] = self[i] * other
        else:
            return NotImplemented
        return result
    
    def __sub__(self,other):
        if isinstance(other,BlockVector):
            assert other.shape == self.shape
            result = BlockVector(self.shape[0])
            for i in range(self.shape[0]):
                if self.row_len[i] > 0 or other.row_len[i]>0:
                    result[i] = self[i] - other[i]
        elif isinstance(other,numbers.Number):
            result = BlockVector(self.shape[0])
            for i in range(self.shape[0]):
                if self.row_len[i] > 0:
                    result[i]= self[i] - other
        else:
            return NotImplemented
        return result
    
    def __rmul__(self,other):
        return self * other

    def flatten(self):
        """Turn a block vector into dense vector
        """
        temp = []
        for i in np.where(self.nonzero)[0]:
            if isinstance(self[i],BlockVector):
                temp.append(self[i].flatten())
            else:
                temp.append(self[i])
        return np.concatenate(temp)
    
def dot_bv_bv(bv1:BlockVector,bv2:BlockVector):
    """Calculate scalar product of two block vectors
    """
    assert bv1.shape == bv2.shape
    scalar_product = 0

    ii = np.where(bv1.nonzero)[0][0]
    if isinstance(bv1[ii],BlockVector):
        for i in range(bv1.shape[0]):
            scalar_product = scalar_product + dot_bv_bv(bv1[i],bv2[i])
    else:
        for i in range(bv1.shape[0]):
            scalar_product = scalar_product + np.dot(bv1[i],bv2[i])
    return scalar_product

def conj_grad_normal(bmat:BlockMatrix,b:BlockVector,n:int,lambda_factor=0,x0=None,recalc_r=False,r_tol=0,additional_returns=False):
    rms_mem = np.zeros(n)
    if x0 is None:
        r_old = b
    else:
        r_old = b-((bmat.T.dot(bmat.dot(x0))+lambda_factor*x0))
    d = r_old
    x = BlockVector(bmat.shape[1])
    if additional_returns:
        x_mem = []
    termination = 'max_iter'
    for k in range(n):
        z = bmat.T.dot(bmat.dot(d))+lambda_factor*d
        
        alpha = dot_bv_bv(r_old,r_old)/dot_bv_bv(d,z)
        dx = alpha*d
        x = x+dx
        if recalc_r:
            r_new = b-(bmat.T.dot(bmat.dot(x))+lambda_factor*x)
        else:
            r_new = r_old - alpha * z

        beta = dot_bv_bv(r_new,r_new)/dot_bv_bv(r_old,r_old)
        d  = r_new + beta * d
        r_old = r_new
        rms_mem[k] = np.sqrt(np.mean(np.square(r_old.flatten())))
        if additional_returns:
            x_mem.append(x)
        if r_tol > 0:
            rms_change = np.sqrt(np.mean(np.square(dx.flatten()))) / np.sqrt(np.mean(np.square(x.flatten())))
            if rms_change < r_tol:
                termination = 'r_tol'
                x_mem = x_mem[:k]
                break

    if additional_returns:
        return x,rms_mem,x_mem,termination
    else:
        return x
    

def init_block_matrix(spacing,es_problem:oo_implementation.ESProblem,bmat_outer = 2):
    es_input = es_problem.input_data
    worldmap = es_problem.worldmap
    sources = es_problem._create_block_sources(spacing)
    block_definition = sources.block_definition
    block_matrix = get_ano_design_block_matrix(sources,es_input,worldmap,es_problem.stations_epochs,outer=bmat_outer)
    
    return block_definition,block_matrix,sources

def init_block_data(es_input,block_matrix,sources:oo_implementation.ESBlockSources,worldmap,year_shift=None):
    if year_shift is None:
        year_shift = np.zeros(es_input.N_data)
    
    block_definition = sources.block_definition
    ## Assign data to blocks
    lonlatz = es_input.lonlatz
    block_stat = block_definition.assign(*worldmap(lonlatz[:,0],lonlatz[:,1]))
    block_ix = -np.ones(lonlatz.shape[0],dtype=int)
    for n,(i,j) in enumerate(block_definition):
        in_block_stat = (block_stat[0] == i) & ( block_stat[1] == j)
        block_ix[in_block_stat] = n
    if np.any(block_ix==-1):
        raise ValueError('Unassigned Stations remaining!')
    block_TFA = BlockVector(block_matrix.shape[0])
    block_TFA.assign(block_ix,es_input.TFA-year_shift)

    rhs = block_matrix.T.dot(block_TFA)

    ## Assign dipoles to blocks
    block_ix_dip = np.ones(sources.dipoles.shape[0],dtype=int)
    for n,(i,j) in enumerate(block_definition):
        in_block_dip = (sources.block_row.flatten() == i) & (sources.block_col.flatten() == j)
        block_ix_dip[in_block_dip] = n
    block_eqs = BlockVector(block_matrix.shape[1])
    block_eqs.assign(block_ix_dip,np.random.randn(sources.dipoles.shape[0]))
    
    return block_ix,block_ix_dip,rhs,block_TFA,block_eqs

def build_smoothing_matrix(sources:oo_implementation.ESBlockSources,lambda_s):
    ## Convert smoothing matrix into block matrix
    ## Here, ny and nx refers to the number of dipoles
    ny,nx = sources.block_row.shape
    block_definition = sources.block_definition
    n_blocks = block_definition.shape[0]* block_definition.shape[1]
    Cx =equi.build_1st_order_central_differences(nx,ny,direction=1)
    Cy =equi.build_1st_order_central_differences(nx,ny,direction=0)

    Cx_bmat = BlockMatrix(n_blocks,n_blocks)
    Cy_bmat = BlockMatrix(n_blocks,n_blocks)
    block_row,block_col = sources.block_row.flatten(),sources.block_col.flatten()
    for n,(i,j) in enumerate(block_definition):
        in_block_dip = (block_row == i) & (block_col == j)
        for i2,j2 in block_definition.neighbors(i,j,1):
            in_neighbor = (block_row == i2) & (block_col == j2)
            neighbor_ix = i2 * block_definition.shape[1] + j2
            if np.any(Cx[in_block_dip][:,in_neighbor].todense()):
                Cx_bmat[n,neighbor_ix] = Cx[in_block_dip][:,in_neighbor]  * lambda_s
            if np.any(Cy[in_block_dip][:,in_neighbor].todense()):
                Cy_bmat[n,neighbor_ix] = Cy[in_block_dip][:,in_neighbor] * lambda_s
    return Cx_bmat,Cy_bmat



def get_v_mean(v):
    S = 0
    N = 0
    for i in range(len(v)):
        if np.any(v[i]):
            S += v[i].sum()
            N += len(v[i])
    return S/N


def get_mean_colsum(bmat):
    col_sum = np.zeros(bmat.shape[0],dtype=object)
    for i in range(bmat.shape[0]):
        for j in range(bmat.shape[1]):
            if np.any(bmat[i,j]):
                col_sum[i] = col_sum[i] + np.sum(np.abs(bmat[i,j]),axis=1)
    return get_v_mean(col_sum)


def get_ano_design_block_matrix(es_block_sources : oo_implementation.ESBlockSources,es_input : oo_implementation.ESInput,worldmap,stations_epochs=None,
                                outer = 1,dry_run=False):
    lonlatz = es_input.lonlatz
    dipoles = es_block_sources.dipoles
    block_definition = es_block_sources.block_definition
    igrf_NED_dip = es_block_sources.igrf_NED_dip
    igrf_NED_stat = es_input.igrf_NED_stat
    block_stat = block_definition.assign(*worldmap(lonlatz[:,0],lonlatz[:,1]))
    block_dip = (es_block_sources.block_row.flatten(),es_block_sources.block_col.flatten())

    N = block_definition.block_centers()[0].size
    if dry_run:
        block_matrix = np.zeros((N,N,2),dtype=int)
    else:
        block_matrix = BlockMatrix(N,N)
    
    block_ix = np.zeros(block_definition.block_centers()[0].shape,dtype=int)
    for n,(i,j) in enumerate(block_definition):
        block_ix[i,j] = n
    
    for n,(i,j) in enumerate(block_definition):
        in_block_dip = (block_dip[0] == i) & (block_dip[1] == j)
        for di in np.arange(-outer,outer+1,dtype=int):
            for dj in np.arange(-outer,outer+1,dtype=int):
                i2 = i + di
                j2 = j + dj
                if i2<0 or j2<0 or i2>block_ix.shape[0]-1 or j2>block_ix.shape[1]-1:
                    continue
                m = block_ix[i2,j2]
                in_block_stat = (block_stat[0] == i2) & (block_stat[1] == j2)
                if in_block_stat.sum() == 0:
                    continue
                if dry_run:
                    block_matrix[m,n] = in_block_stat.sum(),in_block_dip.sum()
                    continue
                if igrf_NED_dip.ndim == 3:
                    ano_design_matrix = equi.calculate_ano_design_matrix_multi_year(lonlatz[in_block_stat],dipoles[in_block_dip],
                                                                igrf_NED_stat[in_block_stat],
                                                                igrf_NED_dip[:,in_block_dip],stations_epochs[in_block_stat])
                else:
                    ano_design_matrix = equi.calculate_ano_design_matrix(lonlatz[in_block_stat],dipoles[in_block_dip],
                                                                igrf_NED_stat[in_block_stat],igrf_NED_dip[in_block_dip])
                block_matrix[m,n] = ano_design_matrix.copy()
    return block_matrix


def eq_run_bmat(es_problem,dipole_spacing,year_shift=None,n_iter=100):
    if year_shift is None:
        year_shift = np.zeros(es_problem.N_data)
    block_definition,block_matrix,sources = init_block_matrix(dipole_spacing,es_problem)
    lambda_factor = get_mean_colsum(block_matrix.internal)
    Cx_bmat,Cy_bmat = build_smoothing_matrix(sources,lambda_factor*es_problem.settings.lambda_s)

    block_ix_stat,block_ix_dip,_,block_TFA,_ = init_block_data(es_problem.input_data,block_matrix,sources,es_problem.worldmap,year_shift)

    big_stack = BlockMatrix(3,1)
    big_stack[0,0] = block_matrix
    big_stack[1,0] = Cx_bmat
    big_stack[2,0] = Cy_bmat 

    rhs_stack = BlockVector(3)
    rhs_stack[0] = block_TFA
    rhs_stack[1] = BlockVector(block_matrix.shape[1])
    rhs_stack[2] = BlockVector(block_matrix.shape[1])

    for j in range(block_matrix.shape[1]):
        rhs_stack[1][j] = np.zeros(Cx_bmat.col_len[j])
        rhs_stack[2][j] = np.zeros(Cx_bmat.col_len[j])

    ATb = big_stack.T.dot(rhs_stack)
    
    inversion_cg,rms_mem,x_mem,termination = conj_grad_normal(big_stack,ATb,n_iter,additional_returns=True,lambda_factor=1e-99)
    inversion_cg = inversion_cg[0]
    predicted = block_matrix.dot(inversion_cg)
    return sources,oo_implementation.ESResults(sources.dipoles,inversion_cg.unblock(block_ix_dip),predicted.unblock(block_ix_stat),None,dipole_spacing)
