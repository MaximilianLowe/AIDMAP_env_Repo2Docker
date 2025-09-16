import numpy as np
import equivalent_layer_model_v1 as equi

def corner_check(x,y,corners):
    return (x >= corners[0]) & (x <= corners[1]) & (y >= corners[2]) & (y <= corners[3]) 

class CartesianBlockDefinition:
    """Divides a rectangular area into blocks
    """
    def __init__(self,corners,block_size,buff=0):
        self.corners = np.zeros(4)
        self.corners[0] = corners[0] - buff
        self.corners[1] = corners[1] + buff
        self.corners[2] = corners[2] - buff
        self.corners[3] = corners[3] + buff
        
        nx = int(np.ceil((self.corners[1]-self.corners[0])/block_size))
        ny = int(np.ceil((self.corners[3]-self.corners[2])/block_size))

        self.left = self.corners[0] + np.arange(0,nx,dtype=int) * (self.corners[1]-self.corners[0])/nx
        self.right = self.left + (self.corners[1]-self.corners[0])/nx
        self.down = self.corners[2] + np.arange(0,ny,dtype=int) * (self.corners[3]-self.corners[2])/ny
        self.up = self.down + (self.corners[3]-self.corners[2])/ny
        
    def assign(self,x,y):
        """Determine block row and column coordinate for a set of points.
        """
        block_row = np.argmax(y[None,...]<self.up.reshape(self.up.shape+(1,)*y.ndim),axis=0)
        block_col = np.argmax(x[None,...]<self.right.reshape(self.right.shape+(1,)*x.ndim),axis=0)
        out_of_bounds = ~corner_check(x,y,self.corners)
        block_row[out_of_bounds] = -1
        block_col[out_of_bounds] = -1
        
        return block_row,block_col
    
    def assign_grid(self,x,y):
        """Determine block and row coordinates independently for x and y
        """
        row_y = np.argmax(y[None,:]<self.up[:,None],axis=0)
        col_x = np.argmax(x[None,:]<self.right[:,None],axis=0)
        block_col,block_row = np.meshgrid(col_x,row_y)
        
        out_x = ~((x>=self.corners[0]) & (x<=self.corners[1]))
        out_y = ~((y>=self.corners[2]) & (y<=self.corners[3]))
        out_of_bounds = (out_x[None,:] | out_y[:,None])

        block_col[out_of_bounds] = -1
        block_row[out_of_bounds] = -1

        return block_row,block_col
    
    def check_inside_extension(self,x,y,extension,block_i,block_j):
        """Check if points are inside a certain block, allowing for extension
        """
        left  = self.left[block_j] - extension
        right = self.right[block_j] + extension
        down  = self.down[block_i] - extension
        up    = self.up[block_i] + extension
        
        return corner_check(x,y,(left,right,down,up))
        
    def __iter__(self):
        return CartesianBlockIterator(self)
    
    def count(self,x,y):
        """Count how many of a set of points are in which block
        """
        block_row,block_col = self.assign(x,y)
        count = np.zeros((len(self.up),len(self.right)),dtype=int)
        for i,j in CartesianBlockIterator(self):
            count[i,j] += np.sum((block_row==i) & (block_col==j))
        return count
    
    def block_centers(self):
        """Calculate the center points of the blocks
        """
        return np.meshgrid(0.5*(self.right+self.left),0.5*(self.up+self.down))
    
    def neighbors(self,i,j,order):
        return NeighborIterator(self,i,j,order)
    
class CartesianBlockIterator:
    def __init__(self,block_definition):
        self.nx = len(block_definition.right)
        self.ny = len(block_definition.down)
        self.i,self.j = 0,0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        result = self.i,self.j
        if self.i == self.ny:
            raise StopIteration
        if self.j == self.nx-1:
            self.i = self.i + 1
            self.j = 0
        else:
            self.j = self.j + 1

        return result

class NeighborIterator:
    def __init__(self,block_definition,i0,j0,order):
        self.nx = len(block_definition.right)
        self.ny = len(block_definition.down)
        self.i0,self.j0 = i0,j0
        ii,jj = np.meshgrid(np.arange(i0-order,i0+order+1,1,dtype=int),np.arange(j0-order,j0+order+1,1,dtype=int))
        sel = (ii>=0)&(ii<=self.ny-1)&(jj>=0)&(jj<=self.nx-1)
        self.ii = ii[sel]
        self.jj = jj[sel]
        self.n = 0
        self.n_max = len(self.ii)
        self.order = order
    
    def __iter__(self):        
        return self
    
    def __next__(self):
        if self.n == self.n_max:
            raise StopIteration
        result = self.ii[self.n],self.jj[self.n]
        self.n = self.n +1
        return result


def equivalent_source_block_inversion(worldmap,block_definition,dipoles,aeromag,igrf_NED_dip,igrf_NED_stat,
                                    lambda_d,iterations=1,far_field = None,use_extension=False,verbose=False):

    block_stat = block_definition.assign(*worldmap(aeromag[:,0],aeromag[:,1]))
    block_dip = block_definition.assign(*worldmap(dipoles[:,0],dipoles[:,1]))
    
    if far_field is None:
        far_field = np.zeros(aeromag.shape[0])

    equivalent_sources = np.zeros(dipoles.shape[0])
    for i,j in block_definition:
        in_block_dip = (block_dip[0] == i) & (block_dip[1] == j)
        in_block_stat = (block_stat[0] == i) & (block_stat[1] == j)

        if in_block_dip.sum() == 0:
            continue
        elif in_block_stat.sum() == 0:
            continue

        if verbose:
            print(i,j,end = " ")

        if use_extension:
            in_big_block_dip = (np.abs(block_dip[0] - i ) <= 1) & (np.abs(block_dip[1] - j ) <= 1)
            in_big_block_stat = (np.abs(block_stat[0] - i ) <= 1) & (np.abs(block_stat[1] - j ) <= 1)
        else:
            in_big_block_dip = in_block_dip
            in_big_block_stat = in_block_stat

        ano_design_matrix = equi.calculate_ano_design_matrix(aeromag[in_big_block_stat],dipoles[in_big_block_dip],
                                                        igrf_NED_stat[in_big_block_stat],igrf_NED_dip[in_big_block_dip])

        lambda_factor = np.abs(ano_design_matrix.T.dot(ano_design_matrix)).sum(1).mean(0)
        big_matrix = ano_design_matrix.T.dot(ano_design_matrix) + np.eye(ano_design_matrix.shape[1]) * lambda_d * lambda_factor
        temp = np.linalg.solve(big_matrix,ano_design_matrix.T.dot(aeromag[in_big_block_stat,3]-far_field[in_big_block_stat]))
        
        rhs = aeromag[in_big_block_stat,3]-far_field[in_big_block_stat]
        for k in range(iterations):
            temp = temp+np.linalg.solve(big_matrix,ano_design_matrix.T.dot(rhs-ano_design_matrix.dot(temp)))
        temp2 = np.zeros(equivalent_sources.shape)
        temp2[in_big_block_dip] = temp
        equivalent_sources[in_block_dip] += temp2[in_block_dip]
    return equivalent_sources


def predict_at_auxilliary(worldmap,block_definition,dipoles,auxiliary,igrf_NED_dip,igrf_NED_aux,equivalent_sources,verbose=False):
    """Predict magnetic field at arbitrary locations
    This will only calculate the magnetic effect of sources on stations in the same block!
    No far-field will be calculated!
    """
    block_aux = block_definition.assign(*worldmap(auxiliary[:,0],auxiliary[:,1]))
    block_dip = block_definition.assign(*worldmap(dipoles[:,0],dipoles[:,1]))
    
    auxiliary_field = np.zeros(auxiliary.shape[0])
    for i,j in block_definition:
        in_block_dip = (block_dip[0] == i) & (block_dip[1] == j)
        in_block_aux = (block_aux[0] == i) & (block_aux[1] == j)

        if in_block_dip.sum() == 0:
            continue
        
        if verbose:
            print(i,j,end = " ")

        ano_design_matrix = equi.calculate_ano_design_matrix(auxiliary[in_block_aux],dipoles[in_block_dip.flatten()],
                                                        igrf_NED_aux[in_block_aux],igrf_NED_dip[in_block_dip.flatten()])
        auxiliary_field[in_block_aux] = ano_design_matrix.dot(equivalent_sources[in_block_dip])
    return auxiliary_field


def calculate_far_field(worldmap,block_definition,dipoles,aeromag,igrf_NED_dip,igrf_NED_stat,equivalent_sources,verbose=False,
                       inner=1,outer=1,stations_epochs=None,block_dip=None,callback=lambda:None):
    """Calulate the far field effect from neighboring blocks onto stations
    """
    if igrf_NED_dip.ndim == 3 and stations_epochs is None:
        raise ValueError('Missing stations epochs in multi-year calculation')
    
    far_field_stat = np.zeros(aeromag.shape[0])
    
    block_stat = block_definition.assign(*worldmap(aeromag[:,0],aeromag[:,1]))
    if block_dip is None:
        block_dip = block_definition.assign(*worldmap(dipoles[:,0],dipoles[:,1]))
    
    use_full = True
    blocks_with_stations = []
    
    for i,j in block_definition:
        in_block_stat = (block_stat[0] == i) & (block_stat[1] == j)
        if in_block_stat.sum() > 0:
            blocks_with_stations.append((i,j))

    print('In total there are %d blocks with stations'%len(blocks_with_stations))
    for i,j in block_definition:
        in_block_dip = (block_dip[0] == i) & (block_dip[1] == j)
        if in_block_dip.sum() == 0:
            continue
        if np.all(equivalent_sources[in_block_dip] == 0):
                continue
        for i2,j2 in blocks_with_stations:
            in_block_stat = (block_stat[0] == i2) & (block_stat[1] == j2)
            if in_block_stat.sum() == 0:
                continue
            di = np.abs(i2-i)
            dj = np.abs(j2-j)
            level = max(di,dj)
            
            if level < inner or level > outer:
                continue
            
            if verbose:
                print(i2,j2,end=" ")
            
            if use_full and in_block_stat.sum()>0:
                if igrf_NED_dip.ndim == 3:
                    ano_design_matrix = equi.calculate_ano_design_matrix_multi_year(aeromag[in_block_stat],dipoles[in_block_dip.flatten()],
                                                        igrf_NED_stat[in_block_stat],igrf_NED_dip[:,in_block_dip.flatten()],stations_epochs[in_block_stat])
                else:
                    ano_design_matrix = equi.calculate_ano_design_matrix(aeromag[in_block_stat],dipoles[in_block_dip.flatten()],
                                                        igrf_NED_stat[in_block_stat],igrf_NED_dip[in_block_dip.flatten()])
                

                far_field_stat[in_block_stat] += ano_design_matrix.dot(equivalent_sources[in_block_dip]) 

        if verbose:
            callback()
            print(' --',i,j)
    
    return far_field_stat


def calculate_far_field_turbo(worldmap,block_definition,dipoles,aeromag,igrf_NED_dip,igrf_NED_stat,equivalent_sources,verbose=False,
                       inner=1,outer=1,stations_epochs=None,block_dip = None,callback=lambda:None):
    """Calulate the far field effect from neighboring blocks onto stations

    This calculates all dipoles on the stations  in the same block. It's normally faster but it needs more ram
    """
    if igrf_NED_dip.ndim == 3 and stations_epochs is None:
        raise ValueError('Missing stations epochs in multi-year calculation')
    
    if block_dip is None:
        block_dip = block_definition.assign(*worldmap(dipoles[:,0],dipoles[:,1]))
    

    far_field_stat = np.zeros(aeromag.shape[0])
    
    block_stat = block_definition.assign(*worldmap(aeromag[:,0],aeromag[:,1]))
    
    
    for i2,j2 in block_definition:
        in_block_stat = (block_stat[0] == i2) & (block_stat[1] == j2)

        if in_block_stat.sum()==0:
            continue
        
        level = np.maximum(np.abs(block_dip[0] - i2),np.abs(block_dip[1]-j2))
        neighbor_dip = (level<=outer) & (level>=inner)

        if igrf_NED_dip.ndim == 3:
            ano_design_matrix = equi.calculate_ano_design_matrix_multi_year(aeromag[in_block_stat],dipoles[neighbor_dip.flatten()],
                                                igrf_NED_stat[in_block_stat],igrf_NED_dip[:,neighbor_dip.flatten()],stations_epochs[in_block_stat])
        else:
            ano_design_matrix = equi.calculate_ano_design_matrix(aeromag[in_block_stat],dipoles[neighbor_dip.flatten()],
                                                igrf_NED_stat[in_block_stat],igrf_NED_dip[neighbor_dip.flatten()])
            

        far_field_stat[in_block_stat] = ano_design_matrix.dot(equivalent_sources[neighbor_dip.flatten()]) 
        callback()

    return far_field_stat