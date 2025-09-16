import numpy as np
import equivalent_layer_model as equi

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

        self.shape = ny,nx
        self.size = ny*nx
        
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
    
    def count(self,x,y,use_recursive=True):
        """Count how many of a set of points are in which block
        """
        block_row,block_col = self.assign(x,y)
        if use_recursive:
            return self.count_rec(block_row,block_col)
        else:
            count = np.zeros((len(self.up),len(self.right)),dtype=int)
            for i,j in CartesianBlockIterator(self):
                count[i,j] += np.sum((block_row==i) & (block_col==j))
            return count
    
    def count_rec(self,ii,jj,row_lims=None,col_lims=None,rec_lim=1,verbose=False):
        """Count how many of a set of points are in which block recursively
        """
        def say(*s):
            if verbose:
                print(*s)

        ny,nx = len(self.up),len(self.left)
        if row_lims is None:
            row_lims = 0,ny-1
        if col_lims is None:
            col_lims = 0,nx-1

        valid_points = (ii>=row_lims[0]) & (ii<=row_lims[-1]) & (jj>=col_lims[0]) & (jj<=col_lims[-1])
        say('Counting rows',row_lims,'Counting cols ',col_lims,'#points',len(ii),'#Valid',valid_points.sum())
        rows = np.arange(row_lims[0],row_lims[1]+1,dtype=int)
        cols = np.arange(col_lims[0],col_lims[1]+1,dtype=int)
        if max(len(rows),len(cols))<=rec_lim or min(len(rows),len(cols))==1:
            count = np.zeros((len(rows),len(cols)),dtype=int)
            for i,row in enumerate(rows):
                for j,col in enumerate(cols):
                    count[i,j] = np.sum((ii==row) & (jj==col))
            ## Actually do the counting
            say('actually counting',count.sum())
            return count
        else:
            rows_split = np.array_split(rows,2)
            cols_split = np.array_split(cols,2)
            sub_counts = []
            for i,sel_rows in enumerate(rows_split):
                for j,sel_cols in enumerate(cols_split):
                    sub_data = (ii>=sel_rows[0]) & (ii<=sel_rows[-1]) & (jj>=sel_cols[0]) & (jj<=sel_cols[-1])
                    sub_counts.append(self.count_rec(ii[sub_data],jj[sub_data],(sel_rows[0],sel_rows[-1]),(sel_cols[0],sel_cols[-1]),rec_lim,verbose))
            count = np.bmat([[sub_counts[0],sub_counts[1]] , [sub_counts[2],sub_counts[3]]])
            say('Recursive counting',count.sum(),valid_points.sum())
            return count
    
    def get_ij(self,n):
        if (n<=self.size-1) and (n>=0):
            i = n//self.shape[1]
            j = n%self.shape[1]
            return i,j
        else:
            raise ValueError('Invalid in n in get_ij')

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
    
    for i,j in block_definition:
        in_block_dip = (block_dip[0] == i) & (block_dip[1] == j)
        if in_block_dip.sum() == 0:
            continue

        for i2,j2 in block_definition:
            di = np.abs(i2-i)
            dj = np.abs(j2-j)
            level = max(di,dj)
            
            if level < inner or level > outer:
                continue
            
            if verbose:
                print(i2,j2,end=" ")
            in_block_stat = (block_stat[0] == i2) & (block_stat[1] == j2)
            if np.all(equivalent_sources[in_block_dip] == 0):
                continue
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