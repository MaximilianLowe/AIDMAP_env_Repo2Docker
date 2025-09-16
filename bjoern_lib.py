import numpy as np
import os
import equivalent_layer_model_v3j as equi

from scipy import sparse

class DataStructDefinition:
    # Make a data structure
    def __init__(self,data, date):
        
        nx, ny = data.shape
        
        #Survey Index
        self.survey_idx = data[:,0].astype(int) 
        
        #Longitude and Latitude (DD.dd), height in m (positive upward)
        self.longlatZ = data[:,1:4]
                
        #data value in nT (DGRF/IGRF corrected)
        self.val = data[:,4]
        
        #IGRF field in nT and deg
        self.IGRF_tot = data[:,5] #Total value
        self.IGRF_incl = data[:,6] #Inclination
        self.IGRF_decl = data[:,7] #Declination
        
        
        if ny > 8:
            self.error_idx = data[:,8].astype(int) #Index linked to an error estimate
        else :
            self.error_idx = np.ones(nx).astype(int)
        
        self.err = np.ones(nx) #assign a default error of 1.0 nT, if no errors are estimatted 
        
        #Date of data acquistion
        nx_date = date.size
        
        if nx_date != nx:
            print("WARNING! The number of lines are not the same for data and date")
        
        #acqusition date
        self.year = np.zeros(nx_date) # year
        self.month = np.zeros(nx_date) #months
        self.day = np.zeros(nx_date) #day

        for i in range (nx_date):
            line = np.fromstring(date[i], dtype= int, sep="/" )
            self.year[i] = line[0]
            self.month[i] = line[1]
            self.day[i] = line[2]
                        
        #Number of data points
        self.nr = len(self.val)
        
        #Predicted data (zero at the beginning)
        self.predicted = np.zeros(self.nr)
        
        #Assign the years of the used DGRFs
        self.nr_of_DGRF = 1          #set to one DGRF as default at initialization
        self.first_idx_DGRF = np.zeros([1])   # first index of the first (default) DGRF
        self.nr_idx_DGRF = np.array([self.nr])  #number of data points for the first (default) DGRF
        
    #rearrange the data by a header word (in ascending order)
    def sort_by_value(self, sort_data):
        sorted_idx = np.argsort(sort_data)
        
        self.survey_idx = self.survey_idx[sorted_idx]
        
        self.longlatZ = self.longlatZ[sorted_idx,:]
        self.val = self.val[sorted_idx]
        self.predicted = self.predicted[sorted_idx]
        
        self.error_idx = self.error_idx[sorted_idx]
        self.err = self.err[sorted_idx]
        
        self.IGRF_tot = self.IGRF_tot[sorted_idx]
        self.IGRF_incl = self.IGRF_incl[sorted_idx]
        self.IGRF_decl = self.IGRF_decl[sorted_idx]
        
        self.year = self.year[sorted_idx]
        self.month = self.month[sorted_idx]
        self.day = self.day[sorted_idx]
        
        if hasattr(self,"igrf_NED"):
            self.igrf_NED = self.igrf_NED[sorted_idx,:]
    
    #assign errors to the data
    def assign_errors(self, err_used_idx, err_var):
        
        nr = self.error_idx.size        
        nr_idx = err_var.size
                
        if err_used_idx != 0:
            
            nlist = 0
            
            #Ensure that the errors are positive
            list_zero = np.where(err_var <= 0.0)
            err_var[list_zero] = (err_var[list_zero]*(-1)) + 0.00001
            
            if err_var[list_zero].size > 0:
                print("WARNING! Some error estimates were either negative or zero- They are multiplied by -1 and added with 0.00001")
                
            for idx in range(nr_idx):
                liste = np.where(self.error_idx == (idx+1))  
                self.err[liste] = err_var[idx]
                
                nlist = nlist + self.err[liste].size
        
            if nlist != nr:
                print("WARNING: Not all values had a defined error:", nlist, "of", nr, "data have error information")
                print("An error of 1 nT are assumed for data without error assignment")
                
        # delattr(self,"error_idx")
    
    
    #adapt data range
    #ATTENTION - THE ASSIGNNEMT OF THE DGRF TO THE CORRECT INDECES IS MAY DESTROYED AND HAS TO BE REOBTAINED BY RE-RUNNING dat.assign_data_to_DGRF
    def resize(self,idx):
        
        self.survey_idx = self.survey_idx[idx]
        
        self.longlatZ = self.longlatZ[idx,:]
        self.val = self.val[idx]
        self.predicted = self.predicted[idx]
        
        self.error_idx = self.error_idx[idx]
        self.err = self.err[idx]
        
        self.IGRF_tot = self.IGRF_tot[idx]
        self.IGRF_incl = self.IGRF_incl[idx]
        self.IGRF_decl = self.IGRF_decl[idx]
        
        self.year = self.year[idx]
        self.month = self.month[idx]
        self.day = self.day[idx]
        
        if hasattr(self,"igrf_NED"):
            self.igrf_NED = self.igrf_NED[idx,:]
        
        self.nr = len(self.val)
   

    #determine the first data point and the number of data point for the different DGRFs associated with "years":
    #ATTENTION - the data have to be sorted by the acqusition year in ascending order in forhand!!
    def assign_data_to_DGRF(self, year):

        year_idx = np.argsort(year)
        
        self.nr_of_DGRF = len(year_idx)    #The number of used DGRFs
        self.year_of_DGRF = year           #The years of used DGRFs
        self.first_idx_DGRF = np.zeros(len(year_idx))  #The index of the first data point used for the associated DGRF
        self.nr_idx_DGRF    = np.ones(len(year_idx))   #The number of data points used for the associated DGRF
        
        min_idx_year = 0 #first data point associated with a DGRF
        
        for yr in range(len(year_idx)-1):
            
            last_year = np.floor((year[year_idx[yr]] + year[year_idx[yr+1]])/2.0) #find the last year that is associated with the DGRF of year "year[year_idx[yr]]" 

            #print("year:", year[year_idx[yr]], last_year)
            
            list_idx_year = self.year <= last_year
            list_idx_year = np.nonzero(list_idx_year)
            
            if np.any(list_idx_year):
                max_idx_year = np.max(list_idx_year) #last data point associated with a DGRF
            else:
                max_idx_year = -1
          
            #print("min/max_idx_year:", min_idx_year,  max_idx_year)
            
            self.first_idx_DGRF[yr] = min_idx_year                   
            self.nr_idx_DGRF[yr] = max_idx_year - min_idx_year +1
            
            if max_idx_year+1 < self.nr:
                min_idx_year = max_idx_year+1
            else:
                min_idx_year = self.nr
                             
        max_idx_year = self.nr -1
    
        #print("min/max_idx_year:", min_idx_year,  max_idx_year)
        
        self.first_idx_DGRF[len(year_idx)-1] = min_idx_year
        self.nr_idx_DGRF[len(year_idx)-1] = max_idx_year - min_idx_year +1
       
        self.first_idx_DGRF = self.first_idx_DGRF.astype(int)
        self.nr_idx_DGRF = self.nr_idx_DGRF.astype(int)
    
        print("Number of DGRFs:", self.nr_of_DGRF)
        print("Years of DGRFs:", self.year_of_DGRF)
        print("First index of each DGRF:", self.first_idx_DGRF)
        print("Number of data points in each DGRF:", self.nr_idx_DGRF)
            
        return year[year_idx]
    
    #Modify the assignment of the DGRF after re-sampling of the used data
    def adapt_assigment_DGRF_for_resampling(self, data_int):
        

        org_index_vector = np.zeros(self.nr)
        first_idx_DGRF_resamp = np.intc(np.zeros(self.first_idx_DGRF.size))
        nr_idx_DGRF_resamp = np.intc(np.zeros(self.nr_idx_DGRF.size))
        
        #Build a data vector with the associated DGRF as an index
        for i in range(self.nr_of_DGRF): #loop over all used DGRFs
            
            first_idx_DGRF_resamp[i] = self.first_idx_DGRF[i]
            nr_idx_DGRF_resamp[i] = self.nr_idx_DGRF[i]
                       
            if self.nr_idx_DGRF[i] > 0.0:
                tmp_idx_vec = i * np.ones(self.nr_idx_DGRF[i])
                org_index_vector[self.first_idx_DGRF[i]: self.first_idx_DGRF[i] + self.nr_idx_DGRF[i]] = tmp_idx_vec
        
        #Resample index vector
        idx_vec_resamp = org_index_vector[::data_int]
         
        #Adapt the data structure for resampling
        first_idx = 0
        
        for i in range(self.nr_of_DGRF): #loop over all used DGRFs
            
            list_vec_resampi = idx_vec_resamp == i
            
            first_idx_DGRF_resamp[i] = first_idx
            
            if np.any(list_vec_resampi):                
                nr_idx_DGRF_resamp[i] = idx_vec_resamp[list_vec_resampi].size 
                
                first_idx = first_idx + nr_idx_DGRF_resamp[i]
            else:
                nr_idx_DGRF_resamp[i] = 0
        

        print('Data_int', data_int)
        print('org first_idx_DGRF', self.first_idx_DGRF)
        print('mod first_idx_DGRF_resamp', first_idx_DGRF_resamp)
        print('org nr_idx_DGRF', self.nr_idx_DGRF)   
        print('mod nr_idx_DGRF_resamp', nr_idx_DGRF_resamp)   
        
        return first_idx_DGRF_resamp, nr_idx_DGRF_resamp

    
    #Export data:
    def export_data (self,  data_background, ddir, refine_iter, dipole_spacing):
        if not os.path.exists(ddir): #Create directory
            os.mkdir(ddir)
            
        str_filename_dat = "{0}Data_refinement{1:d}interval{2:d}.txt".format(ddir,refine_iter,dipole_spacing)
        
        #store: lomg, lat, height, observed data - modelled data already fitted, IGRF(tot, decl, incl)
        str_year= np.empty(self.year.size, dtype=object)
        for i in range(self.year.size):
            str_year[i] = "{0:d}/{1:d}/{2:d}".format(self.year[i].astype(int), self.month[i].astype(int), self.day[i].astype(int))
        
        dat_save = np.vstack((self.survey_idx, self.longlatZ[:,0], self.longlatZ[:,1], self.longlatZ[:,2],  data_background, self.IGRF_tot, self.IGRF_incl, self.IGRF_decl, self.error_idx, str_year)).T   
        
        np.savetxt(str_filename_dat,dat_save, delimiter=",",fmt="%d,%.5f,%.5f,%.5f,%.2f,%.2f,%.2f,%.2f,%d,%s") #safe to txt file
        
    

class GridStructDefinition:
    #Make uniform grid of dipoles/auxiliaries in given projection and transform it back to LongLat
    @staticmethod
    def from_spacing_height(worldmap,spacing,height):
        #make a rounding such that the dipole locations/auxiliaries locations are consistently located indpendent on the selected area 
        llcr_x = np.ceil(worldmap.llcrnrx/spacing)*spacing
        llcr_y = np.ceil(worldmap.llcrnry/spacing)*spacing
        urcr_x = np.floor(worldmap.urcrnrx/spacing)*spacing
        urcr_y = np.floor(worldmap.urcrnry/spacing)*spacing
            
        #x und y Positionen der Gridpunkte (in der Projektion)
        #x_proj = np.arange(worldmap.llcrnrx,worldmap.urcrnrx+spacing,spacing)
        #y_proj = np.arange(worldmap.llcrnry,worldmap.urcrnry+spacing,spacing)
        
        x_proj = np.arange(llcr_x, urcr_x + spacing,spacing)
        y_proj = np.arange(llcr_y, urcr_y + spacing,spacing)
        
        xi_proj,yi_proj = np.meshgrid(x_proj,y_proj)
        
        #x und y Positionen der Gridpunkte (in Long Lat)
        lon,lat = worldmap(xi_proj,yi_proj,inverse=True)
        
        data_grd = np.vstack((lon.flatten(),lat.flatten(),height*np.ones(lon.size),spacing*spacing*np.ones(lon.size))).T
        
        #Make grid structure
        grd = GridStructDefinition(data_grd)
        
        grd.x_proj = x_proj
        grd.y_proj = y_proj

        #setattr(grd, "lon", lon)
        #setattr(grd, "lat", lat)
        
        grd.nx = len(x_proj)
        grd.ny = len(y_proj)

        return  grd

    #Make a class for gridded data as dipoles and auxiliaries
    def __init__(self,grd_points): 
        #Input
        #grd_points: list of gridpoints (in long, lat, cell_sizes)
        
        #Nr of points
        self.nr = len(grd_points[:,0])
        
        #Longitude and Latitude (DD.dd), height in m (positive upward)
        self.longlatZ = grd_points[:,0:3]

        #sizes (surface) of the grid cells
        self.cell_size = grd_points[:,3]    
        
    #Determine the IGRF at grid locations
    def determine_IGRF(self,lon, lat, pysh, year, height):
        
        self.igrf_NED = np.zeros((len(year), len(height),3))
        
        for i in range(len(year)):

            igrf_coeffs = pysh.SHMagCoeffs.from_array(pysh.shio.read_igrf('igrf13coeffs.txt',year=year[i]),6371.2)
                    
            order = [1,2,0]
            # Note: pyshtools uses rad,theta,phi
            # Here I change to NED system

            height=6371.2+height*0.001
            
            if np.isscalar(height):
                tmp_igrf_NED = igrf_coeffs.expand(lon=lon.flatten(),lat=lat.flatten(),a=height) #IGRF an den Dipollokationen
            else:
                tmp_igrf_NED = igrf_coeffs.expand(lon=lon.flatten(),lat=lat.flatten(),a=height.flatten()) #IGRF an den Dipollokationen
            
            self.igrf_NED[i,:,:] = tmp_igrf_NED
            
        
        #Umsortieren nach Nord, Ost, down
        self.igrf_NED[:,:,0] = - self.igrf_NED[:,:,0]
        self.igrf_NED[:,:,1] = - self.igrf_NED[:,:,1]
        self.igrf_NED = self.igrf_NED[:,:,order]
        
    #Export model:
    def export_mod(self, str_type, ddir, refine_iter, dipole_spacing_org, dipole_spacing_curr, used_projection):
        
        if not os.path.exists(ddir): #Create directory
            os.mkdir(ddir)
        
        str_filename_mod = "{0}_refinement{1:d}interval{2:d}.txt".format(str_type,refine_iter,round(dipole_spacing_curr))
        str_path_mod = "{0}{1}".format(ddir,str_filename_mod)

        str_line1 = "Projection: {0}\n".format(used_projection) 
        str_line2 = "Iteration of refinement: {0:d}\n".format(refine_iter)
        str_line3 = "Original spacing: {0:d}m\n".format(round(dipole_spacing_org))
        str_line4 = "Current spacing: {0:d}m\n".format(round(dipole_spacing_curr))
        if hasattr(self,"igrf_NED"):
            str_line5 = "\n Long Lat height Model IGRF(N) IGRF(E) IGRF(Z)".format()
        else :
            str_line5 = "\n Long Lat height Model".format()
    
        str_header = "{0}{1}{2}{3}{4}".format(str_line1,str_line2,str_line3,str_line4,str_line5)
        
        
        #ATTENTION: Only the FIRST used IGRF model is stored - since it is not used in the later forward modelling, this is not problematic
        igrf_NED_used = np.squeeze(self.igrf_NED[0,:,:])
        
        if hasattr(self,"igrf_NED"):
            mod_save = np.vstack((self.longlatZ[:,0], self.longlatZ[:,1], self.longlatZ[:,2], self.final_mod,igrf_NED_used[:,0],igrf_NED_used[:,1],igrf_NED_used[:,2])).T
            np.savetxt(str_path_mod,mod_save, delimiter=",",fmt="%.5f,%.5f,%.5f,%.10f,%.5f,%.5f,%.5f", header=str_header) #safe to txt file
        else :
            mod_save = np.vstack((self.longlatZ[:,0], self.longlatZ[:,1], self.longlatZ[:,2], self.final_mod)).T
            np.savetxt(str_path_mod,mod_save, delimiter=",",fmt="%.5f,%.5f,%.5f,%.10f", header=str_header) #safe to txt file


        return  str_filename_mod  
    

def uniform_grid_in_proj(worldmap,spacing,height):
    return GridStructDefinition.from_spacing_height(worldmap,spacing,height)

#Read in a grid file with topography and determine the height for grid locations associated with auxiliaries or equivalent sources
#The projection can be different, but the projection of the grid file needs to be supported by pyproj

def determine_height_from_grdfile(topo_file, dummy_val, worldmap, grd):
    #Open the topography grid file
    raster = gdal.Open(topo_file) 
    
    #extract projection
    raster_proj = raster.GetProjection()
    #Prints size
    print("Size of raster:", raster.RasterXSize, raster.RasterYSize)
    #Determine EPSG
    print("EPSG:", raster.GetSpatialRef().GetAttrValue('AUTHORITY',1))
    
    #Gridding in x and y direction
    raster_xy = raster.GetGeoTransform()
    raster_x = np.arange(raster_xy[0],raster_xy[0] + (raster.RasterXSize)*raster_xy[1],raster_xy[1])
    raster_y = np.arange(raster_xy[3],raster_xy[3] + (raster.RasterYSize)*raster_xy[5],raster_xy[5])   
  
    #transfer grid to a numpy array
    rasterArray = raster.ReadAsArray()
    raster_X, raster_Y = np.meshgrid(raster_x,raster_y)
    raster_X = raster_X.flatten()
    raster_Y = raster_Y.flatten()
    raster_Z = rasterArray.flatten()

    #transfer grid to long lat 
    proj = Transformer.from_crs(raster_proj, 4326)
    raster_lat, raster_long = proj.transform(raster_X, raster_Y)      
     # ... and then to the defined projection
    raster_nX, raster_nY = worldmap(raster_long, raster_lat)
    
    #Set undefined values to zero
    condition = (raster_Z == np.min(raster_Z)) #HAS TO BE CHANGED IF DUMMIES ARE NOT THE MINIMUM VALUES
    raster_Z[condition] = dummy_val  #SET TO ZERO TOPOGRAPHY VALUES
    
    xi_proj,yi_proj = np.meshgrid(grd.x_proj,grd.y_proj)

    #Interpolate height data onto the grid
    zi_proj = spint.griddata((raster_nX, raster_nY), raster_Z, (xi_proj, yi_proj),  method='linear', fill_value=0.0)
    
    #Transfer data to z_coodinate 
    grd.longlatZ[:,2] = zi_proj.flatten()
    
    ############################
    #Plot original and re-gridded data
    plt.figure(figsize=(10,10))
    plt.scatter(raster_nX,raster_nY,150,raster_Z,'s')
    worldmap.pcolormesh(grd.x_proj,grd.y_proj,zi_proj,cmap=plt.cm.RdBu)
    ############################
    
    return grd


def add_shifts_cells_in_matrices_vec(dat_surv_idx, DG, G, mod, C_x, C_y, weight_shift, prev_model_shift_vec, unique_surv_idx_prev, lambda_s, sparse_idx):
    #Modify the matries and the model vector if "shifted" cells associated with corrections for constant arror shifts in the datasets    
    DG_NORM = 0 #if != 0 weight the contribution in the regularization with the error weighted data matrix  
    
    TYPE_REG = 0 # Type of regulrization, if ==0, it is constrained towards zero, but otherwise the negative datashifts from the previously calculated dipole layers
    
    nrow, ncol = np.shape(G) #number of rows and columns without shift cells (when available)
    
    unique_values,count=np.unique(dat_surv_idx,return_counts=True) #determine a list of involved datasets and their number of points
    nr_datasets = len(unique_values)
        
    print(" ")
    print("Correct datasets for shifts ")
    print("Survey IDX:", unique_values)
    print("Number of points in surveys:", count)
    print("Nr of surveys:", nr_datasets)
        
    #resize the model vector
    mod = np.pad(mod, (0,nr_datasets), 'constant', constant_values=(0))
    
    #Reference vector
    length_mod_shift = np.squeeze(prev_model_shift_vec.shape)    
    new_model_shift_vec = np.zeros(nr_datasets)
    
    length_unique_idx_prev = unique_surv_idx_prev.size #number of surveys involved in the calculation of the previous grod layer
    
    
    if DG_NORM != 0.0:     
        if sparse_idx == 0 :
            sens_col_all = np.abs(DG).sum(0).mean(0) #mean coverage of a data column
        else :
            sens_col_all = np.abs(DG).sum(0).mean(1) #mean coverage of a data column        
    else:
        if sparse_idx == 0 :
            sens_col_all = np.abs(G).sum(0).mean(0) #mean coverage of a data column
        else :
            sens_col_all = np.abs(G).sum(0).mean(1) #mean coverage of a data column
                
    #resize the Jacobian matrix by a "shift" cell for each dataset
    if sparse_idx == 0 :
        G = np.pad(G,((0,0),(0,nr_datasets)), 'constant', constant_values=(0)) #zero-padding if G is a dense matrix
    
    #... and create the associated regularization part of stabilizing the "shifts" towards 0 
    C_shift = sparse.lil_matrix((nr_datasets,ncol+nr_datasets)) #this matrix is always created as a sparse matrix
        
    #resize the regularization matrices
    if lambda_s > 0.0 : 
        nrow_regx, ncol_regx = np.shape(C_x)
        nrow_regy, ncol_regy = np.shape(C_y)            
        C_x.resize((nrow_regx,ncol_regx+nr_datasets)) #note that the regularization parts are always sparse matrices
        C_y.resize((nrow_regy,ncol_regy+nr_datasets))
        
    vec_sens_shift = np.zeros((nr_datasets,3))
        
        
    #Loop over all datasets
    for iter_dataset in range(nr_datasets): 
        
        row_list= np.argwhere(dat_surv_idx== unique_values[iter_dataset])  #find indeces/rows of associated data          
        row_list = np.squeeze(row_list)
        
        
        #Determine the average sensitivities of rows from a specific dataset  
        if row_list.size > 1 or sparse_idx != 0:
            sens_shift_org = np.abs(G[row_list,:ncol]).sum(1).mean(0)     #determine average weight of selected row, which is associated with the sensitivity of the corresponding shift cell
        else :
            sens_shift_org = np.abs(G[row_list,:ncol]).mean(0)
        
        sens_shift = sens_shift_org * weight_shift            # ... after multiplied with a scalar
                                    
        #Fill the G matrix with the shift cell sensitivites
        if sparse_idx == 0 :
            if row_list.size > 1 :
                for k in range(count[iter_dataset]):
                    G[row_list[k],ncol+iter_dataset] = sens_shift
            else :
                G[row_list,ncol+iter_dataset] = sens_shift
                
        else: # for sparse matrices a different way is chosen to speed up the program
            val_vec  = np.empty(shape=(1,0))
            row_vec =  np.empty(shape=(1,0))
            col_vec =  np.empty(shape=(1,0))
            
            ncount = count[iter_dataset]
            for k in range(ncount):
                val_vec = np.append(val_vec,sens_shift)
                col_vec = np.append(col_vec,0)

                if ncount != 1 :
                    row_vec = np.append(row_vec,row_list[k])
                else:
                    row_vec = np.append(row_vec,row_list)
           
            G_add = sparse.coo_matrix((val_vec, (row_vec, col_vec)), shape=(nrow,1)).tocsr() 
            G = sparse.hstack([G,G_add]).tocsr()
                        
        # Store the sensitivities + associated indices to a vector
        vec_sens_shift[iter_dataset,0] = unique_values[iter_dataset] #index of the dataset
        vec_sens_shift[iter_dataset,1] = sens_shift #the sensitivities will be later used to determine the data shifts
                        
        if DG_NORM != 0.0:
            if sparse_idx == 0 :
                if row_list.size > 1 :
                    sens_col = np.abs(DG[row_list,:]).sum(0).mean(0) #mean coverage of a data column
                else :
                    sens_col = np.abs(DG[row_list,:]).mean(0) #mean coverage of a data column
            else :
                sens_col = np.abs(DG[row_list,:]).sum(0).mean(1) #mean coverage of a data column

        else:
            if sparse_idx == 0 :
                if row_list.size > 1 :
                    sens_col = np.abs(G[row_list,:ncol]).sum(0).mean(0) #mean coverage of a data column
                else :
                    sens_col = np.abs(G[row_list,:ncol]).mean(0) #mean coverage of a data column
            else :
                sens_col = np.abs(G[row_list,:ncol]).sum(0).mean(1) #mean coverage of a data column
                

        #Fill the regularization part associated with the "shift"        
        C_shift[iter_dataset,ncol+iter_dataset] = (sens_col/sens_col_all) #regularization is scaled with the corresponding sensitivity of the data
        
        #.. and re-weight the reference vector:
        if length_unique_idx_prev != 0 & TYPE_REG != 0 :
            idx_prev = unique_surv_idx_prev == unique_values[iter_dataset] #find the index of the survey in the previous dipole layer
            
            tmp_mod = prev_model_shift_vec[idx_prev]
            
            if tmp_mod.size == 1 :
                if vec_sens_shift[iter_dataset,1] != 0.0:
                    new_model_shift_vec[iter_dataset] = tmp_mod/vec_sens_shift[iter_dataset,1] # modification of the model reference vector
                else:
                    new_model_shift_vec[iter_dataset] = tmp_mod
        
    #print(new_model_shift_vec)
   
    C_shift = C_shift.tocsr()
    
    if lambda_s > 0.0:
        return vec_sens_shift, mod, G, C_shift, new_model_shift_vec, C_x, C_y
    else:
        return vec_sens_shift, mod, G, C_shift, new_model_shift_vec

    
###########################################################
# add the datashift from the last iteration to determine the mod-reference for the next iteration
      
def calc_weight_nominator(vec_sens_shift, model_shift_vec_tot, unique_surv_idx_prev):

    nr_of_surv_idx, dummy  = vec_sens_shift.shape
    unique_surv_idx = vec_sens_shift[:,0] # the survey indeces
    
    nr_of_surv_idx_prev = unique_surv_idx_prev.size
    
    if model_shift_vec_tot.size == 0:
        model_shift_vec_tot = np.zeros(nr_of_surv_idx)
                    
    if nr_of_surv_idx  == nr_of_surv_idx_prev : 
        model_shift_vec_tot = model_shift_vec_tot - vec_sens_shift[:,2] #the model shift has to be weighted, since the total data component due to the shift should be small and not the actual model shift cells themselves
        mod_shift_idx = unique_surv_idx_prev
    else:  #to handle the exceptional case that the number of involved datasets change from one to the next dipole grid refinement
                
        print("ATTENTION: The total number of involved datasets changed from one to the next dipole spacing")
        
        tmp_vec_mod = np.zeros(nr_of_surv_idx)

        for iter_surv_idx in range(nr_of_surv_idx):
            same_idx = np.argwhere(unique_surv_idx_prev == unique_surv_idx[iter_surv_idx])
                                        
            if same_idx.size == 0:                    
                tmp_vec_mod[iter_surv_idx] = - vec_sens_shift[iter_surv_idx,2]
                print("New survey index:", unique_surv_idx[iter_surv_idx])
            else:
                tmp_vec_mod[iter_surv_idx] = model_shift_vec_tot[same_idx] - vec_sens_shift[iter_surv_idx,2] #the model shift has to be weighted, since the total data component due to the shift should be small and not the actual model shift cells themselves               
                        
        model_shift_vec_tot = tmp_vec_mod
        
    mod_shift_idx  = unique_surv_idx
    
    return model_shift_vec_tot, mod_shift_idx


#Die ist eine Tikonov gedämpfte iterative Inversion ohne Zerlegung in Blöcken 
#Dense matrix

def inversion_dense (dat_val, dat_err, dat_longlatZ, dat_igrf_NED, dat_surv_idx, dipoles, predicted_bck, model_shift_vec, unique_surv_idx_prev, first_idx_DGRF,  nr_idx_DGRF, lambda_d, lambda_s, lambda_s_decay, lambda_min, lambda_s_sptl, lambda_s_sptl_max, weight_shift, lambda_shift, data_weight_threshold, iterations,stp_criteria):
    
    mod = dipoles.start_mod # Startmodel 
    delta_d = np.zeros(len(dat_val))
    
    #G = equi.calculate_ano_design_matrix(dat.longlatZ,dipoles.longlatZ,dat.igrf_NED,dipoles.igrf_NED, first_idx_DGRF,  nr_idx_DGRF) #Jacobian matrix
    G = equi.calculate_ano_design_matrix(dat_longlatZ,dipoles.longlatZ, dipoles.cell_size,dat_igrf_NED,dipoles.igrf_NED, first_idx_DGRF,  nr_idx_DGRF) #Jacobian matrix
    
    #Determine an extra weight for data with large misfits
    weight_lval = np.ones(len(dat_err))
    
    if data_weight_threshold > 0.0:
        
        delta_dabs = np.abs(dat_val - predicted_bck)      
        idx_lval = np.where(delta_dabs > data_weight_threshold)
        
        for i in range(len(idx_lval)):
            weight_lval[idx_lval[i]] = data_weight_threshold/delta_dabs[idx_lval[i]] #this is the extra weight determined by the ratio of threshold and data misfit    
           

    
    #Add the error weigthing matrix
    inv_dat_err = weight_lval/dat_err
    D = sparse.diags(inv_dat_err,format="csr")
    
    #D*G error weighted Jacobian matrix
    DG = D.dot(G)    
    nrow, ncol = np.shape(DG) #number of rows and columns without shift cells (when available)
    print(nrow,ncol)
 
    #lambda_factor = np.abs(DG).sum(1).mean(0)     #determine average weight of a row    
    lambda_factor = np.abs(DG).sum(0).mean(0)     #determine average weight of a colmnn
    print(lambda_factor)
    
    
    if lambda_s > 0.0 : #Calculate the difference operator matrix
            
        ##################################################################
        #Determine the coverage values to control the spatial variation of the smoothing
        
        lambda_factor_spatial = np.ones(ncol) #vector defining the spatial weight of the smoothing         
        
        if(lambda_s_sptl > 0.0):
            
            EPS = 1e-30
            
            if lambda_factor != 0.0:
                norm_coverage = np.abs(DG).sum(0)/lambda_factor     #Determine the ratio of the overage at each cell and the mean coverage
            else:
                norm_coverage = np.abs(DG).sum(0)/EPS
            
            
            b =   np.where(norm_coverage < lambda_s_sptl)    #find the cells where the coverage is such low that its smoothness is increased 
            
            lambda_factor_spatial[b] = (lambda_s_sptl+ EPS)/(norm_coverage[b] + EPS) #take the ratio of actual value and threshold for weighting
            
            c =   np.where(lambda_factor_spatial > lambda_s_sptl_max) #border the smooting by a maximum value 
            lambda_factor_spatial[c] = lambda_s_sptl_max
            
            print(" ")
            print("threshold for coverage: ",  lambda_factor*lambda_s_sptl)
            print("Number of cells with increased smoothing: ", len(lambda_factor_spatial[b]), "of", ncol)
            print("Number of cells with maximum smoothing:", len(lambda_factor_spatial[c]), "of", ncol)
            print(" ")
            
        
        ##################################################################
        
        C_x, C_y = equi.build_1st_order_diffmatrix(dipoles.nx, dipoles.ny) 
        print("The smoothing matrix is calculated")   
        
        #(Rx,Cx,Vx) = sparse.find(C_y)
        #print(Rx,Cx,Vx)
        
        if(lambda_s_sptl > 0.0): #apply the reweighting of the smoothing matrix differences
            C_x = equi.rewweight_diffmatrix(C_x, lambda_factor_spatial)
            C_y = equi.rewweight_diffmatrix(C_y, lambda_factor_spatial)    
         
    else :
        C_x = sparse.csr_matrix(np.empty((0, 0)))
        C_y = sparse.csr_matrix(np.empty((0, 0)))
    

    nrow_C_x, ncol_C_x = np.shape(C_x)
    nrow_C_y, ncol_C_y = np.shape(C_y)

    #(Rx,Cx,Vx) = sparse.find(C_y)
    #print(Rx,Cx,Vx)

    nrow_C = nrow_C_x + nrow_C_y  #Number of rows of the regularization 
    
    ##############################################
    #make a vector including the sensitvities associated with the shift cells
    vec_sens_shift = [] #empty array as place holder

    
    if weight_shift > 0.0:  #determine constant shifts in the datasets that are associated with errors in the processing        
       
        #Mcreate, modify and/or extend the matrices (G, C_shift, C_x, C_y) and the model vector by the entries from the "shift" cells
        if lambda_s > 0.0: #with smoothing
            vec_sens_shift, mod, G, C_shift, model_shift_vec, C_x, C_y  = add_shifts_cells_in_matrices_vec(dat_surv_idx, DG, G, mod, C_x, C_y, weight_shift, model_shift_vec, unique_surv_idx_prev, lambda_s, 0)
        else:
            vec_sens_shift, mod, G, C_shift, model_shift_vec = add_shifts_cells_in_matrices_vec(dat_surv_idx, DG, G, mod, C_x, C_y, weight_shift, model_shift_vec, unique_surv_idx_prev, lambda_s, 0)     
        
        #Calculate the data weighted Jacobian matrix once again
        DG = D.dot(G)    
        
    ##############################################
    
    #Determine Gt*Dt*D*G + lambda_d*I + lambda_s*CtC     
    big_matrix = DG.T.dot(DG) + np.eye(DG.shape[1]) * lambda_d**2  * lambda_factor**2 #"np.eye" ist die Einheitsmatrix
    
    if lambda_s > 0.0:  #with smoothing
        
        C_x2 = C_x.T.dot(C_x)
        C_y2 = C_y.T.dot(C_y)            
        
        big_matrix = big_matrix + C_x2.multiply((lambda_s**2)*(lambda_factor**2)) + C_y2.multiply((lambda_s**2)*(lambda_factor**2))  
        del C_x2, C_y2
    
    if weight_shift > 0.0 and lambda_shift > 0.0: # if "shift" cells are used
        C_shift2 = C_shift.T.dot(C_shift)

        big_matrix = big_matrix + C_shift2.multiply((lambda_shift**2)*(lambda_factor**2))   
        del C_shift2
        
    
    predicted = G.dot(mod) #vorhergesgte Daten
    delta_d = dat_val - predicted - predicted_bck  #d - F(m) - F(m)_from_adjustes_dipoles 
    
    #Determine RMS and error normalized RMS
    RMS =  np.sqrt(np.mean(np.square(delta_d)))
    NRMS= np.sqrt(np.mean(np.square(delta_d/dat_err)))
    
    print("Iter:", 0, "; RMS", RMS, "; Normalized RMS", NRMS)  
    pre_RMS = RMS

    #Iteration
    for k in range(iterations):
        
        right_term = DG.T.dot(D.dot(delta_d)) # Dt*Gt*D(d_obs - G*m_i)   
     
        if lambda_s > 0.0: #with smoothing
            C_x2 = C_x.T.dot(C_x)
            C_y2 = C_y.T.dot(C_y)            
            mod_term = (C_x2.multiply((lambda_s**2)*(lambda_factor**2)) + C_y2.multiply((lambda_s**2)*(lambda_factor**2))) #lambda_s*CtC
            del C_x2, C_y2
                    
            right_term = right_term - mod_term.dot(mod)
        
        if weight_shift > 0.0 and lambda_shift > 0.0: # if "shift" cells are used
            
            C_shift2 = C_shift.T.dot(C_shift)
            mod_term = C_shift2.multiply((lambda_shift**2)*(lambda_factor**2))   
            
            mod_ref = np.append(np.zeros(ncol),model_shift_vec)
                        
            right_term = right_term - (mod_term.dot(mod) - C_shift2.dot(mod_ref*(lambda_shift**2)*(lambda_factor**2)))
            del C_shift2


        mod = mod + np.linalg.solve(big_matrix,right_term) # (Dt*Gt*G*D + lambda_d*I + lambda_s*CtC)delta_m = Dt*Gt*D(d_obs - G*m_i) - lambda_s*CtC*m_i
        
        
        #Determine RMS and error normalized RMS
        predicted = G.dot(mod) #vorhergesgte Daten
        delta_d = dat_val - predicted  - predicted_bck  #d - F(m) - F(m)_from_adjustes_dipoles
        
        RMS =  np.sqrt(np.mean(np.square(delta_d))) 
        NRMS= np.sqrt(np.mean(np.square(delta_d/dat_err)))
        
        print("Iter:", k+1, "; RMS", RMS, "; Normalized RMS", NRMS)
    
        if abs(pre_RMS - RMS) < stp_criteria: #inversion is converged
            print("Inversion criteria is reached")
            
            if lambda_s_decay != 1.0 and lambda_s > 0.0 and lambda_s > lambda_min: 
                #Modify the smoothing parameter at every iteration
                lambda_s = lambda_s * lambda_s_decay 
                print('Smoothing parameter is ', lambda_s)
            else:
                break
        
        pre_RMS = RMS
        

    new_model_shift_vec = np.empty(shape=(0))
    
    if weight_shift > 0.0:
        vec_sens_shift[:,2] = vec_sens_shift[:,1] * mod[ncol:] #Determine data change due to the "shift" cells        
        
        new_model_shift_vec = mod[ncol:] #these are the model shifts
        
        print(" ")
        print("Proposed data shifts for surveys in nT:", vec_sens_shift[:,2])
        print(" ")
        
        mod = mod[:ncol]  #this is required to remove the "shift" cells from the model vector
        
    
    return mod,predicted, RMS, vec_sens_shift, new_model_shift_vec


#Die ist eine Tikonov gedämpfte iterative Inversion ohne Zerlegung in Blöcken 
#Sparse matrix mit LSQR Solver

def inversion_sparse(dat_val, dat_err, dat_longlatZ, dat_igrf_NED,  dat_surv_idx, dipoles, predicted_bck, 
                     model_shift_vec, unique_surv_idx_prev, first_idx_DGRF,  nr_idx_DGRF, lambda_d, lambda_s, lambda_s_decay, lambda_min, 
                     lambda_s_sptl, lambda_s_sptl_max, weight_shift, lambda_shift, data_weight_threshold, iterations,
                     iterations_exakt,stp_criteria, dist_ignored_inv, dist_ignored_forward, MAX_MAT, gc,FLAG_CALC_Md =  1):
    """
    FLAG_CALC_Md : int
        Flag that defines if the m = M*d is calculated row by row (== 0), or by using sparse (sub-)matrices (==1)
    lambda_d : float
        Smoothing weight
    lambda_s : float
        Shifting weight
    """

    mod = dipoles.start_mod # Startmodel 

    G = equi.calculate_ano_design_matrix_sparse(dat_longlatZ,dipoles.longlatZ,dipoles.cell_size,dat_igrf_NED,dipoles.igrf_NED, dist_ignored_inv, MAX_MAT, first_idx_DGRF,  nr_idx_DGRF) #Jacobian matrix (sparse)

    #use iterative LSQR solver for solving linear system
    predicted = G.dot(mod)
    delta_d = dat_val - predicted  - predicted_bck  #d - F(m) - F(m)_from_adjustes_dipoles
    
    #Determine an extra weight for data with large misfits
    weight_lval = np.ones(len(dat_err))
    
    if data_weight_threshold > 0.0:
        delta_dabs = np.abs(delta_d)      
        idx_lval = np.where(delta_dabs > data_weight_threshold)
        weight_lval[idx_lval] = data_weight_threshold/delta_dabs[idx_lval] #this is the extra weight determined by the ratio of threshold and data misfit
           
        
    #Add the error weigthing matrix
    inv_dat_err = weight_lval/dat_err
    D = sparse.diags(inv_dat_err,format="csr")
    
    #D*G error weighted Jacobian matrix
    DG = D.dot(G)    
    nrow_org, ncol_org = DG.shape
    
    #Determine RMS and error normalized RMS
    RMS =  np.sqrt(np.mean(np.square(delta_d)))
    NRMS= np.sqrt(np.mean(np.square(delta_d/dat_err)))

    print("Iter:", 0, "; RMS", RMS, "; Normalized RMS", NRMS)  
    pre_RMS = RMS
    
    lambda_factor = np.abs(DG).sum(0).mean(1)     #determine average weight of a colmnn

    print(lambda_factor)
    
    lambda_d_factor = lambda_factor * lambda_d
      
    #######################################
    #extend size of the matrix by smoothing contribution
    lambda_s_factor = lambda_factor * lambda_s
    
    #Add the smoothing part to the data vector and Jacobian matrix
    if lambda_s > 0.0: 
        

        ##################################################################
        #Determine the coverage values to control the spatial variation of the smoothing
        lambda_factor_spatial = np.ones(ncol_org) #vector defining the spatial weight of the smoothing         
        norm_coverage = np.zeros(ncol_org)
        
        if(lambda_s_sptl > 0.0):
            
            EPS = 1e-30
            
            if lambda_factor != 0.0:
                tmp_norm_coverage = np.abs(DG).sum(0)/lambda_factor     #Determine the ratio of the overage at each cell and the mean coverage
            else:
                tmp_norm_coverage = np.abs(DG).sum(0)/EPS
            
            for iter in range(ncol_org):
                norm_coverage[iter] = tmp_norm_coverage[:,iter]
                        
            b =   np.where(norm_coverage < lambda_s_sptl)    #find the cells where the coverage is such low that its smoothness is increased 
            
            lambda_factor_spatial[b] = (lambda_s_sptl+ EPS)/(norm_coverage[b] + EPS) #take the ratio of actual value and threshold for weighting
            
            c =   np.where(lambda_factor_spatial > lambda_s_sptl_max) #border the smooting by a maximum value 
            lambda_factor_spatial[c] = lambda_s_sptl_max
            
            print(" ")
            print("threshold for coverage: ",  lambda_factor*lambda_s_sptl)
            print("Number of cells with increased smoothing: ", len(lambda_factor_spatial[b]), "of", ncol_org)
            print("Number of cells with maximum smoothing:", len(lambda_factor_spatial[c]), "of", ncol_org)
            print(" ")
            
        ##################################################################

        #Calculate the difference operator matrix
        C_x, C_y = equi.build_1st_order_diffmatrix(dipoles.nx, dipoles.ny) 
        print("The smoothing matrix is calculated")
        
        if(lambda_s_sptl > 0.0): #apply the reweighting of the smoothing matrix differences
            C_x = equi.rewweight_diffmatrix(C_x, lambda_factor_spatial)
            C_y = equi.rewweight_diffmatrix(C_y, lambda_factor_spatial)
            
    else:
        C_x = sparse.csr_matrix(np.empty((0, ncol_org)))
        C_y = sparse.csr_matrix(np.empty((0, ncol_org)))
           
    #######################################
    lambda_shift_factor = lambda_factor * lambda_shift
    
    #make a vector including the sensitvities associated with the shift cells
    vec_sens_shift = [] #empty array as place holder

    if weight_shift > 0.0:  #determine constant shifts in the datasets that are associated with errors in the processing        
    
        #Mcreate, modify and/or extend the matrices (G, C_shift, C_x, C_y) and the model vector by the entries from the "shift" cells
        if lambda_s > 0.0: #with smoothing
            vec_sens_shift, mod, G, C_shift, model_shift_vec, C_x, C_y  = add_shifts_cells_in_matrices_vec(dat_surv_idx, DG, G, mod, C_x, C_y, weight_shift, model_shift_vec, unique_surv_idx_prev, lambda_s, 1)
        else:
            vec_sens_shift, mod, G, C_shift, model_shift_vec = add_shifts_cells_in_matrices_vec(dat_surv_idx, DG, G, mod, C_x, C_y, weight_shift, unique_surv_idx_prev, model_shift_vec, lambda_s, 1)     
        
        #Calculate the data weighted Jacobian matrix once again
        DG = D.dot(G)    
    else:
        C_shift = sparse.csr_matrix(np.empty((0, ncol_org)))
    ##############################################
    

    if lambda_d > 0.0:
        nrow2, ncol2 = DG.shape
        ID = sparse.diags(np.ones(ncol2))
        #Note that DG includes from now on also the regularization terms !!!
        DG = sparse.vstack([DG, ID.multiply(lambda_d_factor), C_x.multiply(lambda_s_factor), C_y.multiply(lambda_s_factor), C_shift.multiply(lambda_shift_factor)]) 
    else:
        #Note that DG includes from now on also the regularization terms !!!
        DG = sparse.vstack([DG, C_x.multiply(lambda_s_factor), C_y.multiply(lambda_s_factor), C_shift.multiply(lambda_shift_factor)])
    
    nrow_all, ncol_all = DG.shape
        
    #extend lenght of vector    
    delta_all = np.zeros(nrow_all)  
    delta_all[0:nrow_org] = D.dot(delta_d)    
    
    
    #add the shifting entries to the model vector
    if weight_shift > 0.0:
        mod_shift_vec_strt = delta_all.size - model_shift_vec.size 
        #delta_all[mod_shift_vec_strt:] = lambda_shift_factor*model_shift_vec        
        mod_ref = np.append(np.zeros(ncol_org),model_shift_vec)        
        delta_all[mod_shift_vec_strt:] = (C_shift.multiply(lambda_shift_factor)).dot(mod_ref)   
        
        #TMP
        #print("check ref vector:", delta_all[mod_shift_vec_strt:]/lambda_factor)


    idx_on = 0
    #Iteration
    for k in range(iterations + iterations_exakt):
        
        gc.collect() #Garbage collection
        
        #Data vector for the i-th model (that will be subtracted)
        dat_u_i = DG.dot(mod)
        
        if lambda_d > 0.0:
            dat_u_i[nrow_org + ncol_all:] = 0.0
        else:
            dat_u_i[nrow_org:] = 0.0      
            
        #delta_mod, istop, itn, r1norm = sparse_linalg.lsqr(DG, delta_all, damp=lambda_d_factor)[:4] #solving damped linear system with lsqr
        mod, istop, itn, r1norm = sparse_linalg.lsqr(DG, delta_all + dat_u_i)[:4] #solving damped linear system with lsqr
        
        #mod =  mod + delta_mod                       #m_i+1 = m_i + delta_m
    
        if k < (iterations) and idx_on == 0: #abgeschätzte Berechnung von Fehlern in Vorwärtsrechnung (sparse matrix) 
            predicted = G.dot(mod)
            predicted = predicted[0:nrow_org] #in case of smoothing, the resulting vector is larger than the actual data vector
            delta_d = dat_val - predicted  - predicted_bck  #d - F(m) - F(m)_from_adjustes_dipoles
        else:  #exakte Berechnung von Fehlern in Vorwärtsrechnung (dense matrix) 
            
            
            if idx_on == 0:
                print("Forward calculation is exact")
                idx_on = 1
                
            if FLAG_CALC_Md == 0:
                if __name__ == '__main__': #necessary is safety if the code is parallelized with multiprocessing
                    predicted = equi.calculate_design_matrix_multi_model_vec(dat_longlatZ,dipoles.longlatZ,dipoles.cell_size,dat_igrf_NED,dipoles.igrf_NED, mod[:ncol_org], dist_ignored_forward, first_idx_DGRF, nr_idx_DGRF)
            else:
                predicted = equi.calculate_sparse_design_matrix_multi_model_vec(dat_longlatZ,dipoles.longlatZ,dipoles.cell_size,dat_igrf_NED,dipoles.igrf_NED, mod[:ncol_org], dist_ignored_forward, MAX_MAT, first_idx_DGRF, nr_idx_DGRF)
                
            
            #add the contribution from the "shift" cells
            if weight_shift > 0.0:
                predicted = predicted + G[:,ncol_org:].dot(mod[ncol_org:])
            
            delta_d = dat_val - predicted - predicted_bck  #d - F(m) - F(m)_from_adjusted_dipoles
            
        #Determine RMS and error normalized RMS
        RMS =  np.sqrt(np.mean(np.square(delta_d)))
        NRMS= np.sqrt(np.mean(np.square(delta_d/dat_err)))

        print("Iter:",  k+1, "; RMS", RMS, "; Normalized RMS", NRMS)
    
        if abs(pre_RMS - RMS) < stp_criteria: #inversion is converged
            print("Inversion criteria is reached")
            
            if lambda_s_decay != 1.0 and lambda_s > 0.0 and lambda_s > lambda_min: 
                
                #Modify the smoothing parameter at every iteration
                lambda_s_old = lambda_s
                lambda_s = lambda_s * lambda_s_decay 
                lambda_s_factor = lambda_s_factor * (lambda_s/lambda_s_old)
                
                print('Smoothing parameter is ', lambda_s)
                
                #Modify the weight in smoothing part in the Jacobian matrix
                DG.resize(nrow_org, ncol_all)

                if lambda_d > 0.0:
                    DG = sparse.vstack([DG, ID.multiply(lambda_d_factor), C_x.multiply(lambda_s_factor), C_y.multiply(lambda_s_factor), C_shift.multiply(lambda_shift_factor)])
                else:
                    DG = sparse.vstack([DG, C_x.multiply(lambda_s_factor), C_y.multiply(lambda_s_factor), C_shift.multiply(lambda_shift_factor)])
            
            elif idx_on == 0 and iterations_exakt > 0:
                print("Forward calculation is exact")
                idx_on = 1
            else:
                break
        
        pre_RMS = RMS
                
        
        delta_all[0:nrow_org] = D.dot(delta_d)

    
    del G, DG
    if lambda_s > 0.0:
        del C_x, C_y
    
    
    new_model_shift_vec = np.empty(shape=(0))

    if weight_shift > 0.0:
        vec_sens_shift[:,2] = vec_sens_shift[:,1] * mod[ncol_org:] #Determine data change due to the "shift" cells        
        
        new_model_shift_vec = mod[ncol_org:] #these are the model shifts
        
        print(" ")
        print("Proposed data shifts for surveys in nT:", vec_sens_shift[:,2])
        print(" ")
        
        mod = mod[:ncol_org]  #this is required to remove the "shift" cells from the model vector

    
    return mod, predicted, RMS, vec_sens_shift, new_model_shift_vec