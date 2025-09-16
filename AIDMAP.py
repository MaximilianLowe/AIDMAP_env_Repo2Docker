# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:33:57 2024

@authors: Alan Aitken; Mareen Loesing, Lu Li, Joerg Ebbing

This version created : 11/12/2024

This provides a semi-automated magnetic data processing workflow for the Antarctic Interactive Digital Magnetic Anomaly Map 
"""
import numpy as np
import pandas as pd
from pyproj import Proj
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter,sosfiltfilt
import scipy.interpolate as interpolate
from scipy import spatial
import igrf_utils as iut
import pywt
from dateutil.parser import isoparse #may replace with pandas function
from hapiclient import hapi

class DataPrep:
    def __init__(self,data, proj = None, data_limits = [(None,None),(None,None)], nanvalue = -99999.9):
        required_atts = ['LineName','Lon','Lat','X','Y','Height_WGS84','Date','Time','RawTMI']
        for i in required_atts:
            if i not in data.keys():
                raise AttributeError('DataFrame object has no attribute {}'.format(i))
        try:
            #if suitable limits provided cut the data
            self.data = data[(data['X']<data_limits[1][0]) & (data['X']>data_limits[0][0]) & (data['Y']<data_limits[1][1]) & (data['Y']>data_limits[0][1])].reset_index(drop=True)
        except TypeError:
            #otherwise use the whole dataset
            self.data = data
        try:
            self.proj = Proj(proj)
        except TypeError:
            #if not provided assume polar stereographic
            self.proj = Proj("+proj=stere +lat_0=-90 +lat_ts=-71")
        DateTimeStrings = self.data.Date + ' ' + self.data.Time
        DateTimes = [None]*len(DateTimeStrings)
        for i,j in enumerate(DateTimeStrings):
            DateTimes[i] = isoparse(j)    
        self.data['DateTime'] = DateTimes
        self.line_names = np.unique(self.data.LineName)
        self.data['RawTMI'] = np.where(self.data['RawTMI'] == nanvalue, np.nan, self.data['RawTMI'])

    def IGRFcorrection(self):
        IGRFvalues = IGRFcorrection(self.data.Lat,self.data.Lon,self.data.Height_WGS84,self.data.DateTime)
        self.data["IGRF"] = IGRFvalues
        IGRFcorrected = self.data.RawTMI - IGRFvalues
        self.data['TMI_IGRFcorr'] =  IGRFcorrected
        
    def split_lines_by_time(self, maxtime=10):
        result = []
        xx = []
        yy = []
        time=np.diff(self.data.DateTime)
        ind_jump=np.argwhere(time>maxtime)[:,0]
        ind_jump=np.hstack([0,ind_jump+1,len(self.data.DateTime)])
        for k in range(len(ind_jump)-1):
            result.append(self.data[ind_jump[k]:ind_jump[k+1]])
            xx.append(self.data.x[ind_jump[k]:ind_jump[k+1]])
            yy.append(self.data.y[ind_jump[k]:ind_jump[k+1]])
        return result,xx,yy
            
    def split_lines_by_dist(self, maxdist=2000):
        result = []
        xx = []
        yy = []
        dist=np.sqrt(np.diff(self.data.X)**2+np.diff(self.data.Y)**2)
        ind_jump=np.argwhere(dist>maxdist)[:,0]
        ind_jump=np.hstack([0,ind_jump+1,len(self.data.x)])
        for k in range(len(ind_jump)-1):
            result.append(self.data[ind_jump[k]:ind_jump[k+1]])
            xx.append(self.data.x[ind_jump[k]:ind_jump[k+1]])
            yy.append(self.data.y[ind_jump[k]:ind_jump[k+1]])
        return result,xx,yy
    
    def split_lines_by_name(self):
        #a dict keyed by line name
        self.data_lines = {}
        for line in self.line_names:
            indices = np.where(self.data.LineName == line)[0]
            self.data_lines[line] = self.data.loc[indices]
            ldata = self.data_lines[line]
            #make a distance column for the along-line distance
            distance=np.cumsum(np.sqrt(np.diff(ldata.X)**2+np.diff(ldata.Y)**2))
            distance = np.insert(distance,0,0.0)
            n_cols = len(self.data.keys())
            self.data_lines[line].insert(n_cols,"Distance",distance)
    
    def filter_lines(self,filter_type = 'LP', sampling = 'time', BWorder = None):   
        def sampling(line):
            if sampling == 'dist':
                dd = line.Distance
                sampling_rate = 1/np.mean(np.diff(dd))
            elif sampling == 'time':
                tt = line.DateTime
                sampling_rate = 1/np.mean(np.diff(tt))
            else:
                sampling_rate = 1
            nyquist_freq = sampling_rate/2
            cutoff_freq = 0.6 * nyquist_freq
            return sampling_rate, cutoff_freq
        
        def apply_filter(line,sampling_rate,cutoff_freq):
            Mdata = np.pad(line.TMI_IGRFcorr,5,mode='symmetric')            
            if filter_type == 'BW':
                filtered = fourier_BW_filter(Mdata, cutoff_freq, BWorder)
            elif filter_type == 'LP':
                filtered = fourier_LP_filter(Mdata, sampling_rate, cutoff_freq)
            return filtered[5:-5]
        
        for i,j in enumerate(self.data_lines):
            data = self.data_lines[j]
            required_atts = ['TMI_IGRFcorr']
            for i in required_atts:
                if i not in data.keys():
                    raise AttributeError('DataFrame object has no attribute {}'.format(i))
            sampling_rate, cutoff = sampling(data)
            filtered_data = apply_filter(data, sampling_rate, cutoff)    
            self.data_lines[j]["TMI_Filt"]=filtered_data
        
    def Nth_diff(self, order = 4, pad = 'yes'):
        if order % 2 != 0:
            raise(ValueError('for array size consistency, order must be equal number'))
        for i,j in enumerate(self.data_lines):
            data = self.data_lines[j]
            required_atts = ['TMI_IGRFcorr']
            for i in required_atts:
                if i not in data.keys():
                    raise AttributeError('DataFrame object has no attribute {}'.format(i))
            if "TMI_Filt" in data.keys():
                Mdata = np.array(data.TMI_Filt)
            else:
                Mdata = np.array(data.TMI_IGRFcorr)
            Dist = np.array(data.Distance)    
            if pad == 'yes': #maybe in the end we just assume padding is wanted?
                    Mdata = np.pad(Mdata, (int(order),), mode='symmetric')
                    Dist = np.pad(Dist, (int(order),), mode='symmetric')        
            nth_diff=np.diff(Mdata, n = order)
            if pad == 'yes':
                nth_diff = nth_diff[order//2:-order//2]
            else:
                #pad missing ends with nans
                n_els = len(Mdata) - len(nth_diff) 
                arr = np.full(n_els//2,np.nan)
                if n_els % 2 == 0:
                    nth_diff = np.insert(nth_diff,0,arr)
                    nth_diff = np.append(nth_diff,arr)
                else:
                    nth_diff = np.insert(nth_diff,0,np.nan)
                    nth_diff = np.insert(nth_diff,0,arr)
                    nth_diff = np.append(nth_diff,arr)
            self.data_lines[j]["Nth_Diff"] = nth_diff
            
    def Despike(self):
        #here to avoid detective work we just treat every data point as a 'spike'
        # this will also partly deal with generally noisy data
        for i,j in enumerate(self.data_lines):
            data = self.data_lines[j]
            required_atts = ['TMI_IGRFcorr', "Nth_Diff"]
            for i in required_atts:
                if i not in data.keys():
                    raise AttributeError('DataFrame object has no attribute {}'.format(i))
            if "TMI_Filt" in data.keys():
                Mdata = np.array(data.TMI_Filt)
            else:
                Mdata = np.array(data.TMI_IGRFcorr)        
            despike_val = data.Nth_Diff/6.0
            despiked = Mdata-despike_val
            self.data_lines[j]["TMI_Filt"] = despiked
    
    def FlagSteps(self, tol = 0.0):
        #For steps we identify asymmetry and flag it 
        for i,j in enumerate(self.data_lines):
            data = self.data_lines[j]
            required_atts = ["Nth_Diff"]
            for i in required_atts:
                if i not in data.keys():
                    raise AttributeError('DataFrame object has no attribute {}'.format(i))
            StepFlag = np.zeros_like(data.Nth_Diff)
            def Step_picker(nth_diff,i,tol):
                sub_array = nth_diff[i-2:i+2]
                step_array = [1.,-3.,3.,-1.]
                try: 
                    defect = sub_array/step_array
                except ValueError:
                    defect = [0,0,0,0]
                if np.nanmax(defect)-np.nanmin(defect) > tol:
                   Flag = np.mean(defect) 
                else:
                   Flag = 0.0
                return Flag
            StepFlag = [Step_picker(data.Nth_Diff,i,tol) for i,j in enumerate(data.Nth_Diff)]
            self.data_lines[j]["StepFlag"] = StepFlag
            
    def MergeLines(self):
        for i,j in enumerate(self.data_lines):
            data = self.data_lines[j]
            if i == 0:
                self.data = data
            else:
                self.data = pd.concat([self.data,data])
    
    def RunDataPrep(self, filt_type = 'BW', sampling = 'time', n_diff = 4, pad = 'yes', tol = 0.0, BW_order = 1):
        self.IGRFcorrection()
        self.split_lines_by_name()
        self.filter_lines(filter_type = filt_type, sampling = sampling, BWorder = BW_order) #optional step
        self.Nth_Diff(order = n_diff, pad = pad)
        self.Despike()
        self.FlagSteps(tol = tol)
        self.MergeLines
        return(self.data)
    
class WaveletFiltering:
    def __init__(self,data, data_prop = 'RawTMI', proj = None, data_limits = [(None,None),(None,None)],nanvalue = -99999.9):
        required_atts = ['LineName','X','Y','Date','Time','RawTMI']
        for i in required_atts:
            if i not in data.keys():
                raise AttributeError('DataFrame object has no attribute {}'.format(i))
        try:
            #if suitable limits provided cut the data
            self.data = data[(data['X']<data_limits[1][0]) & (data['X']>data_limits[0][0]) & (data['Y']<data_limits[1][1]) & (data['Y']>data_limits[0][1])].reset_index(drop=True)
        except TypeError:
            #otherwise use the whole dataset
            self.data = data
        try:
            self.proj = Proj(proj)
        except TypeError:
            #if not provided assume polar stereographic
            self.proj = Proj("+proj=stere +lat_0=-90 +lat_ts=-71")
        self.line_names = np.unique(self.data.LineName)
        self.data[data_prop] = np.where(self.data[data_prop] == nanvalue, np.nan, self.data[data_prop])
        self.data_prop= data_prop
    
    def split_lines_by_name(self):
        #a dict keyed by line name
        self.data_lines = {}
        for line in self.line_names:
            indices = np.where(self.data.LineName == line)[0]
            self.data_lines[line] = self.data.loc[indices]
            ldata = self.data_lines[line]
            #make a distance column for the along-line distance
            if 'Distance' not in self.data.keys():
                distance=np.cumsum(np.sqrt(np.diff(ldata.X)**2+np.diff(ldata.Y)**2))
                distance = np.insert(distance,0,0.0)
                n_cols = len(self.data.keys())
                self.data_lines[line].insert(n_cols,"Distance",distance)
            
    def resample(self):
        one_hour_ns = np.timedelta64(3600, 's')
        for i, j in enumerate(self.data_lines): 
            data = self.data_lines[j]
            #@Mareen - I added this line to also remove entries where data is nan... 
            data = data[~np.isnan(data[self.data_prop])] 
            
            if 'DateTime' not in data.keys():
                data["DateTime"] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'])
            
            if data["DateTime"].duplicated().any():
                data = data[~data["DateTime"].duplicated(keep='first')]
    
            differences = np.diff(data["DateTime"])
            differences = differences[~np.isnan(differences)]
            
            if differences.size == 0:
                line_interpolated = data
            else:
                if (max(differences) >= one_hour_ns):
                    line_interpolated = data
                else:
                    differences_in_seconds = differences.astype('timedelta64[s]').astype(float)
                    freq = max(np.min(differences_in_seconds), 1)
                    freq_seconds = f"{np.round(freq, 2)}s"
                    time_regular = pd.date_range(start=data["DateTime"].iloc[0], end=data["DateTime"].iloc[-1], freq=freq_seconds)

                    L = data.set_index("DateTime")
                    line_regular = L.reindex(time_regular, method='ffill', tolerance=pd.to_timedelta(freq_seconds))
                    line_interpolated = line_regular.apply(pd.to_numeric, errors='coerce').interpolate(method='linear')
            self.data_lines[j] = line_interpolated
        
    
    def WaveletFilterDWT(self, wavelet_name='db4'):
        wavelet = pywt.Wavelet(wavelet_name)
   
        for i,j in enumerate(self.data_lines):
            data = self.data_lines[j]
            if data[self.data_prop].empty:
                print(f"Warning: attribute {self.data_prop} for line {j} is empty.")
                continue     
            signal_length = len(data)
            max_level = pywt.dwt_max_level(signal_length, wavelet.dec_len)
            padded_signal = np.pad(data[self.data_prop], pad_width=10, mode='constant')
            coeffs = pywt.wavedec(padded_signal, wavelet, level=max_level)
            reconstructed_signal = pywt.waverec(coeffs, wavelet)
            reconstructed_signal_unpadded = reconstructed_signal[10: 10 + signal_length]
            self.data_lines[j]["TMI_DWT_reconstructed"] = reconstructed_signal_unpadded
            
    def WaveletFilterCWT(self, wavelet_name='cmor'):  
        for i, j in enumerate(self.data_lines):
            data = self.data_lines[j]
            if data[self.data_prop].empty:
                print(f"Warning: attribute {self.data_prop} for line {j} is empty.")
                continue            
            signal_length = len(data[self.data_prop])
            padded_signal = np.pad(data[self.data_prop], pad_width=10, mode='reflect')
            
            if signal_length == 0: continue
            else:
                max_scale = signal_length
                scales = np.arange(1, min(128, max_scale))
                coeffs, freqs = pywt.cwt(padded_signal, scales, wavelet_name)
                coeffs_real = np.real(coeffs)

                reconstructed_signal = np.sum(coeffs_real / scales[:, None], axis=0)
                reconstructed_signal_unpadded = reconstructed_signal[10: 10 + signal_length]
                reconstructed_signal_scaled = reconstructed_signal_unpadded / (0.01 * np.max(data[self.data_prop]))
                self.data_lines[j]["TMI_CWT_reconstructed"] = reconstructed_signal_scaled

    def align_with_data(self, target_data):
        target_data_copy = target_data.data.copy()
        tree = spatial.cKDTree(np.array((self.data['X'], self.data['Y'])).T)
        distances, indices = tree.query(np.array((target_data_copy.X, target_data_copy.Y)).T)
        for column in self.data.columns:
            if column not in target_data_copy.columns:
                recon_values = self.data[column].iloc[indices]
                target_data_copy[column] = recon_values.values
        self.data = target_data_copy
    
    def MergeLines(self):
            for i,j in enumerate(self.data_lines):
                data = self.data_lines[j]
                if i == 0:
                    self.data = data
                else:
                    self.data = pd.concat([self.data,data])
                    
    def RunWaveletFilter(self, filt_type = 'DWT', wavelet_names = ['db4','cmor'], target_data = None):
        if wavelet_names == str:
            wavelet_names=list(wavelet_names)
        self.split_lines_by_name()
        self.resample()
        if filt_type == 'DWT' or 'both':
            wavelet_name = wavelet_names[0] 
            self.WaveletFilterDWT(wavelet_name=wavelet_name)
        if filt_type == 'CWT' or 'both':
            wavelet_name = wavelet_names[-1]
            self.WaveletFilterCWT(wavelet_name=wavelet_name)    
        self.MergeLines()
        if target_data is not None:
            self.align_with_data(target_data=target_data)
        return(self.data)

class MultibaseCorrection:
    def __init__(self, data, data_prop = 'RawTMI', proj = None, data_limits = [(None,None),(None,None)], has_local=False, local_base = (None,None,None) , local_data_path = None, nanvalue = -99999.9):
        required_atts = ['LineName','Lon','Lat','X','Y','DateTime','RawTMI']
        for i in required_atts:
            if i not in data.keys():
                raise AttributeError('DataFrame object has no attribute {}'.format(i))
        try:
            #if suitable limits provided cut the data
            self.data = data[(data['X']<data_limits[1][0]) & (data['X']>data_limits[0][0]) & (data['Y']<data_limits[1][1]) & (data['Y']>data_limits[0][1])].reset_index(drop=True)
        except TypeError:
            #otherwise use the whole dataset
            self.data = data
        try:
            self.proj = Proj(proj)
        except TypeError:
            #if not provided assume polar stereographic
            self.proj = Proj("+proj=stere +lat_0=-90 +lat_ts=-71")
        self.line_names = np.unique(self.data.LineName)
        self.has_local = has_local
        self.local_base = local_base
        self.local_data_path = local_data_path
        self.data[data_prop] = np.where(self.data[data_prop] == nanvalue, np.nan, self.data[data_prop])
        self.data_prop = data_prop
        
    def GetBasesandLocs(self, n):
        #function to identify the n-closest intermagnet bases to the data, and return their locations
        BaseLocs = {}
        with open('IAGA_Observatory_locs.txt','r') as f:
            next(f)
            for line in f:   
                line_data = line.split("\t")
                BaseLocs[line_data[0]]=(float(line_data[4]),float(line_data[3]),float(line_data[5]))
                
        #BL,TR,median_centroid....we do not deal yet with 180/-180 vs 360/0
        BL = (np.nanmin(self.data.Lon),np.nanmin(self.data.Lat))
        TL = (np.nanmin(self.data.Lon),np.nanmax(self.data.Lat))
        TR = (np.nanmax(self.data.Lon),np.nanmax(self.data.Lat)) 
        BR = (np.nanmax(self.data.Lon),np.nanmin(self.data.Lat))
        Centroid = (np.nanmedian(self.data.Lon),np.nanmedian(self.data.Lat))
        
        #For the data domain get the station distances, and get the closest n...adding bases as needed to cover all the data domain
        Dists = pd.Series({key: GreatCircleDist(BaseLocs[key],Centroid) for key in BaseLocs.keys()})
        Bases = Dists.nsmallest(n)
        BaseList = list(Bases.keys())
        Dists = pd.Series({key: GreatCircleDist(BaseLocs[key],BL) for key in BaseLocs.keys()})
        Bases = Dists.nsmallest(n)
        BaseList += [key for key in Bases.keys() if key not in BaseList]
        Dists = pd.Series({key: GreatCircleDist(BaseLocs[key],TL) for key in BaseLocs.keys()})
        Bases = Dists.nsmallest(n)
        BaseList += [key for key in Bases.keys() if key not in BaseList]
        Dists = pd.Series({key: GreatCircleDist(BaseLocs[key],TR) for key in BaseLocs.keys()})
        Bases = Dists.nsmallest(n)
        BaseList += [key for key in Bases.keys() if key not in BaseList]
        Dists = pd.Series({key: GreatCircleDist(BaseLocs[key],BR) for key in BaseLocs.keys()})
        Bases = Dists.nsmallest(n)
        BaseList += [key for key in Bases.keys() if key not in BaseList]
        BaseLocs = {key: BaseLocs[key] for key in BaseList}
        if self.has_local:
            BaseLocs['local'] = self.local_base
        self.BaseLocs = BaseLocs
        
    def GetBaseWeights(self, exponent = 2, scaleby = 'median'):
        B_Dists = pd.DataFrame(columns = self.BaseLocs.keys())
        for key in self.BaseLocs.keys():
            B_Dists[key] = GreatCircleDist(self.BaseLocs[key],(self.data.Lon,self.data.Lat))
        B_Dists['LengthScale'] = B_Dists.agg(scaleby, axis = 1)
        B_Weights = pd.DataFrame(columns = self.BaseLocs.keys())
        for key in self.BaseLocs.keys():
            B_Weights[key] = np.where(B_Dists[key] <= B_Dists['LengthScale'], 1-(B_Dists[key]/B_Dists['LengthScale'])**exponent, 0.0)
        B_WeightsScaling = B_Weights.sum(axis = 1)
        for key in self.BaseLocs.keys():
            B_Weights[key] /= B_WeightsScaling
        self.B_Weights = B_Weights
    
    def GetBaseData(self, maxtime = 120., padtime = 60., filter_period = 30., tunit = 'm'):
        maxtime_ = pd.Timedelta(maxtime, unit = tunit)
        padtime_ = pd.Timedelta(padtime,unit = tunit)
        filttime_ = pd.Timedelta(filter_period,unit = tunit)
        #sort the data by time
        data = self.data.sort_values(by = 'DateTime')
        #We split on gaps greater than maxtime.
        time_gaps=data.DateTime.diff()
        # Beginnings of time chunks
        gaps = time_gaps.where(time_gaps > maxtime_)
        Bchunks = data.DateTime[gaps.notna()]
        B_ilocs = [0] + [data.index.get_loc(key) for key in Bchunks.keys()]
        E_ilocs = [data.index.get_loc(key)-1 for key in Bchunks.keys()] +[-1]
        Begins = data.DateTime.iloc[B_ilocs] - padtime_
        Ends = data.DateTime.iloc[E_ilocs] + padtime_
        self.data_timeslices = pd.DataFrame({'begin': np.array(Begins), 'end': np.array(Ends)})
        #empty dataframe
        BaseDataDF = pd.DataFrame()
        RunningMeans = {}
        RunningNs = {}
        #For each intermagnet base, access the 1 minute data for each segment, filter the segments, then concatenate
        for key in self.BaseLocs.keys():
            BaseData = []    
            if key != 'local':
                for index, row in self.data_timeslices.iterrows():
                    BD = GetInterMagnetData(key,row['begin'],row['end']) #base, begin, end
                    if BD is not None:
                        BDtimes = np.array([pd.to_datetime(BD[0].decode('UTF-8')) for BD in BD])
                        BDvalues = np.array([BD[1] for BD in BD])
                        if np.all(BDvalues == 99999.) or np.all(np.isnan(BDvalues)):
                            print('station {} segment {} is all nans, skipping'.format(key,index))
                            BaseData.append([])
                            continue
                        elif np.any(BDvalues == 99999.):
                            num = np.count_nonzero(BDvalues == 99999.)
                            print('station {} segment {} contains {} nans, replacing with interpolated values'.format(key,index,num))
                            BDvalues = np.where(BDvalues ==99999.,np.nan,BDvalues) #np.nan values are replaced with interpolated values during filtering
                        dur = (BDtimes[-1]-BDtimes[0])
                        IGRFvalues = IGRFcorrection(self.BaseLocs[key][1],self.BaseLocs[key][0],self.BaseLocs[key][2],BDtimes)
                        BDfiltvalues = BDvalues - IGRFvalues
                        if dur > padtime_:
                            sf = dur/(len(BDtimes)-1) #should be 60 seconds normally
                            BDfiltvalues = fourier_BW_filter(BDfiltvalues, sf/filttime_, 10)
                        BD = [BDtimes,BDvalues,IGRFvalues,BDfiltvalues]
                        try:
                            RunningMeans[key] = (RunningMeans[key]*RunningNs[key] + np.nansum(BDfiltvalues))/(RunningNs[key]+len(BDfiltvalues))
                            RunningNs[key] = RunningNs[key]+len(BDfiltvalues)
                        except KeyError:
                            RunningMeans[key] = np.sum(BDfiltvalues)/len(BDfiltvalues)
                            RunningNs[key] = len(BDfiltvalues)
                        BaseData.append(BD)
                    else:
                        BaseData.append([])
                BaseDataDF[key] = BaseData
            elif key == 'local':
                BaseData = []
                for index, row in self.data_timeslices.iterrows():
                    BD = pd.read_csv(self.local_data_path)    
                    BD = BD[row['begin']:row['end']] #take the relevant timeslice - not working for now
                    if len(BD) > 0:
                        BDtimes = np.array([pd.to_datetime(BD[0].decode('UTF-8')) for BD in BD])
                        BDvalues = np.array([BD[1] for BD in BD])
                        dur = (BDtimes[-1]-BDtimes[0])
                        IGRFvalues = IGRFcorrection(self.local_base[1],self.local_base[0],self.local_base[2],BDtimes)
                        BDfiltvalues = BDvalues - IGRFvalues
                        if dur > padtime_:
                            sf = dur/(len(BDtimes)-1)
                            BDfiltvalues = fourier_BW_filter(BDfiltvalues, sf/filttime_, 10)
                        BD = [BDtimes,BDvalues,IGRFvalues,BDfiltvalues]
                        try:
                            RunningMeans[key] = (RunningMeans[key]*RunningNs[key] + np.sum(BDfiltvalues))/(RunningNs[key]+np.sum(~np.isnan(BDfiltvalues)))
                            RunningNs[key] = RunningNs[key]+len(BDfiltvalues)
                        except KeyError:
                            RunningMeans[key] = np.nansum(BDfiltvalues)/np.sum(~np.isnan(BDfiltvalues))
                            RunningNs[key] = np.sum(~np.isnan(BDfiltvalues))
                        BaseData.append(BD)
                    else:
                        BaseData.append([])    
                BaseDataDF['local'] = BaseData
        self.BaseData = BaseDataDF
        self.StationMeans = RunningMeans

    def ApplyCorrections(self, BaseData = None):
        if BaseData is None:
            BaseData = self.BaseData
        #segments
        segments = self.data_timeslices
        segmentIDs = segments.index
        #series of times
        ts = self.data.DateTime
        vals = []
        for i,t in enumerate(ts): #iterate through df by index
            #identify segments - initially we allow more than one, but most data should fit into one only    
            seg_ids = [SegID for SegID in segmentIDs if t > segments['begin'][SegID] and t < segments['end'][SegID]] #is it in the time window
            if len(seg_ids)>1:
                #if more than one pick the one that is closest to middle of a segment
                seg_times_from_middle = [np.abs(t-(segments['begin'][SegID]+segments['begin'][SegID])/2) for SegID in seg_ids]
                seg_id = seg_ids[np.argmin(seg_times_from_middle)]
            else:
                seg_id = seg_ids[0]    
            t = t.floor('min') #truncate to minute to match the base data
            seg_time_from_start = t-segments['begin'][seg_id].floor('min')
            seg_index = seg_time_from_start//pd.Timedelta(1,unit = 'm') # 1 minute data is assumed
            vs = {} 
            for key in self.BaseLocs.keys():
                try:
                    tb = BaseData[key][seg_id][0][seg_index] #read base time
                except IndexError:
                    vs[key] = np.nan
                    break
                try:
                    t = t.tz_localize('UTC') #somehow we lost the UTC sometimes?
                except TypeError:
                    pass
                dt = tb-t #check if time is correct and correct the index if needed .. +/- 1 minute is done silently
                if dt != pd.Timedelta(0,unit = 'm'): 
                    seg_index -= dt//pd.Timedelta(1,unit = 'm')
                    if dt > pd.Timedelta(1,unit = 'm'): #flag if the adjustement is more than 1 minute
                        print('corrected indexing error for base {}, segment {}, for t, tb, dt of {},{},{} to new index {} and tb {}'.format(key,seg_id,t,tb,dt,seg_index,BaseData[key][seg_id][0][seg_index]))
                vs[key] = BaseData[key][seg_id][3][seg_index]-self.StationMeans[key] #filtered TMI anomaly minus the long-term mean
            vals.append(vs)    
        self.data['BaseCorrs'] = vals
        # now generate the weighted sum for each point 
        values = []
        for i,val in enumerate(vals):
            v = []
            for key in val.keys():
               #here we do NOT rescale the weights to account for nans...just sum the valid corrections at their weight
               #this is preferred behaviour I think but an argument could be made to re-weight here - perhaps something to test?
               v.append(val[key]*self.B_Weights[key][i])   
            values.append(np.nansum(v))
        self.data['BaseCorrection'] = values
        self.data['TMI_BaseCorrected'] = self.data[self.data_prop].sub(self.data.BaseCorrection, fill_value = 0)       

    def RunBaseCorrection(self, num_bases = 1, exponent = 2.0, scaleby = 'median', filter_period = 120.):
     self.MultiBase.GetBasesandLocs(num_bases)
     self.GetBaseWeights(exponent = exponent, scaleby = scaleby)
     self.GetBaseData(maxtime = filter_period*12, padtime = filter_period*6, filter_period = filter_period, tunit = 'm')
     self.ApplyCorrections() 
     return(self.data)


class MedianLevelling:
    def __init__(self, data, data_prop = 'RawTMI', proj = None, data_limits = [(None,None),(None,None)], nanvalue = -99999.9):
        required_atts = ['LineName','Lon','Lat','X','Y','Height_WGS84','RawTMI']
        for i in required_atts:
            if i not in data.keys():
                raise AttributeError('DataFrame object has no attribute {}'.format(i))
        try:
            #if suitable limits provided cut the data
            self.data = data[(data['X']<data_limits[1][0]) & (data['X']>data_limits[0][0]) & (data['Y']<data_limits[1][1]) & (data['Y']>data_limits[0][1])].reset_index(drop=True)
        except TypeError:
            #otherwise use the whole dataset
            self.data = data
        try:
            self.proj = Proj(proj)
        except TypeError:
            #if not provided assume polar stereographic
            self.proj = Proj("+proj=stere +lat_0=-90 +lat_ts=-71")
        self.line_names = np.unique(self.data.LineName)
        self.data[data_prop] = np.where(self.data[data_prop] == nanvalue, np.nan, self.data[data_prop])
        self.data_prop = data_prop
        
    def split_lines_by_name(self):
        #a dict keyed by line name
        self.data_lines = {}
        for line in self.line_names:
            indices = np.where(self.data.LineName == line)[0]
            self.data_lines[line] = self.data.loc[indices]

    def GetIntersects(self, Tol = None):
        Intersects = []
        donelines = [] # we only identify intersections once...
        for flightline in self.data_lines:
            #print("finding ties for flight line {}".format(flightline))
            fl_data = self.data_lines[flightline] 
            for tieline in self.data_lines:
                if tieline != flightline and tieline not in donelines: #no self-intersects and no repeats
                    tl_data = self.data_lines[tieline]
                    Overlap = TestOverlap(fl_data,tl_data)
                    if Overlap:
                        Dists,IDs = ComputeMinDistance(fl_data,tl_data)
                        if Tol is None:
                            flspacing = (np.nanmax(fl_data['Distance'])-np.nanmin(fl_data['Distance']))/(2*len(fl_data))
                            tlspacing = (np.nanmax(tl_data['Distance'])-np.nanmin(tl_data['Distance']))/(2*len(tl_data))
                            tol = np.sqrt(flspacing**2 + tlspacing**2)
                        else:
                            tol=Tol
                        fl_loc,tl_loc = Intersect(Dists,IDs,tol)       
                        if len(fl_loc) > 0:
                            for (v1,v2) in zip(fl_loc,tl_loc):
                                X = fl_data['X'].iloc[v1]
                                Y = fl_data['Y'].iloc[v1]
                                Fvalue = fl_data[self.data_prop].iloc[v1]
                                Tvalue = tl_data[self.data_prop].iloc[v2]
                                FtoT = Tvalue-Fvalue
                                Intersects.append([X,Y,flightline,tieline,v1,v2,Fvalue,Tvalue,FtoT])
            donelines.append(flightline)               
        self.Intersects = pd.DataFrame(Intersects, columns = ['X','Y','flightline','tieline','fl_loc','tl_loc','fl_TMI','tl_TMI','FtoTDiff'])

    def MedianBasedLevelling(self, TM = None, MaxCycles = 20):
        Inters = self.Intersects
        CDs = Inters['FtoTDiff'].abs()
        print('maximum initial absolute crossdiff is {}, median absolute crossdiff is {}, mean absolute crossdiff is {}'.format(np.nanmax(CDs), np.nanmedian(CDs), np.nanmean(CDs)))
        
        # if no target is specified aim for 10% of the median 
        if TM is None:
            TM = 0.1*np.nanmedian(CDs) #10% of median of medians
        print('Target worst median is {}'.format(TM))
        
        #make the empty dataframe    
        Medians = pd.DataFrame(columns = ['Line','Median','AbsMedian','DCShift','Crosses']) 
        
        #for each line identify 'crosslines'
        lines = []
        Crosses = {}
        for line in self.data_lines:    
            lines.append(line)
            Crosses[line] = GetCrosses(Inters,line)
        Medians['Crosses'] = Crosses
        Medians['Line'] = lines
        
        #calculate medians from crosses and store in a dataframe
        Meds = [] 
        for line in self.data_lines:    
            Med = GetMedian(line,Medians['Crosses'])
            Meds.append(Med)
        
        Medians['Median'] = Meds
        Medians['AbsMedian'] = np.abs(Meds)
        Medians['DCShift'] = np.zeros_like(Meds)
        
        #sort dataframe by absolute median 
        Medians.sort_values(by ='AbsMedian', ascending = False, inplace = True)
        #identify line with largest absolute median
        WM = Medians['Median'].iloc[0]
        WML = Medians['Line'].iloc[0]
        print ("Initial worst line median is {} for line {}".format(WM,WML))

        #set up loop to iterate several cycles:
        Cycle = 0
        while np.abs(WM) > TM and Cycle < MaxCycles:
            #Set up loop to do corrections line by line - worst first order
            for idx in Medians.index:
                line = Medians['Line'][idx]
                Mval = Medians['Median'][idx] #this is the median value to be removed
                #modify the DCShift value and median on the active line to reflect adjustment (subtract Mval)
                Medians.loc[idx,'DCShift'] += Mval
                Medians.loc[idx,'Median'] -= Mval
                #print(Medians.loc[idx])
                #modify the value and intersects on cross-lines to reflect adjustment (subtract Mval on line, add on crossline)
                for crossline in Medians.loc[idx,'Crosses']:
                    Medians.loc[idx,'Crosses'][crossline] -= Mval #correct all ties on the line
                    c_idx = Medians['Line'].loc[Medians['Line'] == crossline].index[0] #find crossline
                    Medians.loc[c_idx,'Crosses'][line] +=Mval #correct tie on the crossline
                    newMed = GetMedian(crossline,Medians['Crosses']) #recalculate median for the crossline including update
                    Medians.loc[c_idx,'Median'] = newMed
            Medians['AbsMedian'] = np.abs(Medians['Median'])
            #re-sort by absolute value for next cycle
            Medians.sort_values(by ='AbsMedian', ascending = False, inplace = True)
            #report worst median
            WM = Medians['Median'].iloc[0]
            WML = Medians['Line'].iloc[0]
            print ("Worst line median after Cycle {} is {} for line {}".format(Cycle,WM,WML))
            Cycle +=1
        self.medians = Medians
     
    def ApplyLevelling(self):
        self.data['TMI_Median_Levelled'] = self.data[self.data_prop] #get data values
        LineNames = self.data['LineName'] #get data values
        for line in self.medians['Line']:
            DCShift = self.medians.loc[line]["DCShift"]
            if(~np.isnan(DCShift)):
                #print('DCshift {} applied to line {}'.format(DCShift,line))
                NewValues = self.data['TMI_Median_Levelled'][LineNames == line] + DCShift
                self.data.update({'TMI_Median_Levelled':NewValues})
        self.data['DCShift'] = self.data['TMI_Median_Levelled']-self.data[self.data_prop]   

    def RunMedianlevelling(self, Tol = None, TM = None, MaxCycles = 20):
        self.split_lines_by_name()
        self.GetIntersects(Tol=Tol)
        self.MedianBasedLevelling(TM = TM, MaxCycles=MaxCycles) #desired fit in nT
        self.ApplyLevelling()
        return self.data


#Helper Functions used in various places    
def fourier_LP_filter(signal, sampling_rate, cutoff_freq):
    #check for nans and if found interpolate
    if np.any(np.isnan(signal)):
        signal_s = np.linspace(0,len(signal)-1,len(signal))
        signal_nn = np.where(~np.isnan(signal))[0]
        f = interpolate.interp1d(signal_s[signal_nn],signal[signal_nn], fill_value = 'extrapolate')
        signal = f(signal_s)
    signal_fft = fft(signal)
    freqs = fftfreq(len(signal), d=1/sampling_rate)
    # here we maybe cause Gibbs Oscillations?
    mask = np.abs(freqs) < cutoff_freq
    filtered_fft = signal_fft * mask
    filtered_signal = ifft(filtered_fft).real
    return filtered_signal
     
def fourier_BW_filter(signal, cutoff_freq, order):
    #check for nans and if found interpolate
    if np.any(np.isnan(signal)):
        signal_s = np.linspace(0,len(signal)-1,len(signal))
        signal_nn = np.where(~np.isnan(signal))[0]
        f = interpolate.interp1d(signal_s[signal_nn],signal[signal_nn], fill_value = 'extrapolate')
        signal = f(signal_s)
    sos = butter(order,cutoff_freq,output='sos')
    filtered_signal = sosfiltfilt(sos, signal)
    return filtered_signal 

def IGRFcorrection(Lat,Lon,H,DateTime):
    # Read IGRF coeffs
    IGRF_FILE = r'./IGRF14.shc'
    igrf = iut.load_shcfile(IGRF_FILE, None)
    alt, colat = iut.gg_to_geo(H/1000, 90-Lat)
    #convert DateTime to decimal years (just year and day of year...)
    DTyears = [float(DateTime.strftime("%Y"))+float(DateTime.strftime("%j"))/365 for DateTime in DateTime]
    f = interpolate.interp1d(igrf.time, igrf.coeffs, fill_value='extrapolate')
    coeffs = f(DTyears)
    Br, Bt, Bp = iut.synth_values(coeffs.T, alt, colat, Lon, igrf.parameters['nmax'])
    dec, hoz, inc, eff = iut.xyz2dhif(-Bt,Bp,-Br)
    IGRFvals = eff # get IGRF/DGRF values
    return IGRFvals

def GreatCircleDist(Base,Data):
    theta = np.arccos(np.sin(Base[1]*np.pi/180)*np.sin(Data[1]*np.pi/180)+np.cos(Base[1]*np.pi/180)*np.cos(Data[1]*np.pi/180)*np.cos(np.abs(Base[0]*np.pi/180-Data[0]*np.pi/180)))   
    return theta

def GetInterMagnetData(BaseCode,Begin,End, silent_mode = False): 
    #here we use the HAPI to access Intermagnet data from the web
    start      = str(Begin).split()[0]+'T'+str(Begin).split()[1][0:8] + 'Z'
    stop       = str(End).split()[0]+'T'+str(End).split()[1][0:8] + 'Z'
    
    #main intermagnet server
    IMserver     = 'https://imag-data.bgs.ac.uk/GIN_V1/hapi'
    IMdataset    = BaseCode + '/best-avail/PT1M/xyzf'
    IMparameters = 'Field_Magnitude'
    
    #WDC server (for older data - definitive only)
    WDCserver = 'https://wdcapi.bgs.ac.uk/hapi'
    WDCdataset    = BaseCode + '/PT1M/xyz'
    WDCparameters = 'Field_Vector'
    
    def check_all_in_range(server,dataset,start,stop):
        TriesRemaining = 5
        while TriesRemaining > 0:
            try:        
                S = hapi(server, dataset)
                TriesRemaining = 0
            except:
                #try again...
                TriesRemaining -= 1
        SD = isoparse(S['startDate']) > isoparse(start)
        ED = isoparse(S['stopDate']) < isoparse(stop) 
        if ED or SD:
            print('dataset {}/{} does not cover data period: {},{}'.format(server,dataset,start,stop))
            check = False
        else: 
            check = True
        return check
    
    # Check if data is in base station time range
    checkIM = check_all_in_range(IMserver, IMdataset, start, stop)
    checkWDC = check_all_in_range(WDCserver, WDCdataset,start, stop)
    if checkIM:
        try:
            print ('accessing intermagnet dataset {} for data period {},{}'.format(IMdataset,start,stop))
            data, meta = hapi(IMserver, IMdataset, IMparameters, start, stop)
            if np.all(np.array([data[1] for data in data])==99999.):
                raise Exception()
        except:
            print ('something went wrong for intermagnet, trying WDC instead')
            try:
                data, meta = hapi(WDCserver, WDCdataset, WDCparameters, start, stop)
                #here we only include WDC data if xyz are all not NaN 
                data = np.where([(j[0],np.any(j[1]==99999.)) for i,j in enumerate(data)],
                                [(item[0], np.nan) for item in data],
                                [(item[0], np.sqrt(np.sum([X**2 for X in item[1]]))) for item in data])
                if np.all(np.isnan([data[1] for data in data])):
                    raise Exception()
            except:
                print ('something went wrong with WDC also, no data returned')
                data = None 
    elif checkWDC:
        try:
            print ('accessing WDC dataset {} for data period {},{}'.format(WDCdataset,start,stop))
            data, meta = hapi(WDCserver, WDCdataset, WDCparameters, start, stop)
            data = np.where([(j[0],np.any(j[1]==99999.)) for i,j in enumerate(data)],
                            [(item[0], np.nan) for item in data],
                            [(item[0], np.sqrt(np.sum([X**2 for X in item[1]]))) for item in data])
            if np.all(np.isnan([data[1] for data in data])):
                raise Exception()
        except:
            print ('something went wrong with WDC, no data returned')
            data = None 
    else:
         print ('no data available, no data returned')
         data = None # if we don't have base data
    return data

def TestOverlap(flightline,tieline):
    if np.nanmax(tieline.X) < np.nanmin(flightline.X) or np.nanmin(tieline.X) > np.nanmax(flightline.X):
          Overlap = False
    elif np.nanmax(tieline.Y) < np.nanmin(flightline.Y) or np.nanmin(tieline.Y) > np.nanmax(flightline.Y):
          Overlap = False
    else:
          Overlap = True
    return Overlap

def ComputeMinDistance(flightline,tieline):
    fl_array = np.transpose([flightline.X,flightline.Y])
    tl_array = np.transpose([tieline.X,tieline.Y])
    tree = spatial.KDTree(tl_array)
    Result = tree.query(fl_array)
    return Result
    
def Intersect(DistArray,IDarray,Tol):
    Lloc = np.where(DistArray<Tol)
    Tloc = IDarray[Lloc]
    return Lloc[0],Tloc

def GetMedian(line, Crosses):
    crosslines = Crosses[line].keys()
    vals = []
    for crossline in crosslines:
        vals.append(Crosses[line][crossline])  
    Med = np.nanmedian(vals)
    return Med

def GetCrosses(Intersects,line):
    Crosses = {}
    fl_ties = Intersects['tieline'].iloc[np.where(Intersects['flightline']==line)[0]]
    fl_vals = Intersects['FtoTDiff'].iloc[np.where(Intersects['flightline']==line)[0]]
    for fl in fl_ties:
        Crosses[fl] = np.nanmedian(fl_vals[fl_ties == fl]) #take the median value
    tl_ties = Intersects['flightline'].iloc[np.where(Intersects['tieline']==line)[0]]
    tl_vals = -Intersects['FtoTDiff'].iloc[np.where(Intersects['tieline']==line)[0]]
    for tl in tl_ties:
        Crosses[tl] = np.nanmedian(tl_vals[tl_ties == tl]) #take the median value
    return Crosses