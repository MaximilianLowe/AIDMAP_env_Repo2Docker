import csv
import math as mt
import os
import sys
import time

#Variables and things

# verbosity? - True for all the chat, otherwise "errors" only
v=False

#print output? - True to print data to Outfile, otherwise to stdout
o=True

#input file

Infile = "AN16_4base.csv" #InData Format: InData = [FID,X,Y,Z,Time,TMI_PF,TMI_PM,IData,CSY,DMC,DRV,MAW,SBA,VOS]
#Infile = sys.argv[2] # read in from the command line or bash script

DataInMeters = True

skipindex = 6 #null a base...in this case DMC is [1]...set to 6 to (potentially) include all bases

BaseID = ["CSY","DMC","DRV","MAW","SBA","VOS"]
BaseX = [4500.644,3362.189,3687.527,4240.996,2310.018,3229.858] #km X coordinates in UPS South
BaseY = [1063.555,1075.404,-116.16,3147.763,682.162,1627.817] #km Y coordinates in UPS South
BaseZ = [0.040,3.250,0.030,0.012,0.010,3.488] #m

BaseStart = 3104.0 #Decimal Days of first base station record
BaseSample = 1 #Sample rate of base station record (minutes)

exponent=2. # exponent for length-decay -- 1 is linear, 2 is inverse-squared, 3 is inverse-cubed, values less than 1 will decay more slowly etc
maxbase=4 # maximum number of bases to be used
maxDI=10. #maximum magnetic inclination difference to consider.  Values > 180 will permit all.  

#rmin=182039 #First row of file to be corrected, default corrects all data in the file
#rmax=182042 #Last row of file to be corrected, default corrects all data in the file

VerticalDampingFactor = 0. #Sets the degree of vertical damping of the base station signal to site elevation. 0 is none, 1 is full.
UpOnly = True #set so that corrections are only made "upwards" - i.e. upwards damping only, no downwards amplification
# Vertical damping should reduce leverage as extreme values are accounted for...if not it's bogus!
z0 = [0.04,0.000,0.030,0.012,0.010, -0.200] #reference depth for vertical damping in km - use top of solid Earth
Ib = [-81.696,-80.781,-88.546,-67.903,-80.749,-76.712] # Magnetic field inclinations for base stations in degrees. Testing suggests that this probably doesn't matter
lvd =[100.,100.,100.,100.,100.,100.] #widths of "plates" in km - smaller plates will cause greater damping effect 
EVDF = VerticalDampingFactor*lvd[0]

# Define Functions

def GetBaseWeights(X, Y, Z, e, maxbase,I):
    X=float(X)	
    Y=float(Y)
    Z=float(Z)
    I=float(I)
    if DataInMeters:
        X=X/1000.
        Y=Y/1000.
        Z=Z/1000.
    DI = [0.]*len(BaseID)
    for n in range (0,len(BaseID)):
        Vector=[X-BaseX[n],Y-BaseY[n],Z-BaseZ[n]]
        BaseDist[n] = mt.sqrt(sum(i**2 for i in Vector))
        DI[n] = abs(Ib[n]-I)
    try:
        BaseDist[skipindex]=float(max(BaseDist))+1
        if v:
            print "Station %s excluded (skipindex)" %BaseID[skipindex]
    except IndexError:
        pass  
    BDsort = sorted(BaseDist)
    LS = mt.sqrt(BDsort[maxbase-1]**2)
    for n in range (0,len(BaseID)):
        if DI[n] > maxDI:
            BaseDist[n] = float(max(BaseDist))+1
            if v:
                print "Station %s excluded with inclination difference greater than %s" %(BaseID[n],maxDI)
    for n in range (0,len(BaseID)):
        if BaseDist[n] < LS:
             BW[n] = (1-(BaseDist[n]/LS))**e 
        else:
             BW[n] = 0.
             if v:
               print "r=%s Station %s skipped or excluded. Unless previous, due to distance being greater than length scale" %(r,BaseID[n])
    return (BW, LS)

def GetCorrection(BaseVal,BaseWeight):
    count = 0.
    C = 0.
    for n in range (0,len(BaseID)):
        if BaseVal[n] != -99999.9:
            C = C + BaseVal[n]*BaseWeight[n]
            count = count + BaseWeight[n]
        else:
            count = count + BaseWeight[n]
    try:
        Corr = C/count
    except ZeroDivisionError:
        Corr = 0.
        print "r=%s no data at any included station for BASE_FID %s. no correction applied. More base data or increasing maxbase might help" %(r,BaseData[0])
    Leverage = 0.
    for n in range (0,len(BaseID)):
        if BaseVal[n] != -99999.9:
            Leverage = Leverage + abs(BaseWeight[n]*(BaseVal[n]-Corr))
    return (Corr,Leverage)

def GetData(Sitefile):
    with open(Sitefile,'rb') as csvfile:
        Sfile = csv.reader(csvfile,delimiter=',')
        FID = []
        X = []
        Y = []
        Z = []
        T = []
        TMI_PF = []
        TMI_PM = []
        IData = []
        ID = [FID,X,Y,Z,T,TMI_PF,TMI_PM,IData]
        for row in Sfile:
            n=0
            for n in range (0,8):
                ID[n].append(row[n])
    with open(Sitefile,'rb') as csvfile:
        Sfile = csv.reader(csvfile,delimiter=',')
        BD = []
        for n in range(0,len(BaseID)):
            BD.append([]) 
        for row in Sfile:
            n=0
            for n in range (0,len(BaseID)):
                l=n+8
                if n == skipindex:
                    BD[n].append(-99999.9)
                else:
                    BD[n].append(row[l])
    return (ID,BD)           

def VerticalDamping(ZS,ZB,z0,Ib,l,BV,VD): #Assumes "a thin horizontal plate source" at z0 elevation with width l.
    ZS=float(ZS)
    if DataInMeters:
        ZS=ZS/1000.
    G = [1.]*len(BaseID)
    BV_ds=[-99999.9]*len(BaseID)
    for n in range (0,len(BaseID)):
        BV_F=float(BV[n][r])
        if BV_F != -99999.9:
            ds=ZS-z0[n] #site minus base reference elevation, which must be in km 
            I = mt.radians(Ib[n])# Base station magnetic inclination
            db=ZB[n]-z0[n] # Base minus reference, also in km
            A = mt.sin(2*I)
            B = mt.cos(I)**2-mt.sin(I)**2
            rdb = mt.sqrt(db**2+(l[n]/2)**2) #symmetrical as we assume central location
            rds = mt.sqrt(ds**2+(l[n]/2)**2) #symmetrical as we assume central location
            G[n] = ((1/rds)**2*(A-(l[n]/2)*B)-(1/rds)**2*(A+(l[n]/2)*B))/((1/rdb)**2*(A-(l[n]/2)*B)-(1/rdb)**2*(A+(l[n]/2)*B))
            if UpOnly:
                if ZS < ZB[n]:
                    G[n] = 1.
                    if v:
                            print "vertical damping not computed for station %s" %BaseID[n]
            BV_ds[n] = BV_F +(BV_F*G[n]-BV_F)*VD #VD indicates the extent to damp in this way. 0 is nothing,1 is everything.
    return(BV_ds,G)

#--------------------Functions Applied Here------------------------#
t=time.clock()
print "run commenced at %s" %t
dt = time.clock()-t
#these simply initialise global variables - leave alone
i = len(BaseID)
BaseDist = [None]*i
BW = [0]*i 
B_Val = [-99999.9]*i
IBS = 1440/BaseSample

if v:
    print "BaseIDs: %s" %BaseID
    print "exponent is %s" %exponent
    print "using closest %s base stations for length scale" %maxbase
    
if o:
    IF=os.path.splitext(Infile)
    Outfile = "Results_%s_%s_%s_"%(maxbase,exponent,EVDF)+IF[0]+"_I.txt"
    Backup = "backup"+Outfile
    try:
        os.rename(Outfile,Backup)
    except OSError:
        pass
    f=open(Outfile,'w') # w for write --- i.e. it will overwrite existing data.
    f.write("FID X Y Z DecDay TMI_PF Correction Leverage LengthScale CorrectedTMI\n")
    f.close()
    Dumpfile = "Dump"+Outfile
    fx=open(Dumpfile,'w') #dump file for non essential components (weights etc)
    fx.close()
    
print "reading data from %s" %Infile

Data = GetData(Infile) #InData Format: InData = [FID,X,Y,Z,Time,TMI_PF,TMI_PM,IData,BA,BB,...,BX]

InData = Data[0]
if v:
    print "sample of InData row %s: %s %s %s %s %s %s %s %s" %(0,InData[0][0],InData[1][0],InData[2][0],InData[3][0],InData[4][0],InData[5][0],InData[6][0],InData[7][0])
    print "sample of InData row %s: %s %s %s %s %s %s %s %s" %(1,InData[0][1],InData[1][1],InData[2][1],InData[3][1],InData[4][1],InData[5][1],InData[6][1],InData[7][1])
BaseVals = Data[1]
if v:
    print "sample of BaseVals row %s: %s %s %s %s %s %s" %(0,BaseVals[0][0],BaseVals[1][0],BaseVals[2][0],BaseVals[3][0],BaseVals[4][0],BaseVals[5][0])
    print "sample of BaseVals row %s: %s %s %s %s %s %s" %(1,BaseVals[0][1],BaseVals[1][1],BaseVals[2][1],BaseVals[3][1],BaseVals[4][1],BaseVals[5][0])

dt = time.clock()-t

print "data read in, %s seconds" %dt

try:
  if rmax:
    rmax = rmax
except NameError:  
    rmax=len(InData[0])

try:
  if rmin:
    rmin = rmin
except NameError:  
    rmin=0

if o:
    f=open(Outfile,'a') # a for append
    fx=open(Dumpfile,'a') # a for append

nr = rmax-rmin
print "analysing %s rows...patience may be needed" %nr

for r in range (rmin,rmax):
    if v:
        print "FID %s" %InData[0][r]
    Weights = GetBaseWeights(InData[1][r],InData[2][r],InData[3][r],exponent,maxbase,InData[7][r])
    BaseWeight = Weights[0]
    LengthScale = Weights[1]
    if v:
        print "Base Distances are %s" %BaseDist
        print "Base Weights are %s" %BaseWeight
        print "Length Scale is %s" %LengthScale
    BaseDataDamped = VerticalDamping(InData[3][r],BaseZ,z0,Ib,lvd,BaseVals,VerticalDampingFactor) 
    if v:
        print "Damping Factor of %s" %BaseDataDamped[1]
        print "Damped Base Value is %s" %BaseDataDamped[0]
    Correction = GetCorrection(BaseDataDamped[0],BaseWeight)
    if v:
        print "Correction is %s nT" %Correction[0]
        print "Leverage is %s nT*length unit" %Correction[1]
    TMI = float(InData[5][r]) #5 for full pomme correction, 6 for main field correction only
    CorrectedTMI = TMI-Correction[0]
    if v:
        print "final data:"
        print "FID X Y Z DecDay TMI Correction Leverage CorrectedTMI"
        print "%s %s %s %s %s %s %s %s %s %s" %(InData[0][r],InData[1][r],InData[2][r],InData[3][r],InData[4][r],InData[5][r],Correction[0],Correction[1],LengthScale,CorrectedTMI)
    if o:
        s = "%s %s %s %s %s %s %s %s %s %s\n" %(InData[0][r],InData[1][r],InData[2][r],InData[3][r],InData[4][r],InData[5][r],Correction[0],Correction[1],LengthScale,CorrectedTMI)
        f.write(s)
        s="%s %s %s\n" %(r,BaseDist,BaseWeight)
        fx.write(s)
    #pc = 100*r/nr	
    #if pc % 10 == 0:
        #print "%s percent done at row %s" %(pc,r)
if o:
    f.close()
    fx.close()
    print "Results written to %s and %s " %(Outfile,Dumpfile)
dt = time.clock()-t
print"completed, %s seconds" %dt
