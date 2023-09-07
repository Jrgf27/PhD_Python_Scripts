import numpy as np
import math
from scipy.optimize import minimize
import pandas as pd
import easygui
import os

class Params(object):
    def __init__(self,x,LB,UB,fun,n,xsize,y,z):
        self.args=x
        self.LB=LB
        self.UB=UB
        self.fun=fun
        self.n=n
        self.xsize=xsize
        self.OutputFcn=y
        self.BoundClass=z
    pass

def MinimizationAlg(RealData,ImagData,value):
    obj=0
    for k in range(0,len(RealData)):
        obj=obj + ((value.real[k]-RealData[k])**2)/RealData[k] + ((value.imag[k]-ImagData[k])**2)/abs(ImagData[k])
    return obj

def ReferenceModel(x,y):

    Zcpe = (1/(x[1]))*((1j*omega)**(-x[2]))
    value=Zcpe+x[0]

    if y==1:
        return MinimizationAlg(RealRef,ImagRef,value)
    else:
        return value

def CellChamberModel(x,y):

    if valuenumber <=BaselineValues:

        Zcpe = (1/(x[1]))*((1j*omega)**(-x[2]))
        RbCellChamber=FitRef[0]*(CellConstantReference/CellConstantCellChamberBaso)
        Zm=0
        value =  Zcpe + (((x[0])+Zm)*RbCellChamber)/((x[0]+Zm)+RbCellChamber)

    else:

        Zcpe = (1/(x[1]))*((1j*omega)**(-x[2]))
        RbCellChamber=FitRef[0]*(CellConstantReference/CellConstantCellChamberBaso)
        Cm=(1/(1j*omega*x[3]))
        Zm = np.divide((x[4]*Cm),(x[4]+Cm))
        value1=np.divide(((x[0]+Zm)*RbCellChamber),((x[0]+Zm)+RbCellChamber))
        value =  np.add(Zcpe,value1)

    if y==1:
        return MinimizationAlg(RealCell,ImagCell,value)
    else:
        return value

def CellChamberMin(x):

    return CellChamberModel(x,1)

def RefMin(x):

    return ReferenceModel(x,1)

def Nelder_Mead_Bounds(fun,x0,LB,UB):

    def intrafun(x):

        xtrans=xtransform(x,params)

        fval=eval(params.fun)(np.reshape(xtrans,params.xsize))

        return(fval)

    def xtransform(x,params):
        xtrans=np.zeros(params.xsize[0])
        k=0

        for i in range (0,params.n):
            xtrans[i]=(math.sin(x[k])+1)/2
            xtrans[i]=xtrans[i]*(params.UB[i]-params.LB[i])+params.LB[i]
            xtrans[i]=max(params.LB[i],min(params.UB[i],xtrans[i]))
            k+=1

        return(xtrans)

    x0=np.array(x0)
    xsize=x0.shape
    n=len(x0)

    global params
    params=Params([],LB,UB,fun,n,xsize,[],np.zeros(n))
   
    x0u=x0
    k=0

    for i in range(0,n):

        x0u[k]=2*(x0[i]-LB[i])/(UB[i]-LB[i])-1
        x0u[k]=2*np.pi+math.asin(max(-1,min(1,x0u[k])))
        k+=1

    results=minimize(intrafun,x0u,method='Nelder-Mead')

    x=xtransform(results.x,params)

    x=np.reshape(x,xsize)

    return(x)

def ObtainData(filename):
    filenamenew=(filename+'_new')
    with open(filename,'r') as f:
        dataraw=f.readlines()
    datatruncated=dataraw[5:len(dataraw)]
    newfile=open(filenamenew,'w')
    with open(filenamenew, 'w') as newfile:
        for item in datatruncated:
            newfile.write("%s" % item)
    Data=pd.read_csv(filenamenew,sep='\t',lineterminator='\n')
    data = pd.DataFrame(Data)
    os.remove(filenamenew)
    data.columns=["Time Initiated","Frequency","Magnitude","Phase"]

    Magni=np.array(data['Magnitude'])
    Phase=np.deg2rad(np.array(data['Phase']))
    Complex=np.multiply(Magni,(np.cos(Phase)+1j*np.sin(Phase)))
    Frequency = data['Frequency']
    omega=np.array(Frequency[pos1:pos2]*2*np.pi)
    return Complex,omega

msg="Enter experiment details"
title="Experiment Log"
CellLine="Cell Type"
PassageNumber="Cell Passage Number"
CellDensity="Cell Density used"
Challenges="Challenges added to cells"
Notes="Notes about the experiment"
BaselineReadings="Number of Baseline readings before cells were added"
fieldNames=[CellLine,PassageNumber,CellDensity,Challenges,BaselineReadings,Notes]
fieldValues = easygui.multenterbox(msg,title, fieldNames)
CellChamberResults=pd.DataFrame([fieldValues[0],fieldValues[1],fieldValues[2],fieldValues[3],fieldValues[5]])
path=easygui.fileopenbox(multiple=True)
NumberofChips=int(len(path)/2)

CellConstantCellChamberBaso = 2.63e-04
CellConstantCellChamberApical = 1.46e-03
CellConstantReference= 0.001868

for k in range(0,NumberofChips):
    pos1=0
    pos2=49
### Opening each data file and extracting data

    filenameref= fr"{path[((k) * 2) + 1]}"
    ComplexRef, omega = ObtainData(filenameref)

    filenameCell= fr"{path[(k) * 2]}"
    ComplexCell, _ = ObtainData(filenameCell)

    TotalValues=int((len(ComplexRef)/50)-1)
    BaselineValues=int(fieldValues[4])

    if k==0:
        FittingRef=np.zeros((TotalValues-1,NumberofChips+((NumberofChips-1)*4)+4))
        FittingCell=np.zeros((TotalValues-1,NumberofChips+((NumberofChips-1)*6)+6))

    for valuenumber in range (1,TotalValues):

        ImagRef=ComplexRef[pos1:pos2].imag
        RealRef=ComplexRef[pos1:pos2].real
        ImagCell=ComplexCell[pos1:pos2].imag
        RealCell=ComplexCell[pos1:pos2].real

####    Reference Electrode Fitting

        x0ref=[400,1e-4,0.3]

        # CPE_Magnitude_Bounds_ref=(1e-6,1e-3)
        # Medium_Resistance_Bounds_ref=(0,1000)
        # CPE_Exponent_Bounds_ref=(0,1)  
        # FitRef = minimize(RefMin , x0ref , method='Powell', bounds=[Medium_Resistance_Bounds_ref,CPE_Magnitude_Bounds_ref,CPE_Exponent_Bounds_ref],options={'maxiter':1e10})
        # FitRef=FitRef.x

        LB=[0,1e-6,0]
        UB=[1000,1e-3,1]
        FitRef=Nelder_Mead_Bounds('RefMin',x0ref,LB,UB)

        valueRef=ReferenceModel(FitRef,2)

        R2RealRef = np.corrcoef(RealRef,valueRef.real)[0, 1]**2
        R2ImagRef = np.corrcoef(ImagRef,valueRef.imag)[0, 1]**2

        FittingRef[valuenumber-1,k]=FitRef[0]
        FittingRef[valuenumber-1,k+NumberofChips]=FitRef[1]
        FittingRef[valuenumber-1,k+((NumberofChips-1)*2)+2]=FitRef[2]
        FittingRef[valuenumber-1,k+((NumberofChips-1)*3)+3]=R2RealRef
        FittingRef[valuenumber-1,k+((NumberofChips-1)*4)+4]=R2ImagRef

####    Cell Chamber Electrode Fitting

        RbCellChamber=FitRef[0]*(CellConstantReference/CellConstantCellChamberBaso)
        CellChamberRatio = CellConstantCellChamberApical/CellConstantCellChamberBaso
        TeorRA=RbCellChamber/CellChamberRatio

        x0cell=[TeorRA,2.3*FitRef[1],FitRef[2],3e-8,10]

        # CPE_Exponent_Bounds_cell=(FitRef[2]-(FitRef[2]*0.05),FitRef[2]+(FitRef[2]*0.05))
        # CPE_Magnitude_Bounds_cell=((2.3*FitRef[1])-((2.3*FitRef[1])*0.05),(2.3*FitRef[1])+((2.3*FitRef[1])*0.05))
        # Apical_Resistance_cell=(TeorRA-(TeorRA*0.1),TeorRA+(TeorRA*0.1))
        # Cell_Capacitance=(1e-9,1e-7)
        # Cell_Resistance=(0,20000)
        # FitCell = minimize(CellChamberMin , x0cell , method='Powell', bounds=[Apical_Resistance_cell,CPE_Magnitude_Bounds_cell,CPE_Exponent_Bounds_cell,Cell_Capacitance,Cell_Resistance],options={'maxiter':1e10})
        # FitCell=FitCell.x

        LB=[TeorRA-(TeorRA*0.1),(2.3*FitRef[1])-((2.3*FitRef[1])*0.05),FitRef[2]-(FitRef[2]*0.05),1e-9,0]
        UB=[TeorRA+(TeorRA*0.1),(2.3*FitRef[1])+((2.3*FitRef[1])*0.05),FitRef[2]+(FitRef[2]*0.05),1e-7,20000]
        FitCell=Nelder_Mead_Bounds('CellChamberMin',x0cell,LB,UB)

        valueCell=CellChamberModel(FitCell,2)

        R2RealCell = np.corrcoef(RealCell,valueCell.real)[0, 1]**2
        R2ImagCell = np.corrcoef(ImagCell,valueCell.imag)[0, 1]**2

        FittingCell[valuenumber-1,k]=FitCell[0]
        FittingCell[valuenumber-1,k+NumberofChips]=FitCell[1]
        FittingCell[valuenumber-1,k+((NumberofChips-1)*2)+2]=FitCell[2]
        FittingCell[valuenumber-1,k+((NumberofChips-1)*3)+3]=FitCell[3]
        FittingCell[valuenumber-1,k+((NumberofChips-1)*4)+4]=FitCell[4]
        FittingCell[valuenumber-1,k+((NumberofChips-1)*5)+5]=R2RealCell
        FittingCell[valuenumber-1,k+((NumberofChips-1)*6)+6]=R2ImagCell

        pos1+=50
        pos2+=50

NewPath=filenameCell[:-35]

FittingCellResults = pd.DataFrame(FittingCell)
FinalCellResults=pd.concat([CellChamberResults,FittingCellResults])
FinalCellResults.to_csv(NewPath+'Cellfile.csv',header=False,index=False)

FittingRefResults = pd.DataFrame(FittingRef)
FinalCellResults=pd.concat([CellChamberResults,FittingRefResults])
FinalCellResults.to_csv(NewPath+'Reffile.csv',header=False,index=False)