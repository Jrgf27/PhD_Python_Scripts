from sys import argv,exit
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QPushButton, QLineEdit, QFileDialog, QComboBox 
from PyQt5 import QtGui
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os

###### Experiment Constants ######
CELLCONSTANT_CELLCHAMBER_BASO = 1.6e-4
#CELLCONSTANT_CELLCHAMBER_BASO_calculated = 3.8553e-4
CELLCONSTANT_CELLCHAMBER_APICAL = 4.43e-4
CELLCONSTANT_REFERENCE= 1.8856e-3
TIME_PER_MEASUREMENT=17
BOUNDARIES_PERCENTAGE=0.1
ELECTRODE_AREA_RATIO=2.3
CELL_GROWTH_AREA=0.2

###### Initial Guesses ######
X0REF_Resistance=400
X0REF_CPE_MAGNITUDE=1e-4
X0REF_CPE_EXPONENT=0.3
X0CELL_RESISTANCE=10
X0CELL_CAPACITANCE=3e-8

###### Lower Boundary ######
LBREF_RESISTANCE=0
LBREF_CPE_MAGNITUDE=1e-6
LBREF_CPE_EXPONENT=0
LBCELL_RESISTANCE=0
LBCELL_CAPACITANCE=1e-9

###### Upper Boundary ######
UBREF_RESISTANCE=2000
UBREF_CPE_MAGNITUDE=1e-3
UBREF_CPE_EXPONENT=1
UBCELL_RESISTANCE=100000
UBCELL_CAPACITANCE=1e-5

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

    if valuenumber < (BaselineValues-1):

        Zcpe = (1/(x[1]))*((1j*omega)**(-x[2]))
        RbCellChamber=4280
        Zm=0
        value =  Zcpe + (((x[0])+Zm)*RbCellChamber)/((x[0]+Zm)+RbCellChamber)

    else:

        Zcpe = (1/(x[1]))*((1j*omega)**(-x[2]))
        RbCellChamber=4280
        Cm=(1/(1j*omega*x[3]))
        Zm = np.divide((x[4]*Cm),(x[4]+Cm))
        value1=np.divide(((x[0]+(2*Zm))*RbCellChamber),((x[0]+(2*Zm))+RbCellChamber))
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
    ## Nelder Mead is usually only applicable to unconstrained functions. sin(x) variable transformation is used to allow constraints in the model ##########
    ## More information at "John D'Errico (2020). fminsearchbnd, fminsearchcon (https://www.mathworks.com/matlabcentral/fileexchange/8277-fminsearchbnd-fminsearchcon), MATLAB Central File Exchange. Retrieved October 2, 2020." #######
    ## The available MATLAB code was translated into Python and only the required components of the original code were kept ##

    def intrafun(x):
        xtrans=xtransform(x,params)
        fval=eval(params.fun)(np.reshape(xtrans,params.xsize))
        return(fval)

    def xtransform(x,params):
        xtrans=np.zeros(params.xsize[0])
        for i in range (params.n):
            xtrans[i]=(np.sin(x[i])+1)/2
            xtrans[i]=xtrans[i]*(params.UB[i]-params.LB[i])+params.LB[i]
            xtrans[i]=max(params.LB[i],min(params.UB[i],xtrans[i]))
        return(xtrans)

    x0=np.array(x0)
    xsize=x0.shape
    n=len(x0)
    global params
    params=Params([],LB,UB,fun,n,xsize,[],np.zeros(n))
   
    x0u=x0
    for i in range(n):
        x0u[i]=2*(x0[i]-LB[i])/(UB[i]-LB[i])-1
        x0u[i]=2*np.pi+np.arcsin(max(-1,min(1,x0u[i])))

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
    return Complex,Frequency

class LabelTextbox(QMainWindow):

    def __init__(self,parent=None, labeltext=None,xposition=None,yposition=None, width=None, height=None):
        super(LabelTextbox,self).__init__()
        self.Label = QLabel(parent)
        self.Label.setText(labeltext)
        self.Label.move(xposition,yposition)
        self.Label.resize(width,height)
        self.Label.setFont((QtGui.QFont("Arial",12,QtGui.QFont.Bold)))
        self.Text = QLineEdit(parent)
        self.Text.move(xposition, yposition+30)
        self.Text.resize(width,height)
        self.Text.setFont((QtGui.QFont("Arial",12)))

class figure_maker(QMainWindow):
    
    def __init__(self,parent=None,xposition=None,yposition=None, width=None, height=None):
        super(figure_maker,self).__init__()

        self.figure=plt.figure()
        self.canvas=FigureCanvas(self.figure)
        self.canvas.setParent(parent)
        self.canvas.move(xposition,yposition)
        self.canvas.resize(width,height)
        self.toolbar=NavigationToolbar(self.canvas,parent)
        self.toolbar.move(xposition,yposition-20)
        self.toolbar.resize(width ,20)

    def plot_data(self,xdata=[1,2],ydata=[2,1],mode="plain",title=""):

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(xdata,ydata, '*-') 
        ax.set_title(title,fontsize=9)
        ax.tick_params(axis="both",which="major",labelsize=9)
        ax.ticklabel_format(axis='y',style=mode,scilimits=(0,0))
        ax.grid()
        self.canvas.draw() 
    
    def plot_allChips(self,xdata=[1,2],ydata=[2,1],numberofchips=0,mode="plain",title=""):
        self.figure.clear() 
        ax = self.figure.add_subplot(111)
        for i in range(numberofchips):
            ax.plot(xdata,ydata[:,i] , '*-',label=f"Chip {i+1}")
        ax.legend()
        ax.ticklabel_format(axis='y',style=mode,scilimits=(0,0),) 
        ax.set_title(title,fontsize=10)
        self.canvas.draw()
        
class App(QMainWindow):

    def __init__(self):
        super(App,self).__init__()
        self.title = 'Fitting Algorithm'
        self.left = 10
        self.top = 10
        self.width = 1780
        self.height = 1000
        self.initUI()
    
    def initUI(self):

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.CellLineText=LabelTextbox(self, "Cell Line", 20,10,280,40)
        self.PassageNumberText = LabelTextbox(self, "Passage Number", 20,80,280,40)
        self.CellDensityText = LabelTextbox(self, "Cell Density", 320,10,280,40)
        self.ChallengesText = LabelTextbox(self, "Challenges", 320,80,280,40)
        self.NotesText = LabelTextbox(self, "Notes", 620,80,280,40)
        self.BaselineReadingsText = LabelTextbox(self, "Number of baseline readings", 620,10,280,40)

        self.CellResistanceAll=figure_maker(self,20,310, 880, 290)

        self.CellR2Imag=figure_maker(self,20,620,420,180)
        self.CellR2Real=figure_maker(self,20,820,420,180)
        self.RefR2Imag=figure_maker(self,480,620,420,180)
        self.RefR2Real=figure_maker(self,480,820,420,180)

        self.CellResistance=figure_maker(self,920,20,860,180)
        self.CellCapacitance=figure_maker(self,920,220,860,180)
        self.CellApicalResistance=figure_maker(self,1360,420,420,180)
        self.RefMediumResistance=figure_maker(self,920,420,420,180)
        self.CellCPEMagnitude=figure_maker(self,1360,620,420,180)
        self.RefCPEMagnitude=figure_maker(self,920,620,420,180)
        self.CellCPEExponent=figure_maker(self,1360,820,420,180)
        self.RefCPEExponent=figure_maker(self,920,820,420,180)

        self.ImpedanceAnalyserData = QPushButton('Select Data Files to Analyse', self)
        self.ImpedanceAnalyserData.setFont((QtGui.QFont("Arial",18,QtGui.QFont.Bold)))
        self.ImpedanceAnalyserData.move(20,170)
        self.ImpedanceAnalyserData.resize(880,40)

        self.ResultData = QPushButton('Select Result Files (.csv)', self)
        self.ResultData.setFont((QtGui.QFont("Arial",18,QtGui.QFont.Bold)))
        self.ResultData.move(20,230)
        self.ResultData.resize(660,40)

        self.QComboBox=QComboBox(self)
        self.QComboBox.move(700,230)
        self.QComboBox.setFont((QtGui.QFont("Arial",18,QtGui.QFont.Bold)))
        self.QComboBox.resize(200,40)
        self.QComboBox.setDisabled(True)

        self.ImpedanceAnalyserData.clicked.connect(self.on_click_ImpedanceAnalyserData)
        self.ResultData.clicked.connect(self.on_click_ResultsButton)

        self.QComboBox.activated.connect(lambda:self.CellResistance.plot_data(self.days,self.FittingCell[:,self.QComboBox.currentIndex()],"plain","Cell Resistance (\u03A9.cm$^2$)"))
        self.QComboBox.activated.connect(lambda:self.CellCapacitance.plot_data(self.days,self.FittingCell[:,self.numberofchips+self.QComboBox.currentIndex()],"sci", "Cell Capacitance (Farads/cm$^2$)"))
        self.QComboBox.activated.connect(lambda:self.CellApicalResistance.plot_data(self.days,self.FittingCell[:,self.QComboBox.currentIndex()+((self.numberofchips-1)*2+2)],"plain","Cell Chamber Apical Resistance (\u03A9)"))
        self.QComboBox.activated.connect(lambda:self.CellCPEMagnitude.plot_data(self.days,self.FittingCell[:,self.QComboBox.currentIndex()+((self.numberofchips-1)*3+3)],"sci","Cell Chamber CPE Magnitude"))
        self.QComboBox.activated.connect(lambda:self.CellCPEExponent.plot_data(self.days,self.FittingCell[:,self.QComboBox.currentIndex()+((self.numberofchips-1)*4+4)],"plain", "Cell Chamber CPE Exponent"))
        self.QComboBox.activated.connect(lambda:self.CellR2Imag.plot_data(self.days,self.FittingCell[:,self.QComboBox.currentIndex()+((self.numberofchips-1)*6+6)],"plain","Cell Chamber R2 Imaginary Component"))
        self.QComboBox.activated.connect(lambda:self.CellR2Real.plot_data(self.days,self.FittingCell[:,self.QComboBox.currentIndex()+((self.numberofchips-1)*5+5)],"plain", "Cell Chamber R2 Real Component"))
        
        self.QComboBox.activated.connect(lambda:self.RefMediumResistance.plot_data(self.days,self.FittingRef[:,self.QComboBox.currentIndex()],"plain", "Reference Medium Resistance (\u03A9)"))
        self.QComboBox.activated.connect(lambda:self.RefCPEMagnitude.plot_data(self.days,self.FittingRef[:,self.numberofchips+self.QComboBox.currentIndex()],"sci","Reference CPE Magnitude"))
        self.QComboBox.activated.connect(lambda:self.RefCPEExponent.plot_data(self.days,self.FittingRef[:,self.QComboBox.currentIndex()+((self.numberofchips-1)*2+2)],"plain","Reference CPE Exponent"))
        self.QComboBox.activated.connect(lambda:self.RefR2Imag.plot_data(self.days,self.FittingRef[:,self.QComboBox.currentIndex()+((self.numberofchips-1)*4+4)],"plain","Reference R2 Imaginary Component"))
        self.QComboBox.activated.connect(lambda:self.RefR2Real.plot_data(self.days,self.FittingRef[:,self.QComboBox.currentIndex()+((self.numberofchips-1)*3+3)],"plain","Reference R2 Real Component"))

        self.show()
        
    def on_click_ImpedanceAnalyserData(self):
        self.enableUI_Elements(False)
        path=sorted(QFileDialog.getOpenFileNames(self)[0])
        if not self.BaselineReadingsText.Text.text().isdigit():
            self.enableUI_Elements(True)
            self.BaselineReadingsText.Text.setText("")
            return None
        if path == [] or len(path)%2 !=0:
            self.enableUI_Elements(True)
            return None    

        global BaselineValues
        global omega
        global valuenumber
        global RealCell
        global RealRef
        global ImagCell
        global ImagRef
        global FitRef

        ExperimentalNotes=pd.DataFrame([self.CellLineText.Text.text(),self.PassageNumberText.Text.text(),self.CellDensityText.Text.text(),self.ChallengesText.Text.text(),self.NotesText.Text.text()])
        NumberofChips=int(len(path)/2)

        ColumnNameCell=[None]*(NumberofChips*7)
        VariablesCell=[","]*(NumberofChips*7)
        ColumnNameRef= [None]*(NumberofChips*5)
        VariablesRef=[","]*(NumberofChips*5)

        BaselineValues=int(self.BaselineReadingsText.Text.text())
        days=[]
        Combolist=[]

        for k in range(NumberofChips):

        ### Opening each data file and extracting data

            filenameref= path[(k * 2) + 1]
            ComplexRef, Frequency = ObtainData(filenameref)
            
            filenameCell= path[k * 2]
            ComplexCell, _ = ObtainData(filenameCell)

            POINTS_PER_MEASUREMENT=-1
            num=0
            for i in Frequency:
                if i == Frequency[0]:
                    num+=1
                if num==2:
                    break
                POINTS_PER_MEASUREMENT+=1

            pos1=0
            pos2=POINTS_PER_MEASUREMENT

            TotalValues=int(len(ComplexRef)/(POINTS_PER_MEASUREMENT))

            ColumnNameCell[k]=ColumnNameCell[k+NumberofChips]=ColumnNameCell[k+((NumberofChips-1)*2)+2]=ColumnNameCell[k+((NumberofChips-1)*3)+3]= ColumnNameCell[k+((NumberofChips-1)*4)+4]=ColumnNameCell[k+((NumberofChips-1)*5)+5]=ColumnNameCell[k+((NumberofChips-1)*6)+6]=f"Chip{k+1},"
            ColumnNameRef[k]=ColumnNameRef[k+NumberofChips]=ColumnNameRef[k+((NumberofChips-1)*2)+2]=ColumnNameRef[k+((NumberofChips-1)*3)+3]= ColumnNameRef[k+((NumberofChips-1)*4)+4]=f"Chip{k+1},"
            Combolist.append(f"Chip {k+1}")

            if k==0:
                FittingRef=np.zeros((TotalValues,NumberofChips+((NumberofChips-1)*4)+4))
                FittingCell=np.zeros((TotalValues,NumberofChips+((NumberofChips-1)*6)+6))

            VariablesCell[k]="CellBarrierResistance,"
            VariablesCell[k+NumberofChips]= "CellBarrierCapacitance,"
            VariablesCell[k+((NumberofChips-1)*2)+2]="ApicalResistance,"
            VariablesCell[k+((NumberofChips-1)*3)+3]="CPEMagnitude,"
            VariablesCell[k+((NumberofChips-1)*4)+4]="CPEExponent,"
            VariablesCell[k+((NumberofChips-1)*5)+5]="R2Real,"
            VariablesCell[k+((NumberofChips-1)*6)+6]="R2Imaginary,"

            VariablesRef[k]="CellMediumResistance,"
            VariablesRef[k+NumberofChips]= "CPEMagnitude,"
            VariablesRef[k+((NumberofChips-1)*2)+2]="CPEExponent,"
            VariablesRef[k+((NumberofChips-1)*3)+3]="R2Real,"
            VariablesRef[k+((NumberofChips-1)*4)+4]="R2Imaginary,"

            for valuenumber in range (TotalValues):
                ImagRef=ComplexRef[pos1:pos2].imag
                RealRef=ComplexRef[pos1:pos2].real
                ImagCell=ComplexCell[pos1:pos2].imag
                RealCell=ComplexCell[pos1:pos2].real
                omega=np.array(Frequency[pos1:pos2]*2*np.pi)

        ####    Reference Electrode Fitting

                x0ref=[X0REF_Resistance,X0REF_CPE_MAGNITUDE,X0REF_CPE_EXPONENT]

                LB=[LBREF_RESISTANCE,LBREF_CPE_MAGNITUDE,LBREF_CPE_EXPONENT]
                UB=[UBREF_RESISTANCE,UBREF_CPE_MAGNITUDE,UBREF_CPE_EXPONENT]

                FitRef=Nelder_Mead_Bounds('RefMin',x0ref,LB,UB)

                valueRef=ReferenceModel(FitRef,2)

                R2RealRef = np.corrcoef(RealRef,valueRef.real)[0, 1]**2
                R2ImagRef = np.corrcoef(ImagRef,valueRef.imag)[0, 1]**2

                FittingRef[valuenumber,k]=FitRef[0]
                FittingRef[valuenumber,k+NumberofChips]=FitRef[1]
                FittingRef[valuenumber,k+((NumberofChips-1)*2)+2]=FitRef[2]
                FittingRef[valuenumber,k+((NumberofChips-1)*3)+3]=R2RealRef
                FittingRef[valuenumber,k+((NumberofChips-1)*4)+4]=R2ImagRef

        ####    Cell Chamber Electrode Fitting

                RbCellChamber=4280
                CellChamberRatio = CELLCONSTANT_CELLCHAMBER_APICAL/CELLCONSTANT_CELLCHAMBER_BASO
                TeorRA=1551
                
                x0cell=[TeorRA,ELECTRODE_AREA_RATIO*FitRef[1],FitRef[2],X0CELL_CAPACITANCE,X0CELL_RESISTANCE]

                LB=[TeorRA-(TeorRA*BOUNDARIES_PERCENTAGE),(ELECTRODE_AREA_RATIO*FitRef[1])-((ELECTRODE_AREA_RATIO*FitRef[1])*BOUNDARIES_PERCENTAGE),FitRef[2]-(FitRef[2]*BOUNDARIES_PERCENTAGE),LBCELL_CAPACITANCE,LBCELL_RESISTANCE]
                UB=[TeorRA+(TeorRA*BOUNDARIES_PERCENTAGE),(ELECTRODE_AREA_RATIO*FitRef[1])+((ELECTRODE_AREA_RATIO*FitRef[1])*BOUNDARIES_PERCENTAGE),FitRef[2]+(FitRef[2]*BOUNDARIES_PERCENTAGE),UBCELL_CAPACITANCE,UBCELL_RESISTANCE]
                FitCell=Nelder_Mead_Bounds('CellChamberMin',x0cell,LB,UB)

                valueCell=CellChamberModel(FitCell,2)

                R2RealCell = np.corrcoef(RealCell,valueCell.real)[0, 1]**2
                R2ImagCell = np.corrcoef(ImagCell,valueCell.imag)[0, 1]**2

                FittingCell[valuenumber,k+((NumberofChips-1)*2)+2]=FitCell[0]
                FittingCell[valuenumber,k+((NumberofChips-1)*3)+3]=FitCell[1]
                FittingCell[valuenumber,k+((NumberofChips-1)*4)+4]=FitCell[2]
                if valuenumber < BaselineValues:
                    FittingCell[valuenumber,k+NumberofChips]=0
                    FittingCell[valuenumber,k]=0
                else:
                    FittingCell[valuenumber,k+NumberofChips]=FitCell[3]/CELL_GROWTH_AREA
                    FittingCell[valuenumber,k]=FitCell[4]*CELL_GROWTH_AREA
                FittingCell[valuenumber,k+((NumberofChips-1)*5)+5]=R2RealCell
                FittingCell[valuenumber,k+((NumberofChips-1)*6)+6]=R2ImagCell

                if k==0:
                    days.append((valuenumber-BaselineValues)*(TIME_PER_MEASUREMENT/60/24))

                #### Data from Impedance analyser has one empty row after the frequency sweep #####
                pos1+=POINTS_PER_MEASUREMENT+1
                pos2+=POINTS_PER_MEASUREMENT+1   
                print(((valuenumber+1)/TotalValues*100),k+1)         

        NewPath=os.path.dirname(filenameCell) + os.sep + "Results" + os.sep
        try:
            os.mkdir(NewPath)
        except:
            pass

        FittingCellResults = pd.DataFrame(FittingCell,index=days)
        FittingRefResults = pd.DataFrame(FittingRef,index=days)
        with open(NewPath+'CellResults.csv','w',newline='\n') as f:
            f.write(","+"".join(VariablesCell)+'\n')
            f.write("Days,"+"".join(ColumnNameCell)+'\n')
            FittingCellResults.to_csv(f, header=False,index=True)
        
        with open(NewPath+'RefResults.csv','w',newline='\n') as f:
            f.write(","+"".join(VariablesRef)+'\n')
            f.write("Days,"+"".join(ColumnNameRef)+'\n')
            FittingRefResults.to_csv(f,header=False,index=True)
        
        with open(NewPath + 'Experimental Details.txt', 'w' , newline='\n') as f:
            ExperimentalNotes.to_csv(f,header=False, index=False)

        self.days=days
        self.FittingCell=FittingCell
        self.FittingRef=FittingRef
        self.numberofchips=NumberofChips
        
        self.QComboBox.clear()
        self.QComboBox.addItems(Combolist)
        self.refreshtextbox("")
        self.enableUI_Elements(True)
        self.refreshfigures([],[])
        self.plotAllResistances()

    def on_click_ResultsButton(self):

        self.enableUI_Elements(False)
        path=sorted(QFileDialog.getOpenFileNames(self)[0])
        if path == [] or len(path)%2 !=0:
            self.enableUI_Elements(True)
            return None
        RefData=pd.read_csv(path[1],header=None,index_col=None,skiprows=2)
        CellData=pd.read_csv(path[0],header=None,index_col=None,skiprows=2)
        self.FittingRef=np.array(RefData)[:,1:(RefData.shape[1])]
        self.FittingCell=np.array(CellData)[:,1:(CellData.shape[1])]
        self.days=np.array(RefData)[:,0]
        self.numberofchips=int((RefData.shape[1]-1)/5)
        
        Combolist=[f"Chip {i+1}" for i in range (self.numberofchips)]

        self.QComboBox.clear()
        self.QComboBox.addItems(Combolist)
        self.enableUI_Elements(True)
        self.refreshfigures([],[])
        self.plotAllResistances()

    def enableUI_Elements(self,command=True):
        self.ChallengesText.Text.setEnabled(command)
        self.PassageNumberText.Text.setEnabled(command)
        self.CellDensityText.Text.setEnabled(command)
        self.CellLineText.Text.setEnabled(command)
        self.NotesText.Text.setEnabled(command)
        self.BaselineReadingsText.Text.setEnabled(command)
        self.ImpedanceAnalyserData.setEnabled(command)
        self.QComboBox.setEnabled(command)
        self.ResultData.setEnabled(command)
    
    def refreshtextbox(self,text=""):
        self.ChallengesText.Text.setText(text)
        self.PassageNumberText.Text.setText(text)
        self.CellDensityText.Text.setText(text)
        self.CellLineText.Text.setText(text)
        self.NotesText.Text.setText(text)
        self.BaselineReadingsText.Text.setText(text)

    def refreshfigures(self,xdata=[],ydata=[]):
        self.CellResistance.plot_data(xdata,ydata)
        self.CellCapacitance.plot_data(xdata,ydata)
        self.CellApicalResistance.plot_data(xdata,ydata)
        self.RefMediumResistance.plot_data(xdata,ydata)
        self.CellCPEMagnitude.plot_data(xdata,ydata)
        self.RefCPEMagnitude.plot_data(xdata,ydata)
        self.CellCPEExponent.plot_data(xdata,ydata)
        self.RefCPEExponent.plot_data(xdata,ydata)
        self.RefR2Imag.plot_data(xdata, ydata)
        self.RefR2Real.plot_data(xdata, ydata)
        self.CellR2Imag.plot_data(xdata, ydata)
        self.CellR2Real.plot_data(xdata, ydata)

    def plotAllResistances(self):
        self.CellResistanceAll.plot_allChips(self.days,self.FittingCell[:,0:self.numberofchips],self.numberofchips,"plain","Cell Resistance All Chips (\u03A9.cm$^2$)")

if __name__ == '__main__':
    app = QApplication(argv)
    ex = App()
    exit(app.exec_())