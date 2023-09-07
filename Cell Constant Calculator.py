import math
import scipy.special

def CellConstant_perWidth_Calculator(height,gap,length):
    ta=(math.cosh((math.pi*(gap+2*length))/(4*height)))**2
    tb=(math.cosh((math.pi*gap)/(4*height)))**2
    tc=1
    td=0
    kb=math.sqrt(((ta-td)*(tb-tc))/((ta-tc)*(tb-td)))
    kb2=math.sqrt(1-kb**2)
    return scipy.special.ellipk(kb2)/(scipy.special.ellipk(kb)*2)

##Cell Chamber #### Dimensions in m
numberofpoints=1000
lenghts=[i*(1.1e-3/numberofpoints) for i in range(numberofpoints+1)]
height=0.3e-3
heightapical=5.3e-3
gap=2.8e-3
width=2.07e-3
CellConstantBasolateral=0
CellConstantChamber=0
for i in range (numberofpoints+1):
    CellConstantBasolateral+=CellConstant_perWidth_Calculator(height,gap,lenghts[i])*width/(numberofpoints)
    CellConstantChamber+=CellConstant_perWidth_Calculator(heightapical,gap,lenghts[i])*width/(numberofpoints)

CellConstantBasolateral=CellConstantBasolateral*2
CellConstantApical=CellConstantChamber*2-CellConstantBasolateral

##Reference Chamber### Dimensions in m
lenght=0.5e-3
gap=0.2e-3
height=0.3e-3
width=2.8e-3
CellConstantReference=CellConstant_perWidth_Calculator(height,gap,lenght)*width

print(f"Cell Contant Basolateral {CellConstantBasolateral} \n Cell Constant Apical {CellConstantApical} \n Cell Constant Reference {CellConstantReference}")