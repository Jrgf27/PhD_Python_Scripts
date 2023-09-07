#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
Frequency=np.array([i for i in range (10,100000,10)])
Omega=Frequency*2*np.pi
CELLCONSTANT_CELLCHAMBER_BASO = 4.1174e-4
CELLCONSTANT_CELLCHAMBER_APICAL = 1.4622e-3
CPE_Magnitude=3e-5
CPE_Exponent=0.64
Basolateral_Resistance=100
Apical_Resistance=100
# Cell_Capacitance=5e-6
Cell_Resistance=400
magnitude=phase=[]

for i in range(1,10):

    # Cell_Resistance=i*100
    Cell_Capacitance=i*5e-6
    # Zcpe = (1/(CPE_Magnitude))*((1j*Omega)**(-CPE_Exponent))
    Zcpe=0
    Cm=(1/(1j*Omega*Cell_Capacitance*0.2))
    Zm = np.divide(((Cell_Resistance/0.2)*Cm),((Cell_Resistance/0.2)+Cm))
    # value1=np.divide(((Apical_Resistance+Zm)*Basolateral_Resistance),((Apical_Resistance+Zm)+Basolateral_Resistance))   ###System Model
    value1=Zm+Apical_Resistance+Basolateral_Resistance    ####TopDown Model
    value =  np.add(Zcpe,value1)
    magnitude.append(abs(value))
    phase.append(np.degrees(np.angle(value)))
    plt.plot(Frequency,np.degrees(np.angle(value)),label=f"Capacitance={round(Cell_Capacitance*1e6)} uF/cm$^{2}$")
    # plt.plot(Frequency,abs(value),label=f"TEER={Cell_Resistance} \u03A9.cm$^{2}$")
magnitude=pd.DataFrame(magnitude)
phase=pd.DataFrame(phase)
magnitude.to_csv("magnitude.csv",header=False,index=False)
phase.to_csv("Phase.csv",header=False,index=False)
Frequency=pd.DataFrame(Frequency)
Frequency.to_csv("Frequency.csv",header=False,index=False)
plt.xscale('log')
plt.xlabel('Frequency (Hz)',fontsize=22,weight='bold')
plt.ylabel('Phase Angle',fontsize=22,weight='bold')
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend()
plt.show()
# %%
