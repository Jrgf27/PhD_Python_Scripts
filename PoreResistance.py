#%%
import math

radius=200*10**-9                                      #m
lenght=12*10**-6                                        #m
conductivity=1.5                                        #S/m
poredensity= 1*10**8                                      #holes/cm2
CellChamberArea=math.pi*0.25**2                         #cm2
CellChamberHoles=CellChamberArea*poredensity

RTotalPerPore= (1/(conductivity*radius))*(lenght/(radius*math.pi) + 0.25)

print(("Membrane Thickness {} um\n Pore Density {} holes/cm2\n Pore Size {} nm\n Resistance {} ohms").format(
    lenght * (10**6),
    poredensity,
    round((radius*2)*10**9),
    4*RTotalPerPore/CellChamberHoles))
