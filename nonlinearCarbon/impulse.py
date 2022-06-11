from param import *
from library import *
from function import *


## Pattern = 1
if ImpulsePattern == 0:
    """Equivalent Impulse Horizon with Hetero-Value"""
    # ImpulseMin = 0 
    # ImpulseMax = 1100
    # ImpulseStep = 100
    # ImpulsePathSize = int((ImpulseMax-ImpulseMin)/ImpulseStep )

    Carbon   = np.array([0, 100, 500, 1000, 5000, 10000, 50000, 100000])

    ImpulsePathSize = len(Carbon)
    CeMatrix = np.zeros((ImpulsePathSize,t_span))

    CeMatrix[:,0] =     Carbon[:] /2.13

elif ImpulsePattern ==1:
    """Heterogenous Impulse Horizon with Homo-value"""
    ImpulsePathSize = 10
    ImpulseValue = 100
    
    CeMatrix = ImpulseValue*np.eye(ImpulsePathSize, t_span)

elif ImpulsePattern ==2:
    """Fixed Impulse Response"""
    ImpulsePathSize = 2
    ImpulseValue = 10
    
    CeMatrix = np.zeros((ImpulsePathSize, t_span))
    CeMatrix[1,:] = ImpulseValue*np.ones((1,t_span))/2.13


## cearth, tauc Path

cearth_taucMatrix = [[35., 6603. ],
                     [0.107, 20]    ]

cearth_taucMatrixSize = len(cearth_taucMatrix)



## Looping

for ctpathnum in range(cearth_taucMatrixSize):
    figwidth = 10
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(2  * figwidth, 2 *figwidth))
    TvmidBase = np.zeros(10000)

    for pathnum in range(ImpulsePathSize):


        Ce = CeMatrix[pathnum,:]
        cearth, tauc = cearth_taucMatrix[ctpathnum]

        tv, Tvmid, Cv = model(Ts, Cs, cearth, tauc, Ce)

        plotnum = ImpulsePattern*pathnum

        if pathnum ==0:
            TvmidBase = Tvmid

        axs[0].plot(tv, Tvmid, label=f"ImpulseValue_{CeMatrix[pathnum,plotnum]*2.13}")
        axs[0].set_xlabel('Time (year)',fontsize = 16)
        axs[0].set_ylabel('Temperature  (K)',fontsize = 16)
        axs[0].set_title('Temperature Anomaly Dynamics')
        axs[0].grid(linestyle=':')
        axs[0].legend()
        axs[1].plot(tv, Cv, label=f"ImpulseValue_{CeMatrix[pathnum,plotnum]*2.13}")
        axs[1].set_xlabel('Time (year)')
        axs[1].set_ylabel('Carbon (ppm)')
        axs[1].set_title('Carbon Concentration Dynamics')
        axs[1].grid(linestyle=':')
        axs[1].legend()
        axs[2].plot(tv, Tvmid-TvmidBase, label=f"ImpulseValue_{CeMatrix[pathnum,plotnum]*2.13}_Compared2_0")
        axs[2].set_xlabel('Time (year)',fontsize = 16)
        axs[2].set_ylabel('Degree Celsius',fontsize = 16)
        axs[2].set_title('Impulse Response per Teratonne of Carbon')
        axs[2].grid(linestyle=':')
        axs[2].legend()




    plt.tight_layout()
    plt.savefig(f"ImpulsePtn_{ImpulsePattern}_cearth_{cearth}_tauc_{tauc}.pdf")


