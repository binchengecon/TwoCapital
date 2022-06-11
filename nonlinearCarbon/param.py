##################################################################
## Section 1.2: Parameter Initialization
##################################################################

## heat capacity, incoming radiation
Q0 = 342.5 #Incoming radiation


## land fraction and albedo
p = 0.3 #Fraction of land on the planet
alphaland = 0.28328 # land albedo

## outgoing radiation linearized
kappa = 1.74
Tkappa = 154

## CO2 radiative forcing
B = 5.35 # Greenhouse effect parameter
C0 = 280 # CO2 params. C0 is the reference C02 level


## ocean carbon pumps
bP = 0.077 # Solubility dependence on temperature (value from Fowler et al)
bB = 0.090 # Biopump dependence on temperature (Value from Fowler)
cod = 0.54 # Ocean carbon pump modulation parameter


## timescale and reference temperature (from Fowler)
tauc = 20 # timescale 
T0 = 288 # Temperature reference


## Coc0 ocean carbon depending on depth
coc0 = 73.78

## CO2 uptake by vegetation
wa = 0.015
vegcover = 0.4
Thigh = 305
Tlow = 275
Topt1 = 285
Topt2 = 295
acc = 5

## Volcanism
V = 0.028

## Anthropogenic carbon
sa = 1 # Switch to take anthropogenic emissions


##################################################################
## Section 3.1: Model Parameter
##################################################################

sa = 1
Ts = 286.7 + 0.56 # 282.9
Cs = 389 # 275.5

#wa = 0.05
#cod = 0.15
alphaland = 0.28
bP = 0.05
bB = 0.08
cod = 3.035

# cearth = 0.107
# tauc = 20
# cearth = 35.
# tauc = 6603.

coc0 =350
## Ocean albedo parameters
Talphaocean_low = 219
Talphaocean_high = 299
alphaocean_max = 0.84
alphaocean_min = 0.255


Cbio_low = 50
Cbio_high = 700

T0 = 298
C0 = 280

## CO2 uptake by vegetation
wa = 0.015
vegcover = 0.4

Thigh = 315
Tlow = 282
Topt1 = 295
Topt2 = 310
acc = 5


##################################################################
## Section 3.1: Function Parameter
##################################################################
t_span = 100


## Impulse Path
ImpulsePattern = 0
