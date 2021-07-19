'''
SCRIPT: ExtractDataForSoot3Eqs
@Authors:
    Alberto Cuoci [1]
    [1]: CRECK Modeling Lab, Department of Chemistry, Materials, and Chemical Engineering, Politecnico di Milano
@Contacts:
    alberto.cuoci@polimi.it
@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: alberto.cuoci@polimi.it
'''

# Import main libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import dataframe_image as dfi
from scipy.sparse import csr_matrix, find
from scipy import sparse
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
plt.style.use('default')

# Define path to PyTools classes
import sys
sys.path.insert(0, '../src')

# Import PyTools classes
from KineticMechanism import KineticMechanism
from OpenSMOKEppXMLFile import OpenSMOKEppXMLFile
from PolimiSootModule import *
from Utilities import *


# --------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------- 

def CheckForMultipleElements(v):
    for i in range(len(v)):
        count = v.count(v[i])
        if (count > 1): print(v[i], " appearing ", count)
            
def CheckForCrossingValues(v1,v2):
    for i in range(len(v1)):
        for j in range(len(v2)):
            if (v1[i] == v2[j]): print(v1[i])  
            

# --------------------------------------------------------------------------------------
# Functions for soot particles/aggregates
# -------------------------------------------------------------------------------------- 
            
def Diameter(MW, rhoSoot):
    
    pi = 3.14159
    Nav_mol = 6.022e23
    return pow( 6./pi * MW/(rhoSoot/1000.) / (Nav_mol), 1./3. ) * 1.e-2

def Volume(d):
    
    pi = 3.14159
    return pi/6.*pow(d,3.)

def Mass(MW):
    
    Nav_kmol = 6.022e26
    return MW/Nav_kmol
    
def BetaKernel(kinetics, rhoSoot, epsilon, T, PAH1, PAH2):
    
    # Constants
    pi = 3.14159
    
    # Boltzmann's constant (m2*kg/s2/K)
    kb = 1.38064852e-23 
    
    # Molecular weights
    MW1 = kinetics.mws[kinetics.species.index(PAH1)]
    MW2 = kinetics.mws[kinetics.species.index(PAH2)]
    
    # Diameters (m)
    d1 = Diameter(MW1, rhoSoot)
    d2 = Diameter(MW2, rhoSoot)
            
    # Volumes
    v1 = Volume(d1)
    v2 = Volume(d2)
    
    # Masses
    m1 = Mass(MW1)
    m2 = Mass(MW2)
    
    print(d1, d2, v1,v2, m1, m2)
    
    Beta = epsilon * np.sqrt(pi*kb/2) * np.sqrt(T) * math.sqrt(1/m1+1/m2) * pow(d1+d2, 2)
    
    return Beta
            
            
# --------------------------------------------------------------------------------------
# Import kinetics
# -------------------------------------------------------------------------------------- 

# Official CRECK2012 Soot BINJOnly
kin_xml_folder_name="C:\\Users\\acuoci\\OneDrive - Politecnico di Milano\\My Projects\\GitHub\\CRECK_DiscreteSectionalModel_v2012\\CRECK_2012_Soot_OnlyBINJ\\kinetics-CRECK_2012_SootOnlyBINJ\\"
kinetics = KineticMechanism(kin_xml_folder_name + "kinetics.xml")
kinetics.ReadKinetics(kin_xml_folder_name + "reaction_names.xml")

# Define groups of species
kinetics.AddGroupOfSpecies('PAH12', 'PAHs with 1/2 aromatic rings', DefaultPAH12())
kinetics.AddGroupOfSpecies('PAH34', 'PAHs with 3/4 aromatic rings', DefaultPAH34())
kinetics.AddGroupOfSpecies('PAHLP', 'PAHs with more than 4 aromatic rings (molecular and radical)', DefaultPAHLP(kinetics.species))
kinetics.AddGroupOfSpecies('PAHLP-M', 'PAHs with more than 4 aromatic rings (molecular)', DefaultPAHLPM(kinetics.species))
kinetics.AddGroupOfSpecies('PAHLP-R', 'PAHs with more than 4 aromatic rings (radical)', DefaultPAHLPR(kinetics.species))
kinetics.AddGroupOfSpecies('SP', 'BIN sections corresponding to spherical particles (molecular and radical)', DefaultSP(kinetics.species))
kinetics.AddGroupOfSpecies('SP-M', 'BIN sections corresponding to spherical particles (molecular)', DefaultSPM(kinetics.species))
kinetics.AddGroupOfSpecies('SP-R', 'BIN sections corresponding to spherical particles (radical)', DefaultSPR(kinetics.species))
kinetics.AddGroupOfSpecies('AGG', 'BIN sections corresponding to aggregates (molecular and radical)', DefaultAGG(kinetics.species))
kinetics.AddGroupOfSpecies('AGG-M', 'BIN sections corresponding to aggregates (molecular)', DefaultAGGM(kinetics.species))
kinetics.AddGroupOfSpecies('AGG-R', 'BIN sections corresponding to aggregates (radical)', DefaultAGGR(kinetics.species))

# Split reactions
list_of_pahs = kinetics.Group('PAH12')['list'] + kinetics.Group('PAH34')['list'] + kinetics.Group('PAHLP')['list']
reactions_pahs_all, reactions_pahs_pah12, reactions_pahs_pah34, reactions_pahs_pahlp, reactions_pahs_sp, reactions_pahs_agg = kinetics.PartitionSootPrecursorsReactions(list_of_pahs)

reactions_all_sp = kinetics.ReactionsWithMultipleSpecies(kinetics.Group('SP')['list'], ["RP"]*len(kinetics.Group('SP')['list']), 'OR');
reactions_all_agg = kinetics.ReactionsWithMultipleSpecies(kinetics.Group('AGG')['list'], ["RP"]*len(kinetics.Group('AGG')['list']), 'OR');
reactions_all_sp_agg = (np.unique(reactions_all_sp + reactions_all_agg)).tolist();
reactions_minuspahs_sp_agg = list(set(reactions_all_sp_agg) - set(reactions_pahs_sp+reactions_pahs_agg));
reactions_only_sp, reactions_sp_agg, reactions_only_agg = kinetics.SplitSphericalAggregatesReactions(reactions_minuspahs_sp_agg);
reactions_only_gas = list(set(range(kinetics.nr)) - set(reactions_pahs_all) - set(reactions_minuspahs_sp_agg));

# Checking
CheckForCrossingValues(reactions_pahs_pah34, reactions_pahs_pahlp)
CheckForCrossingValues(reactions_only_gas, reactions_pahs_all)
CheckForCrossingValues(reactions_only_gas, reactions_minuspahs_sp_agg)
CheckForCrossingValues(reactions_pahs_all, reactions_minuspahs_sp_agg)

# Print on screen
print('Reactions ALL with SP:', len(reactions_all_sp))
print('Reactions ALL with AGG:', len(reactions_all_agg))
print('Reactions ALL with SP-AGG:', len(reactions_all_sp_agg))
print('Reactions MINUS-PAHS with SP and AGG:', len(reactions_minuspahs_sp_agg))
print('Reactions MINUS-PAHS with SP only:', len(reactions_only_sp))
print('Reactions MINUS-PAHS with SP/AGG:', len(reactions_sp_agg))
print('Reactions MINUS-PAHS with AGG only:', len(reactions_only_agg))

print('Reactions ALL:', kinetics.nr)
print('Reactions GAS ONLY:', len(reactions_only_gas))
print('Reactions PAH PAH12:', len(reactions_pahs_pah12))
print('Reactions PAH PAH34:', len(reactions_pahs_pah34))
print('Reactions PAH PAHLP:', len(reactions_pahs_pahlp))
print('Reactions PAH SP:', len(reactions_pahs_sp))
print('Reactions PAH AGG:', len(reactions_pahs_agg))
print('Reactions SP:', len(reactions_only_sp))
print('Reactions SP/AGG:', len(reactions_sp_agg))
print('Reactions AGG:', len(reactions_only_agg))
print('Sum: ', len(reactions_only_gas)+len(reactions_pahs_pah12)+len(reactions_pahs_pah34)+len(reactions_pahs_pahlp)+len(reactions_pahs_sp)+len(reactions_pahs_agg)+len(reactions_only_sp)+len(reactions_sp_agg)+len(reactions_only_agg))
print('Sum: ', len(reactions_only_gas)+len(reactions_pahs_all)+len(reactions_only_sp)+len(reactions_sp_agg)+len(reactions_only_agg))


# --------------------------------------------------------------------------------------
# Analysis
# -------------------------------------------------------------------------------------- 

# Definitions
iC = 0
iH = 1
rhoSoot = 1800.
Nav_kmol = 6.022e26
Nav_mol = 6.022e23
max_class = 10


# --------------------------------------------------------------------------------------
# 1. Features of main groups
# -------------------------------------------------------------------------------------- 

pah12 = []
pah34 = []
pahlp = []
bin_min_particles = []

list_bin_min_particles = ['BIN5AJ', 'BIN5BJ', 'BIN5CJ']
for species in ( kinetics.Group('PAH12')['list'] + kinetics.Group('PAH34')['list'] + kinetics.Group('PAHLP')['list'] + list_bin_min_particles):
    
    j = kinetics.species.index(species)
    MW = kinetics.mws[j]
    nC = kinetics.atomic[j][iC]
    nH = kinetics.atomic[j][iH]
    d = Diameter(MW, rhoSoot)
    V = Volume(d)
    m = Mass(MW)
    
    element = { 'name': species, 'i': j, 'nC': nC, 'nH': nH, 'MW': MW, 'd': d, 'V': V, 'm': m }
    if species in kinetics.Group('PAH12')['list']:
        pah12.append( element )
    elif species in kinetics.Group('PAH34')['list']:
        pah34.append( element )
    elif species in kinetics.Group('PAHLP')['list']:
        pahlp.append( element )
    elif species in list_bin_min_particles:
        bin_min_particles.append(element)
        
pah12pd = pd.DataFrame(pah12)
pah34pd = pd.DataFrame(pah34)
pahlppd = pd.DataFrame(pahlp)
bin_min_particles_pd = pd.DataFrame(bin_min_particles)


#df_styled = pah12pd.style.background_gradient()
#dfi.export(df_styled, 'pah12pd.png')
print(pah12pd)
print(pah34pd)
print(pahlppd)
print(bin_min_particles_pd)


# --------------------------------------------------------------------------------------
# 2. Reaction classes: fitting of nucleation reactions
# -------------------------------------------------------------------------------------- 

classes = kinetics.OrganizeReactionsInClasses(range(kinetics.nr))
print(classes)

reactions_inception = []
reactions_unclassified = []

reactions_pahlp_pahlp = []
reactions_pah34_pahlp = []
reactions_pah12_pahlp = []

reactions_pah34_pah34 = []
reactions_pah12_pah12 = []
reactions_pah12_pah34 = []

count = 0
for dimer in bin_min_particles:
    
    dimer_j = dimer['i']
    dimer_MW = dimer['MW']
    dimer_nC = dimer['nC']
    dimer_nH = dimer['nH']

    for i in kinetics.ReactionsWithSpecies(dimer['name'],'P'):
        for k in range(len(classes)):
            if i in classes[k]:
                if (k<=max_class):

                    count = count+1

                    ir = sparse.find(kinetics.nur.getrow(i))[1]
                    nur = sparse.find(kinetics.nur.getrow(i))[2]
                    if (len(ir) == 1):
                        if (nur[0] == 2):
                            ir = [ ir[0], ir[0] ]
                            nur = [ 1., 1.]
                        else:
                            print(kinetics.reaction_lines[i])
                            sys.exit("Unexpected stoichiometry")

                    # Features of species
                    PAH1 = kinetics.species[ir[0]]
                    PAH2 = kinetics.species[ir[1]]
                    MW1 = kinetics.mws[ir[0]]
                    MW2 = kinetics.mws[ir[1]]
                    nC1 = kinetics.atomic[ir[0]][iC]
                    nC2 = kinetics.atomic[ir[1]][iC]
                    nH1 = kinetics.atomic[ir[0]][iH]
                    nH2 = kinetics.atomic[ir[1]][iH]
                    
                    ip = sparse.find(kinetics.nup.getrow(i))[1]
                    nup = sparse.find(kinetics.nup.getrow(i))[2]
                    
                    nureal = nup[ip.tolist().index(dimer_j)] 
                    alpha = (nC1+nC2)/dimer_nC
                    beta = (nH1+nH2)/dimer_nH
                    
                    A = kinetics.A[i]
                    Beta = kinetics.Beta[i]
                    E_over_R = kinetics.E_over_R[i]
                    
                    Texp = [400., 700., 1000., 1300., 1600., 1900., 2200., 2500.]
                    yexp = np.log(A) + Beta*np.log(Texp) -E_over_R/Texp - 0.5*np.log(Texp)
                    
                    x1 = 1./np.asarray(Texp)
                    
                    M1 = np.ones((len(Texp),2))
                    M1[:,1] = x1
                    M1TM1 = np.matmul(np.transpose(M1),M1)
                    b1 = np.matmul(np.transpose(M1),yexp)
                    params1 = np.linalg.solve(M1TM1, b1)
                    A1 = math.exp(params1[0])
                    Tatt1 = -params1[1]
                    
                    M2 = np.ones((len(Texp),1))
                    M2TM2 = np.matmul(np.transpose(M2),M2)
                    b2 = np.matmul(np.transpose(M2),yexp)
                    params2 = np.linalg.solve(M2TM2, b2)
                    A2 = math.exp(params2[0])
                    
                    reac = { 'i': i, 'PAH': [PAH1, PAH2], 'BIN': dimer['name'], 'nu': nureal, 'A': A, 'Beta': Beta, 'Tatt': E_over_R, 'A1': A1, 'Tatt1': Tatt1, 'A2': A2, 'epsilon': {} }
                    reac_switch = { 'i': i, 'PAH': [PAH2, PAH1], 'BIN': dimer['name'], 'nu': nureal, 'A': A, 'Beta': Beta, 'Tatt': E_over_R, 'A1': A1, 'Tatt1': Tatt1, 'A2': A2, 'epsilon': {} }

                    # Generic reaction
                    reactions_inception.append(reac)

                    # PAHLP + PAHLP
                    if ( PAH1 in kinetics.Group('PAHLP')['list'] and PAH2 in kinetics.Group('PAHLP')['list']):
                        reactions_pahlp_pahlp.append(reac)

                    # PAH34 + PAH34
                    elif ( kinetics.species[ir[0]] in kinetics.Group('PAH34')['list'] and kinetics.species[ir[1]] in kinetics.Group('PAH34')['list']):
                        reactions_pah34_pah34.append(reac)

                    # PAH12 + PAH12
                    elif ( kinetics.species[ir[0]] in kinetics.Group('PAH12')['list'] and kinetics.species[ir[1]] in kinetics.Group('PAH12')['list']):
                        reactions_pah12_pah12.append(reac)

                    # PAH12 + PAHLP
                    elif ( kinetics.species[ir[0]] in kinetics.Group('PAH12')['list'] and kinetics.species[ir[1]] in kinetics.Group('PAHLP')['list']):
                        reactions_pah12_pahlp.append(reac)
                    elif ( kinetics.species[ir[0]] in kinetics.Group('PAHLP')['list'] and kinetics.species[ir[1]] in kinetics.Group('PAH12')['list']):
                        reactions_pah12_pahlp.append(reac_switch)

                    # PAH12 + PAH34
                    elif ( kinetics.species[ir[0]] in kinetics.Group('PAH12')['list'] and kinetics.species[ir[1]] in kinetics.Group('PAH34')['list']):
                        reactions_pah12_pah34.append(reac)
                    elif ( kinetics.species[ir[0]] in kinetics.Group('PAH34')['list'] and kinetics.species[ir[1]] in kinetics.Group('PAH12')['list']):
                        reactions_pah12_pah34.append(reac_switch)

                    # PAH34 + PAHLP
                    elif ( kinetics.species[ir[0]] in kinetics.Group('PAH34')['list'] and kinetics.species[ir[1]] in kinetics.Group('PAHLP')['list']):
                        reactions_pah34_pahlp.append(reac)
                    elif ( kinetics.species[ir[0]] in kinetics.Group('PAHLP')['list'] and kinetics.species[ir[1]] in kinetics.Group('PAH34')['list']):
                        reactions_pah34_pahlp.append(reac_switch)

                    else:
                        reactions_unclassified.append(reac)

sum = len(reactions_pah12_pah12) + len(reactions_pah12_pah34) + len(reactions_pah12_pahlp) +       len(reactions_pah34_pah34) + len(reactions_pah34_pahlp) + len(reactions_pahlp_pahlp) +       len(reactions_unclassified)

print("Total inception reactions: ", len(reactions_inception))
print(" - PAH12/PAH12:            ", len(reactions_pah12_pah12))
print(" - PAH12/PAH34:            ", len(reactions_pah12_pah34))
print(" - PAH12/PAHLP:            ", len(reactions_pah12_pahlp))
print(" - PAH34/PAH34:            ", len(reactions_pah34_pah34))
print(" - PAH34/PAHLP:            ", len(reactions_pah34_pahlp))
print(" - PAHLP/PAHLP:            ", len(reactions_pahlp_pahlp))
print(" - gas-phase:              ", len(reactions_unclassified))

reactions_pahlp_pahlp_pd = pd.DataFrame(reactions_pahlp_pahlp)
print(reactions_pahlp_pahlp_pd)


# --------------------------------------------------------------------------------------
# 3. Analysis of simulation data
# -------------------------------------------------------------------------------------- 

from LaminarFlame1D import LaminarFlame1D

flame = LaminarFlame1D("C:\\Users\\acuoci\\Desktop\\Output.xml", kinetics)
T = flame.T

for index in range(len(reactions_pahlp_pahlp)):
    
    nu = reactions_pahlp_pahlp[index]['nu']
    A = reactions_pahlp_pahlp[index]['A']
    Beta = reactions_pahlp_pahlp[index]['Beta']
    Tatt = reactions_pahlp_pahlp[index]['Tatt']
    A1 = reactions_pahlp_pahlp[index]['A1']
    Tatt1 = reactions_pahlp_pahlp[index]['Tatt1']
    A2 = reactions_pahlp_pahlp[index]['A2']
    kappaArrhenius = A*pow(T,Beta)*np.exp(-Tatt/T)
    kappaArrhenius1 = A1*pow(T,0.5)*np.exp(-Tatt1/T)
    kappaArrhenius2 = A2*pow(T,0.5)
    PAH1 = reactions_pahlp_pahlp[index]['PAH'][0]
    PAH2 = reactions_pahlp_pahlp[index]['PAH'][1]
    Beta_eps1 = BetaKernel(kinetics, rhoSoot, 1, T, PAH1,PAH2)

    #plt.plot(flame.csi, kappaArrhenius)
    #plt.plot(flame.csi, kappaArrhenius1)
    #plt.plot(flame.csi, kappaArrhenius2)
    #plt.plot(flame.csi, Beta_eps1*Nav_kmol)
    #plt.legend(['kappa', 'kappa1', 'kappa2', 'Beta'])

    
    epsilon =  nu*kappaArrhenius/(Beta_eps1*Nav_kmol)
    epsilon1 = nu*kappaArrhenius1/(Beta_eps1*Nav_kmol)
    epsilon2 = nu*kappaArrhenius2/(Beta_eps1*Nav_kmol)
    
    single_element = { 'e': epsilon, 'e1': epsilon1, 'e2': epsilon2, 'em': epsilon.mean(), 'e1m': epsilon1.mean(), 'e2m': epsilon2.mean() }
    
    reactions_pahlp_pahlp[index]['epsilon'] = single_element
    
print(reactions_pahlp_pahlp)
reactions_pahlp_pahlp_pd = pd.DataFrame(reactions_pahlp_pahlp)

print("List of reactions with Beta > -1.5")
for i in range(len(reactions_pahlp_pahlp)):
    if (reactions_pahlp_pahlp[i]['Beta'] > -1.5):
        print(reactions_pahlp_pahlp[i])


# Plot specific reaction
index = 84
plt.plot(flame.csi, reactions_pahlp_pahlp[index]['epsilon']['e'])
plt.plot(flame.csi, reactions_pahlp_pahlp[index]['epsilon']['e1'])
plt.plot(flame.csi, reactions_pahlp_pahlp[index]['epsilon']['e2'])


i = reactions_pahlp_pahlp[index]['i']
j = kinetics.species.index(reactions_pahlp_pahlp[index]['PAH'][0])

r_sim = flame.rr[:,i] #kmol/m3/s

# Molar fraction
PAH_X = flame.X[:,j]   

# Concentration (kmol/m3)
Ctot = flame.rho / flame.mw 
PAH_C = PAH_X*Ctot

r = kappaArrhenius*PAH_C*PAH_C
r1 = kappaArrhenius1*PAH_C*PAH_C
r2 = kappaArrhenius2*PAH_C*PAH_C
plt.plot(flame.csi, r_sim)
plt.plot(flame.csi, r, '-')
plt.plot(flame.csi, r1, '--')
plt.plot(flame.csi, r2, '.')


# --------------------------------------------------------------------------------------
# 3. Analysis of overall data
# -------------------------------------------------------------------------------------- 

#plt.plot(flame.csi, flame.rr[:,0])
reactions = []

R_pah12_pahlp = [0.]*flame.npts
for i in range(len(reactions_pah12_pahlp)):
    j = reactions_pah12_pahlp[i]["i"]
    nu = reactions_pah12_pahlp[i]["nu"]
    R_pah12_pahlp = R_pah12_pahlp + nu*flame.rr[:,j]
    
R_pah34_pahlp = [0.]*flame.npts
for i in range(len(reactions_pah34_pahlp)):
    j = reactions_pah34_pahlp[i]["i"]
    nu = reactions_pah34_pahlp[i]["nu"]
    R_pah34_pahlp = R_pah34_pahlp + nu*flame.rr[:,j]
    
R_pahlp_pahlp = [0.]*flame.npts
for i in range(len(reactions_pahlp_pahlp)):
    j = reactions_pahlp_pahlp[i]["i"]
    nu = reactions_pahlp_pahlp[i]["nu"]
    R_pahlp_pahlp = R_pahlp_pahlp + nu*flame.rr[:,j]
    
R_unclassified = [0.]*flame.npts
for i in range(len(reactions_unclassified)):
    j = reactions_unclassified[i]["i"]
    nu = reactions_unclassified[i]["nu"]
    R_unclassified = R_unclassified + nu*flame.rr[:,j]
    
R_total = R_pah12_pahlp + R_pah34_pahlp + R_pahlp_pahlp + R_unclassified
    
plt.plot(flame.csi, R_pah12_pahlp)    
plt.plot(flame.csi, R_pah34_pahlp) 
plt.plot(flame.csi, R_pahlp_pahlp) 
plt.plot(flame.csi, R_unclassified) 


# In[27]:


Omega_BIN5 = flame.Omega[:,kinetics.species.index('BIN5AJ')] + \
             flame.Omega[:,kinetics.species.index('BIN5BJ')] + \
             flame.Omega[:,kinetics.species.index('BIN5CJ')]
                        
#R_BIN5 = Omega_BIN5/320.
#plt.plot(flame.csi, R_BIN5) 
#plt.plot(flame.csi, R_total)


example = sparse.find(kinetics.nup.getrow(0))
print(example)


def CalculateRdimer(dimer, flame, kinetics, classes):
    dimer_j = kinetics.species.index(dimer)
    R_dimer = [0.]*flame.npts
    for i in kinetics.ReactionsWithSpecies(dimer,'RP'):
        for k in range(len(classes)):
            if i in classes[k]:
                if (k<=10):

                    ir = sparse.find(kinetics.nur.getrow(i))[1] 
                    if (dimer_j in ir):
                        j = ir.tolist().index(dimer_j)
                        nur = sparse.find(kinetics.nur.getrow(i))[2]
                        R_dimer = R_dimer - nur[j]*flame.rr[:,i]

                    ip = sparse.find(kinetics.nup.getrow(i))[1] 
                    if (dimer_j in ip):
                        j = ip.tolist().index(dimer_j)
                        nup = sparse.find(kinetics.nup.getrow(i))[2]
                        R_dimer = R_dimer + nup[j]*flame.rr[:,i]
                    
    return R_dimer

R_BIN5AJ = CalculateRdimer('BIN5AJ', flame, kinetics, classes)
R_BIN5BJ = CalculateRdimer('BIN5BJ', flame, kinetics, classes)
R_BIN5CJ = CalculateRdimer('BIN5CJ', flame, kinetics, classes)

Omega_BIN5AJ = R_BIN5AJ*kinetics.mws[kinetics.species.index('BIN5AJ')]
Omega_BIN5BJ = R_BIN5BJ*kinetics.mws[kinetics.species.index('BIN5BJ')]
Omega_BIN5CJ = R_BIN5CJ*kinetics.mws[kinetics.species.index('BIN5CJ')]


plt.plot(flame.csi, R_BIN5AJ)
plt.plot(flame.csi, R_BIN5BJ)
plt.plot(flame.csi, R_BIN5CJ)

plt.plot(flame.csi, R_total)
plt.plot(flame.csi, R_BIN5AJ+R_BIN5BJ+R_BIN5CJ)
