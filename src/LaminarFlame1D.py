'''
MODULE: LaminarFlame1D
@Authors:
    Alberto Cuoci [1]
    [1]: CRECK Modeling Lab, Department of Chemistry, Materials, and Chemical Engineering, Politecnico di Milano
@Contacts:
    alberto.cuoci@polimi.it
@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: alberto.cuoci@polimi.it
'''

import xml.etree.ElementTree as ET
import numpy as np
import re
import pandas as pd
import sys
from scipy import interpolate

class LaminarFlame1D:

  '''
  Description of the LaminarFlame1D class
  TODO
  '''
  
  def __init__(self, xml_file_name, kin):
    
    tree = ET.parse(xml_file_name)
    root = tree.getroot()
    
    # Flame type
    flameType = ((root.find('Type')).text).strip()
    if (flameType != "Flamelet" and flameType != "Flame1D"):
        print(flameType)
        sys.exit("Unknown flame type")
    
    
    # Check simulation/kinetics consistency
    dummy = root.find('mass-fractions')
    dummy = (dummy.text).split()
    list_names = []
    for i in range(int(dummy[0])):
        list_names.append(dummy[1+i*3])
    if (len(list_names) != kin.ns):
        sys.exit("The kinetic mechanism is not consistent with the simulation")
    for i in range(kin.ns):
        if (list_names[i] != kin.species[i]):
            sys.exit("The kinetic mechanism is not consistent with the simulation")
    
    
    # Recover additional variables
    dummy = root.find('additional')
    dummy = (dummy.text).split()
    for i in range(int(dummy[0])):
        variable = dummy[1+i*3]
        index = int(dummy[3+i*3]) - 2
        if (variable == 'axial-coordinate'):    index_x = index
        if (variable == 'temperature'):         index_T = index
        if (variable == 'pressure'):            index_P = index
        if (variable == 'mol-weight'):          index_mw = index
        if (variable == 'density'):             index_rho = index
        if (variable == 'heat-release'):        index_Q = index
        if (variable == 'axial-velocity'):      index_u = index
        if (variable == 'mass-flow-rate'):      index_m = index
        if (variable == 'csi'):                 index_csi = index
        if (variable == 'mixture-fraction'):    index_csi = index
        if (variable == 'scalar-dissipation-rate'):     index_chi = index
        if (variable == 'enthalpy'):                    index_h = index
        if (variable == 'specific-heat'):               index_cp = index
        if (variable == 'thermal-conductivity'):        index_kappa = index

    
    # Read profiles
    profiles_size = root.find('profiles-size')
    profiles_size = (profiles_size.text).split()
    npts = int(profiles_size[0])
    nc = int(profiles_size[1])

    profiles = root.find('profiles')
    profiles = (profiles.text).split()
    profiles = np.reshape(profiles, (npts,nc))
    profiles = np.float32(profiles)
    
    # Extract relevant profiles
    T = profiles[:,index_T]
    P = profiles[:,index_P]
    mw = profiles[:,index_mw]
    rho = profiles[:,index_rho]
    Q = profiles[:,index_Q]
    
    if (flameType == 'Flame1D'):
        x = profiles[:,index_x]
        u = profiles[:,index_u]
        m = profiles[:,index_m]
        csi = profiles[:,index_csi]
    
    if (flameType == 'Flamelet'):
        chi = profiles[:,index_chi]
        h = profiles[:,index_h]
        csi = [1]*npts - profiles[:,index_csi]
        cp = profiles[:,index_cp]
        kappa = profiles[:,index_kappa]
        
    # Composition
    Y = profiles[:,-kin.ns:]
    X = Y*mw.reshape(-1,1)/np.transpose(kin.mws.reshape(-1,1))
    
    # Formation rates (mass units, kg/m3/s)
    Omega = root.find('formation-rates')
    Omega = (Omega.text).split()
    Omega = np.reshape(Omega, (npts,kin.ns))
    Omega = np.float32(Omega)
    
    # Reaction rates (molar units, kmol/m3/s)
    rr = root.find('reaction-rates')
    if rr is None:
        print('no reaction-rates available in the xml file')
    else:
        rr = (rr.text).split()
        rr = np.reshape(rr, (npts,kin.nr))
        rr = np.float32(rr)
    
    
    # Reconstruct mixture fractions
    X_atoms = X.dot(kin.atomic)

    ZC = X_atoms[:,kin.iC]*kin.mwe[kin.iC]/mw
    ZH = X_atoms[:,kin.iH]*kin.mwe[kin.iH]/mw
    ZO = X_atoms[:,kin.iO]*kin.mwe[kin.iO]/mw
    WC = kin.mwe[kin.iC]
    WH = kin.mwe[kin.iH]
    WO = kin.mwe[kin.iO]

    Zstar = 2.*ZC/WC + 0.50*ZH/WH - ZO/WO
    ZstarFu = Zstar[0]
    ZstarOx = Zstar[-1]

    Z = (Zstar-ZstarOx)/(ZstarFu-ZstarOx)

    
    # Reconstruct soot relevant variables: mass fractions
    agg_Y = np.sum(Y[:,kin.Group('AGG')['indices']], axis=1)
    sp_Y = np.sum(Y[:,kin.Group('SP')['indices']], axis=1)
    pahlp_Y = np.sum(Y[:,kin.Group('PAHLP')['indices']], axis=1)
    pah34_Y = np.sum(Y[:,kin.Group('PAH34')['indices']], axis=1)
    pah12_Y = np.sum(Y[:,kin.Group('PAH12')['indices']], axis=1)
    soot_Y = sp_Y + agg_Y
    
    # Reconstruct soot relevant variables: mole fractions
    agg_X = np.sum(X[:,kin.Group('AGG')['indices']], axis=1)
    sp_X = np.sum(X[:,kin.Group('SP')['indices']], axis=1)
    pahlp_X = np.sum(X[:,kin.Group('PAHLP')['indices']], axis=1)
    pah34_X = np.sum(X[:,kin.Group('PAH34')['indices']], axis=1)
    pah12_X = np.sum(X[:,kin.Group('PAH12')['indices']], axis=1)    
    soot_X = sp_X + agg_X

    # Reconstruct soot relevant variables: volume fraction
    agg_fv = agg_Y*rho/kin.rho_soot
    sp_fv = sp_Y*rho/kin.rho_soot
    soot_fv = sp_fv + agg_fv
    
    
    # Relevant (possible) variables for progress variable construction
    Y_over_MW_pah12 = np.sum(Y[:,kin.Group('PAH12')['indices']]/kin.Group('PAH12')['mws'], axis=1)
    Y_over_MW_pah34 = np.sum(Y[:,kin.Group('PAH34')['indices']]/kin.Group('PAH34')['mws'], axis=1)
    Y_over_MW_pahlp = np.sum(Y[:,kin.Group('PAHLP')['indices']]/kin.Group('PAHLP')['mws'], axis=1)
    
    Omega_over_MW_pah12 = np.sum(Omega[:,kin.Group('PAH12')['indices']]/kin.Group('PAH12')['mws'], axis=1)
    Omega_over_MW_pah34 = np.sum(Omega[:,kin.Group('PAH34')['indices']]/kin.Group('PAH34')['mws'], axis=1)
    Omega_over_MW_pahlp = np.sum(Omega[:,kin.Group('PAHLP')['indices']]/kin.Group('PAHLP')['mws'], axis=1)
    
    # Assign internal members
    
    self.npts = npts
    
    self.T = T
    self.P = P
    self.mw = mw
    self.rho = rho
    self.Q = Q
    
    if (flameType == 'Flame1D'):
        self.x = x
        self.u = u
        self.m = m
        self.csi = csi
    
    if (flameType == 'Flamelet'):
        self.chi = chi
        self.h = h
        self.cp = cp
        self.kappa = kappa
        self.alpha = kappa/rho/cp
        self.csi = csi

    self.Z = Z
    self.Y = Y
    self.X = X
    self.Omega = Omega
    self.rr = rr
    
    self.agg_Y = agg_Y
    self.agg_X = agg_X
    self.agg_fv = agg_fv
    
    self.sp_Y = sp_Y
    self.sp_X = sp_X
    self.sp_fv = sp_fv
    
    self.pahlp_Y = pahlp_Y
    self.pahlp_X = pahlp_X   
    
    self.pah34_Y = pah34_Y
    self.pah34_X = pah34_X   

    self.pah12_Y = pah12_Y
    self.pah12_X = pah12_X
    
    self.Y_over_MW_pah12 = Y_over_MW_pah12
    self.Y_over_MW_pah34 = Y_over_MW_pah34
    self.Y_over_MW_pahlp = Y_over_MW_pahlp
    
    self.Omega_over_MW_pah12 = Omega_over_MW_pah12
    self.Omega_over_MW_pah34 = Omega_over_MW_pah34
    self.Omega_over_MW_pahlp = Omega_over_MW_pahlp
    
    
  def minmax_progress_variable(self):

    return [ min(self.y), max(self.y) ]
  
    
  def progress_variable(self, kin, alpha):

    npts = len(self.T)
    self.y = [0]*npts
    self.Omegay = [0]*npts
    
    for i in range(len(alpha)):
        
        if (alpha[i][0] == 'PAH12'): 
            self.y = self.y + alpha[i][1] * self.Y_over_MW_pah12
            self.Omegay = self.Omegay + alpha[i][1] * self.Y_over_MW_pah12
        
        elif (alpha[i][0] == 'PAH34'): 
            self.y = self.y + alpha[i][1] * self.Y_over_MW_pah34
            self.Omegay = self.Omegay + alpha[i][1] * self.Y_over_MW_pah34
        
        elif (alpha[i][0] == 'PAHLP'): 
            self.y = self.y + alpha[i][1] * self.Y_over_MW_pahlp
            self.Omegay = self.Omegay + alpha[i][1] * self.Y_over_MW_pahlp
        
        else:
            index = kin.species.index(alpha[i][0])
            self.y = self.y + alpha[i][1] * self.Y[:,index] /kin.mws[index]
            self.Omegay = self.Omegay + alpha[i][1] * self.Omega[:,index] /kin.mws[index]
    
    """
    self.y = alpha_H2O * self.Y[:,kin.iH2O] /kin.mws[kin.iH2O] + \
             alpha_CO2 * self.Y[:,kin.iCO2] /kin.mws[kin.iCO2] + \
             alpha_H2  * self.Y[:,kin.iH2]  /kin.mws[kin.iH2] + \
             alpha_CO  * self.Y[:,kin.iCO]  /kin.mws[kin.iCO] + \
             alpha_O2  * self.Y[:,kin.iO2]  /kin.mws[kin.iO2] + \
             alpha_pah12 * self.Y_over_MW_pah12 + \
             alpha_pah34 * self.Y_over_MW_pah34 + \
             alpha_pahlp * self.Y_over_MW_pahlp            
             
    self.Omegay = alpha_H2O * self.Omega[:,kin.iH2O] /kin.mws[kin.iH2O] + \
                  alpha_CO2 * self.Omega[:,kin.iCO2] /kin.mws[kin.iCO2] + \
                  alpha_H2  * self.Omega[:,kin.iH2]  /kin.mws[kin.iH2] + \
                  alpha_CO  * self.Omega[:,kin.iCO]  /kin.mws[kin.iCO] + \
                  alpha_O2  * self.Omega[:,kin.iO2]  /kin.mws[kin.iO2] + \
                  alpha_pah12 * self.Omega_over_MW_pah12 + \
                  alpha_pah34 * self.Omega_over_MW_pah34 + \
                  alpha_pahlp * self.Omega_over_MW_pahlp                   
    """
    
  def create_grid(self, kin, npts, min_val, max_val ):
    
    npts_Z = npts[0]
    npts_y = npts[1]
    min_Z = min_val[0]
    max_Z = max_val[0]
    min_y = min_val[1]
    max_y = max_val[1]
 
    # Grids
    self.Z_grid = []
    for i in range(npts_Z):
        self.Z_grid.append( min_Z + (max_Z-min_Z)/(npts_Z-1)*i )
        
    self.y_grid = []
    for i in range(npts_y):
        self.y_grid.append( min_y + (max_y-min_y)/(npts_y-1)*i )        
    
    # Main variables
    f = interpolate.interp1d(self.Z, self.T)
    self.T_int_Z = f(self.Z_grid)
    f = interpolate.interp1d(self.Z, self.rho)
    self.rho_int_Z = f(self.Z_grid)
    f = interpolate.interp1d(self.Z, self.cp)
    self.cp_int_Z = f(self.Z_grid)
    f = interpolate.interp1d(self.Z, self.mw)
    self.mw_int_Z = f(self.Z_grid)
    f = interpolate.interp1d(self.Z, self.kappa)
    self.kappa_int_Z = f(self.Z_grid)
    f = interpolate.interp1d(self.Z, self.alpha)
    self.alpha_int_Z = f(self.Z_grid)
    
    # Mass fractions
    self.Y_int_Z = np.empty( (npts_Z,kin.ns) )
    for i in range(kin.ns):
        f = interpolate.interp1d(self.Z, self.Y[:,i])
        dummy = f(self.Z_grid)
        self.Y_int_Z[:,i] = dummy
    
    # Progress variable
    f = interpolate.interp1d(self.Z, self.y)
    self.y_int_Z = f(self.Z_grid) 
    f = interpolate.interp1d(self.Z, self.Omegay)
    self.Omegay_int_Z = f(self.Z_grid)
    
    # PAHs
    f = interpolate.interp1d(self.Z, self.pahlp_Y)
    self.pahlp_int_Z = f(self.Z_grid)
    f = interpolate.interp1d(self.Z, self.pah34_Y)
    self.pah34_int_Z = f(self.Z_grid) 
    f = interpolate.interp1d(self.Z, self.pah12_Y)
    self.pah12_int_Z = f(self.Z_grid)
    