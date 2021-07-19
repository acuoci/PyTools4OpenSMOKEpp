'''
MODULE: OpenSMOKEppXMLFile
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

class OpenSMOKEppXMLFile:

  '''
  Description of the OpenSMOKEppXMLFile class
  TODO
  '''
    
  def __init__(self, xml_file_name, kin):
    
    tree = ET.parse(xml_file_name)
    root = tree.getroot()
    
    # System type
    systemType = ((root.find('Type')).text).strip()
    if (systemType != "HomogeneousReactor" and systemType != "Flamelet" and systemType != "Flame1D" and systemType != "Flame2D"):
        print(systemType)
        sys.exit("Unknown system type")
    
    
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
    n_additional = int(dummy[0])
    for i in range(n_additional):
        variable = dummy[1+i*3]
        index = int(dummy[3+i*3]) - 2
        if (variable == 'temperature'):         index_T = index
        if (variable == 'pressure'):            index_P = index
        if (variable == 'mol-weight'):          index_mw = index
        if (variable == 'density'):             index_rho = index
        if (variable == 'heat-release'):        index_Q = index
        if (variable == 'csi'):                 index_csi = index
        if (variable == 'mixture-fraction'):    index_csi = index
        if (variable == 'axial-coordinate'):    index_csi = index

    
    # Read profiles
    profiles_size = root.find('profiles-size')
    profiles_size = (profiles_size.text).split()
    npts = int(profiles_size[0])
    nc = n_additional + kin.ns

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
    
    if (systemType == 'Flame1D'):
        csi = profiles[:,index_csi]
    elif (systemType == 'Flamelet'):
        csi = [1]*npts - profiles[:,index_csi]
    else:
        csi = [0]*npts
        
    # Composition
    Y = profiles[:,-kin.ns:]
    X = Y*mw.reshape(-1,1)/np.transpose(kin.mws.reshape(-1,1))
    
    
    # Assign internal members
    
    self.npts = npts
    self.ns = kin.ns
    
    self.T = T
    self.P = P
    self.mw = mw
    self.rho = rho
    self.Q = Q
    self.csi = csi
    
    if (systemType == 'Flame1D'):
        self.csi = csi

    if (systemType == 'Flame2D'):
        self.csi = csi
    
    if (systemType == 'Flamelet'):
        self.csi = csi

    self.Y = Y
    self.X = X


  def WriteThermophysicalState(self, f):
  
    for i in range(self.npts):
        f.write('1 ');
        f.write( "{:.6E} ".format(self.csi[i]) )
        f.write( "{:.6E} ".format(self.T[i]) )
        for j in range(self.ns):
            f.write( "{:.6E} ".format(self.Y[i][j]) )
        f.write('\n')

  def WriteThermophysicalStateAsCVS(self, additional, f):
  
    for i in range(self.npts):
        for add in additional:
            f.write( add+"," )
        f.write( "{:.6E},".format(self.csi[i]) )
        f.write( "{:.6E}".format(self.T[i]) )
        for j in range(self.ns):
            f.write( ",{:.6E}".format(self.Y[i][j]) )
        f.write('\n')
        
        
