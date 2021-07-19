'''
MODULE: KineticMechanism
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
from scipy import sparse
import sys

class KineticMechanism:

  '''
  Description of the KineticMechanism class
  TODO
  '''
    
  rho_soot = 1500.
  
  def __init__(self, xml_file_name):
    
    tree = ET.parse(xml_file_name)
    root = tree.getroot()
    
    # List of elements
    elements = root.find('NamesOfElements')
    elements = (elements.text).split()
    ne = len(elements)

    # List of species
    species = root.find('NamesOfSpecies')
    species = (species.text).split()
    ns = len(species)

    # Elemental composition
    atomic = root.find('AtomicComposition')
    atomic = (atomic.text).split()
    atomic = np.reshape(atomic, (ns,ne))
    atomic = np.float32(atomic)

    # Indices of relevant elements
    iC = elements.index('C')
    iH = elements.index('H')
    iO = elements.index('O')
    iN = elements.index('N')
    
    # Elements molecular weights
    mwe = []
    for elem in elements:
        if (elem == 'C'): mwe.append(12.010999679565430)
        if (elem == 'H'): mwe.append(1.008000016212463)
        if (elem == 'O'): mwe.append(15.998999595642090)
        if (elem == 'N'): mwe.append(14.0069999694824)
        if (elem == 'HE'): mwe.append(4.002999782562256)
        if (elem == 'AR'): mwe.append(39.948001861572270)
    mwe = np.array(mwe)
    
    # Species molecular weights
    mws = atomic.dot(mwe)
            
    # Soot classes (if they exists)
    reaction_class_name = []
    reaction_class_size = []
    reaction_class_indices = []
    
    kinetics = root.find('Kinetics')
    polimi_soot_classes = kinetics.find('PolimiSootClasses')
    if (polimi_soot_classes != None):
        
        for child in polimi_soot_classes:
            if (child.tag == 'ClassReactions'):
                reaction_class_name.append(child.attrib['name'])
                reaction_class_size.append(int(child.attrib['n']))
                dummy = (child.text).split()
                reaction_class_indices.append( list(map(int,dummy)) )
        
        self.reaction_class_name = reaction_class_name
        self.reaction_class_size = reaction_class_size
        self.reaction_class_indices = reaction_class_indices
        
    else:
    
        self.reaction_class_name = []
        self.reaction_class_size = []
        self.reaction_class_indices = []
        
    # Kinetic parameters
    kinetic_parameters = kinetics.find('KineticParameters')
    direct = kinetic_parameters.find('Direct')
    
    lnA = direct.find('lnA')
    lnA = (lnA.text).split()
    lnA = np.float64(lnA[1:])
    A = np.exp(lnA)
    
    Beta = direct.find('Beta')
    Beta = (Beta.text).split()
    Beta = np.float64(Beta[1:])
    
    E_over_R = direct.find('E_over_R')
    E_over_R = (E_over_R.text).split()
    E_over_R = np.float64(E_over_R[1:])

    
    # Assign internal members
    
    self.elements = elements
    self.atomic = atomic
    self.species = species
    self.ne = ne
    self.ns = ns
        
    self.iC = iC
    self.iH = iH
    self.iO = iO
    self.iN = iN
    
    self.mwe = mwe
    self.mws = mws
    
    self.groups = []

    self.A = A
    self.Beta = Beta
    self.E_over_R = E_over_R
    
    self.reactions = []
    
    
  def AddGroupOfSpecies(self, name, description, species_list):
  
    indices = []
    mws = []
    for s in species_list:
        if (s in self.species):
            indices.append(self.species.index(s))
            mws.append(self.mws[self.species.index(s)])

    # Define group
    group_definition = { 'name': name, 'description': description, 'list': species_list, 'indices': indices, 'mws': mws }

    # Append group
    self.groups.append( group_definition );
   
   
  def Group(self, name):
  
    for i in range(len(self.groups)):
        if (name == self.groups[i]['name']): return self.groups[i]
    

    
  def ReadKinetics(self, xml_file_name):
  
    tree = ET.parse(xml_file_name)
    root = tree.getroot()
    
    # Reaction lines
    reaction_lines = []
    sub_root = root.find('reaction-chemkin')
    for child in sub_root:
        reaction_lines.append((child.text)[:-1])
    self.reaction_lines = reaction_lines
    
    sub_root = root.find('stoichiometric-matrix')
    reactants_reaction = []
    reactants_species = []
    reactants_nu = []
    products_reaction = []
    products_species = []
    products_nu = []
    
    # List of reactions in which each species appears
    species_in_reaction_as_reactant = []
    species_in_reaction_as_product = []
    species_in_reaction_as_reactant_or_product = []
    for i in range(self.ns):
        species_in_reaction_as_reactant.append([])
        species_in_reaction_as_product.append([])
        species_in_reaction_as_reactant_or_product.append([])
    
    # Analyze
    for child in sub_root:
        j = int(child.attrib['i'])
        nr = int(child.attrib['nr'])
        np = int(child.attrib['np'])
        coefficients = (child.text).split()
        for i in range(nr):
            reactants_reaction.append(j)
            reactants_species.append(int(coefficients[2*i]))
            reactants_nu.append(float(coefficients[2*i+1]))
            species_in_reaction_as_reactant[int(coefficients[2*i])].append(j)
            species_in_reaction_as_reactant_or_product[int(coefficients[2*i])].append(j)
        for i in range(np):
            products_reaction.append(j)
            products_species.append(int(coefficients[2*i + 2*nr]))
            products_nu.append(float(coefficients[2*i+1 + 2*nr]))
            species_in_reaction_as_product[int(coefficients[2*i + 2*nr])].append(j)
            species_in_reaction_as_reactant_or_product[int(coefficients[2*i + 2*nr])].append(j)
    
    self.nr = len(reaction_lines)
    self.nur = sparse.coo_matrix((reactants_nu,(reactants_reaction,reactants_species)),shape=(self.nr,self.ns))
    self.nup = sparse.coo_matrix((products_nu,(products_reaction,products_species)),shape=(self.nr,self.ns))
   
    for i in range(self.ns):
        species_in_reaction_as_reactant[i] = list(set(species_in_reaction_as_reactant[i]))
        species_in_reaction_as_product[i] = list(set(species_in_reaction_as_product[i]))
        species_in_reaction_as_reactant_or_product[i] = list(set(species_in_reaction_as_reactant_or_product[i]))
   
    self.species_in_reaction_as_reactant = species_in_reaction_as_reactant
    self.species_in_reaction_as_product = species_in_reaction_as_product
    self.species_in_reaction_as_reactant_or_product = species_in_reaction_as_reactant_or_product
    
    # Identify the reaction class for every reaction
    self.reaction_class_belonging = [-1]*self.nr
    nc = len(self.reaction_class_name)
    for k in range(nc):
        for i in range(len(self.reaction_class_indices[k])):
            self.reaction_class_belonging[ self.reaction_class_indices[k][i] ] = k
    
    
  def ProcessingReactions(self):
    
    for i in range(self.nr):

        if (i%1000 == 0):
            print('Processing reactions ', i)

        ir = sparse.find(self.nur.getrow(i))[1]
        nur = sparse.find(self.nur.getrow(i))[2]
        if (len(ir)==1 and nur[0] == 2.):
            ir = [ir[0], ir[0]]
            nur = [1.,1.]
        react = [ self.species[i] for i in ir ] 

        ip = sparse.find(self.nup.getrow(i))[1]
        nup = sparse.find(self.nup.getrow(i))[2]
        if (len(ip)==1 and nup[0] == 2.):
            ip = [ip[0], ip[0]]
            nup = [1.,1.]
        prod = [self.species[i] for i in ip]    

        reaction = {    'index': i, \
                        'react': react, 'ir': ir, 'nur': nur, \
                        'prod': prod, 'ip': ip, 'nup': nup, \
                        'A': self.A[i], 'Beta': self.Beta[i], 'E_over_R': self.E_over_R[i] 
                   }

        self.reactions.append(reaction)
  
  
  def ReactionsWithSpecies(self, species_name, flag):
    
    if (species_name in self.species):
    
        index = self.species.index(species_name)
        
        if (flag == 'R'):
            return self.species_in_reaction_as_reactant[index]
        elif (flag == 'P'):
            return self.species_in_reaction_as_product[index]
        elif (flag == 'RP'):
            return self.species_in_reaction_as_reactant_or_product[index]
            
    else:
    
        return []
    
  
  
  def SubReactionsWithSpecies(self, species_name, flag, indices):
    
    
    
    if (species_name in self.species):
    
        index = self.species.index(species_name)
        
        reactions = []
        
        if (flag == 'R'):
            n = len(self.species_in_reaction_as_reactant[index])
            for i in range(n):
                if self.species_in_reaction_as_reactant[index][i] in indices:
                    reactions.append(self.species_in_reaction_as_reactant[index][i])
                    
        elif (flag == 'P'):
            n = len(self.species_in_reaction_as_product[index])
            for i in range(n):
                if self.species_in_reaction_as_product[index][i] in indices:
                    reactions.append(self.species_in_reaction_as_product[index][i])

        elif (flag == 'RP'):
            n = len(self.species_in_reaction_as_reactant_or_product[index])
            for i in range(n):
                if self.species_in_reaction_as_reactant_or_product[index][i] in indices:
                    reactions.append(self.species_in_reaction_as_reactant_or_product[index][i])
                    
        return (np.unique(reactions)).tolist()
        
    else:
    
        return []
    
    
  
  def ReactionsWithMultipleSpecies(self, species_names, flags, logic_operator):
  
    indices = list(range(0,self.nr))
    return self.SubReactionsWithMultipleSpecies(species_names, flags, indices, logic_operator)
  
  
  def SubReactionsWithMultipleSpecies(self, species_names, flags, indices, logic_operator):
    
    ns = len(species_names)

    if (logic_operator == 'OR'):
        selection = []
        for i in range(0,ns):
            print(species_names[i])
            local_selection = self.SubReactionsWithSpecies(species_names[i], flags[i], indices)
            selection.extend(local_selection)
        return (np.unique(selection)).tolist()
        
    elif (logic_operator == 'AND'):
        selection = self.SubReactionsWithSpecies(species_names[0], flags[0], indices)
        for i in range(1,ns):
            selection = self.SubReactionsWithSpecies(species_names[i], flags[i], selection)
        return (np.unique(selection)).tolist()
        
    return []
    
    
  def ReactionsWithoutSpecies(self, species_name, flag):
    
    indices = list(range(0,self.nr))
    return self.SubReactionsWithoutSpecies(species_name, flag, indices)
    
    
  def SubReactionsWithoutSpecies(self, species_name, flag, indices):  
  
    reactions_with = self.SubReactionsWithSpecies(species_name, flag, indices)
    reactions_without = indices
  
    return np.array(list(set(reactions_without) - set(reactions_with)))
    
    
  def ReactionsWithoutMultipleSpecies(self, species_names, flags, logical_operator):
    
    indices = list(range(0,self.nr))
    return self.SubReactionsWithoutMultipleSpecies(species_names, flags, indices, logical_operator)  
    
    
  def SubReactionsWithoutMultipleSpecies(self, species_names, flags, indices, logical_operator):  
  
    reactions_with = self.SubReactionsWithMultipleSpecies(species_names, flags, indices, logical_operator)
    reactions_without = indices
  
    return np.array(list(set(reactions_without) - set(reactions_with)))  
    
    
  def ReactionsWithListsOfSpecies(self, single_species, flag_single_species, list_of_species, flags_list_of_species, logical_operator):
  
    indices = list(range(0,self.nr))
    return self.SubReactionsWithListsOfSpecies(single_species, flag_single_species, list_of_species, flags_list_of_species, indices, logical_operator)
    
    
  def SubReactionsWithListsOfSpecies(self, single_species, flag_single_species, list_of_species, flags_list_of_species, indices, logical_operator):
  
    sub_indices = self.SubReactionsWithSpecies(single_species, flag_single_species, indices)
    
    reactions = []
    if (logical_operator == 'AND'):
        for i in range(len(list_of_species)):
            reactions.extend( self.SubReactionsWithSpecies(list_of_species[i], flags_list_of_species[i], sub_indices) )
    
    return (np.unique(reactions)).tolist()
    
    
  def PrintKineticMechanism(self, file_name, species, reactions):  

    nc = len(self.reaction_class_name)
    nr = len(reactions)

    f = open(file_name, "w")
    
    # ---------------------------------------------------------------------------------------------- #   
    # Write Summary
    # ---------------------------------------------------------------------------------------------- #    
    f.write('! CRECK Modeling Lab @ Politecnico di Milano \n')
    f.write('! Please visit our web-site:  http://creckmodeling.chem.polimi.it/ \n')
    f.write('\n')
    f.write('! Total number of reactions: ' + str(nr) + '\n')
 

    # In case reaction classes are available
    if (nc != 0):
        classes = self.OrganizeReactionsInClasses(reactions)
        f.write('!  * Class [' + 'undefined' + ']: ' + str(len(classes[nc])) + '\n')
        for k in range(nc):
            if (len(classes[k]) != 0):
                f.write('!  * Class [' + self.reaction_class_name[k] + ']: ' + str(len(classes[k])) + '\n')
             
             
    # ---------------------------------------------------------------------------------------------- #   
    # Write ELEMENTS section
    # ---------------------------------------------------------------------------------------------- #    
    self.PrintElementsSectionInCHEMKIN(f)
    
    # ---------------------------------------------------------------------------------------------- #   
    # Write SPECIES section
    # ---------------------------------------------------------------------------------------------- #
    self.PrintSpeciesSectionInCHEMKIN(f, species)

    # ---------------------------------------------------------------------------------------------- #   
    # Write REACTIONS section
    # ---------------------------------------------------------------------------------------------- #
    f.write('REACTIONS' + '\n')
    
    # No reaction classes
    if (nc == 0):
        
        for i in range(nr):
            index = reactions[i]
            f.write(self.reaction_lines[index])
       
    # Reaction classes
    else:
    
        classes = self.OrganizeReactionsInClasses(reactions)

        # Undefined class
        f.write('! Class: [' + 'undefined' + '] (' + str(len(classes[nc])) + ')\n\n')
        for i in range(len(classes[nc])):
            index = classes[nc][i]
            f.write(self.reaction_lines[index])
        f.write('\n')
            
        # Defined class
        for k in range(nc):
            if (len(classes[k]) != 0):
                f.write('! Class: [' + self.reaction_class_name[k] + '] (' + str(len(classes[k])) + ')\n\n')
                for i in range(len(classes[k])):
                    index = classes[k][i]
                    f.write(self.reaction_lines[index])
                f.write('\n\n')
        
    f.write('\n\n')
    f.write('END' + '\n')
    f.close()
   
   
  def SpeciesIndicesInSelectedReactions(self, reactions):
    
    flags = [0] * self.ns
    
    nr = len(reactions)
    for j in range(nr):
        i = reactions[j]
        indices = sparse.find(self.nur.getrow(i))[1]
        for k in range(len(indices)):
            flags[indices[k]] = 1
        indices = sparse.find(self.nup.getrow(i))[1]
        for k in range(len(indices)):
            flags[indices[k]] = 1
            
    return flags
    
  def SpeciesInSelectedReactions(self, reactions):
    
    flags = self.SpeciesIndicesInSelectedReactions(reactions)
    
    species = []
    for i in range(self.ns):
        if flags[i] == 1: species.append(self.species[i])
        
    return species
    
    
  def PartitionSootPrecursorReactions(self, pah_species):
  
    # Recognize family
    if (pah_species in self.Group('PAHLP')['list']): pah_family = 'PAHLP'
    elif (pah_species in self.Group('PAH34')['list']): pah_family = 'PAH34'
    elif (pah_species in self.Group('PAH12')['list']): pah_family = 'PAH12'
    
    # All the reactions in which the species is involved
    selected_all = self.ReactionsWithSpecies(pah_species, 'RP')
    
    # All the reactions in which the PAH is involved together with spherical particles and agrregates
    list_of_species = self.Group('SP')['list'] + self.Group('AGG')['list']
    selected_sp_agg = self.ReactionsWithListsOfSpecies( pah_species, 'RP', list_of_species, ["RP"]*len(list_of_species), "AND" )
    selected_gas_pah12_pah34_pahlp = list(set(selected_all) - set(selected_sp_agg))
    if ( len(selected_all) != len(selected_gas_pah12_pah34_pahlp) + len(selected_sp_agg) ):
        print(len(selected_all), len(selected_gas_pah12_pah34_pahlp), len(selected_sp_agg) )
        sys.exit("PartitionSootPrecursorsReactions failure at decomposition: ALL = GAS-PAH12-PAH34-PAHLP + SP-AGG!")
    
    # All the reactions in which the PAH is involved together with PAHLP
    list_of_species = self.Group('PAHLP')['list']
    if (pah_family == 'PAHLP'): list_of_species = list(set(list_of_species) - set([pah_species]))
    selected_pahlp = self.SubReactionsWithListsOfSpecies( pah_species, 'RP', list_of_species, ["RP"]*len(list_of_species), selected_gas_pah12_pah34_pahlp, "AND" )
    selected_gas_pah12_pah34 = list(set(selected_gas_pah12_pah34_pahlp) - set(selected_pahlp))
    if ( len(selected_gas_pah12_pah34_pahlp) != len(selected_gas_pah12_pah34) + len(selected_pahlp) ):
        print(len(selected_gas_pah12_pah34_pahlp), len(selected_gas_pah12_pah34), len(selected_pahlp))
        sys.exit("PartitionSootPrecursorsReactions failure at decomposition: GAS-PAH12-PAH34-PAHLP = GAS-PAH12-PAH34 + PAHLP!")    
    
    # All the reactions in which the PAH is involved together with PAH34
    list_of_species = self.Group('PAH34')['list']
    if (pah_family == 'PAH34'): list_of_species = list(set(list_of_species) - set([pah_species]))
    selected_pah34 = self.SubReactionsWithListsOfSpecies( pah_species, 'RP', list_of_species, ["RP"]*len(list_of_species), selected_gas_pah12_pah34, "AND" )
    selected_gas_pah12 = list(set(selected_gas_pah12_pah34) - set(selected_pah34))
    if ( len(selected_gas_pah12_pah34) != len(selected_gas_pah12) + len(selected_pah34) ):
        print(len(selected_gas_pah12_pah34), len(selected_gas_pah12), len(selected_pah34))
        sys.exit("PartitionSootPrecursorsReactions failure at decomposition: GAS-PAH12-PAH34 = GAS-PAH12 + PAH34!")    
        
    # All the reactions in which the PAH is involved together with PAH34
    list_of_species = self.Group('PAH12')['list']
    if (pah_family == 'PAH12'): list_of_species = list(set(list_of_species) - set([pah_species]))
    selected_pah12 = self.SubReactionsWithListsOfSpecies( pah_species, 'RP', list_of_species, ["RP"]*len(list_of_species), selected_gas_pah12, "AND" )
    selected_gas = list(set(selected_gas_pah12) - set(selected_pah12))
    if ( len(selected_gas_pah12) != len(selected_gas) + len(selected_pah12) ):
        print(len(selected_gas_pah12), len(selected_gas), len(selected_pah12))
        sys.exit("PartitionSootPrecursorsReactions failure at decomposition: GAS-PAH12 = GAS + PAH12!")
      
    #Final checks
    if ( len(selected_all) != len(selected_gas)+len(selected_pah12)+len(selected_pah34)+len(selected_pahlp)+len(selected_sp_agg)):
        print(len(selected_all), len(selected_gas), len(selected_pah12), len(selected_pah34), len(selected_pahlp), len(selected_sp_agg))
        sys.exit("PartitionSootPrecursorsReactions failure at total decomposition!")
        
    return (np.unique(selected_gas)).tolist(), (np.unique(selected_pah12)).tolist(), \
           (np.unique(selected_pah34)).tolist(), (np.unique(selected_pahlp)).tolist(), \
           (np.unique(selected_sp_agg)).tolist()
  
  
  def RemoveCrossingValues(self, v1,v2):
    
    crossing = []
    for i in range(len(v1)):
        for j in range(len(v2)):
            if (v1[i] == v2[j]): crossing.append(v2[j])
            
    crossing = (np.unique(crossing)).tolist()
    
    for i in range(len(crossing)):
        v2.remove(crossing[i])
        
    return v2
    
    
  def PartitionSootPrecursorsReactions(self, list_of_pahs):

    ns = len(list_of_pahs)
    
    reactions_gas = []
    reactions_pah12 = []
    reactions_pah34 = []
    reactions_pahlp = []
    reactions_sp_agg = []
    for i in range(ns):
        selected_gas, selected_pah12, selected_pah34, selected_pahlp, selected_sp_agg = self.PartitionSootPrecursorReactions(list_of_pahs[i])
        reactions_gas = reactions_gas + selected_gas
        reactions_pah12 = reactions_pah12 + selected_pah12
        reactions_pah34 = reactions_pah34 + selected_pah34
        reactions_pahlp = reactions_pahlp + selected_pahlp
        reactions_sp_agg = reactions_sp_agg + selected_sp_agg
        
    # Remove crossing values
    reactions_pahlp = self.RemoveCrossingValues(reactions_sp_agg,reactions_pahlp)
    reactions_pah34 = self.RemoveCrossingValues(reactions_sp_agg,reactions_pah34)
    reactions_pah12 = self.RemoveCrossingValues(reactions_sp_agg,reactions_pah12)
    reactions_gas   = self.RemoveCrossingValues(reactions_sp_agg,reactions_gas)
    reactions_pah34 = self.RemoveCrossingValues(reactions_pahlp,reactions_pah34)
    reactions_pah12 = self.RemoveCrossingValues(reactions_pahlp,reactions_pah12)
    reactions_gas = self.RemoveCrossingValues(reactions_pahlp,reactions_gas)
    reactions_pah12 = self.RemoveCrossingValues(reactions_pah34,reactions_pah12)
    reactions_gas = self.RemoveCrossingValues(reactions_pah34,reactions_gas)
    reactions_gas = self.RemoveCrossingValues(reactions_pah12,reactions_gas)
    
    # Split gas reactions
    r_pah12,r_pah34,r_pahlp = self.SplitPAHsGasOnlyReactions(list_of_pahs, reactions_gas)
    reactions_pah12 = reactions_pah12+r_pah12
    reactions_pah34 = reactions_pah34+r_pah34
    reactions_pahlp = reactions_pahlp+r_pahlp
    
    # Split spherical/aggregates reactions
    reactions_sp, reactions_agg = self.SplitPAHsSphericalAggregatesReactions(reactions_sp_agg)
        
    # All reactions
    reactions_all = (np.unique(reactions_pah12 + reactions_pah34 + reactions_pahlp + reactions_sp + reactions_agg)).tolist()
    
    return reactions_all, \
           (np.unique(reactions_pah12)).tolist(), (np.unique(reactions_pah34)).tolist(), (np.unique(reactions_pahlp)).tolist(), \
           (np.unique(reactions_sp)).tolist(), (np.unique(reactions_agg)).tolist()
           
           
  def SplitPAHsGasOnlyReactions(self, list_of_pahs, indices):
  
    pahslp = []
    pahs34 = []
    pahs12 = []
    for i in range(len(list_of_pahs)):
        if   ( list_of_pahs[i] in self.Group('PAHLP')['list']):   pahslp.append(list_of_pahs[i])
        elif ( list_of_pahs[i] in self.Group('PAH34')['list']):   pahs34.append(list_of_pahs[i])
        elif ( list_of_pahs[i] in self.Group('PAH12')['list']):   pahs12.append(list_of_pahs[i])
   
    reactions_pahlp = (np.unique( self.SubReactionsWithMultipleSpecies(pahslp, ['RP']*len(pahslp), indices, 'OR') )).tolist()
    indices_minus_pahlp = list(set(indices)-set(reactions_pahlp))
    reactions_pah34 = (np.unique( self.SubReactionsWithMultipleSpecies(pahs34, ['RP']*len(pahs34), indices_minus_pahlp, 'OR') )).tolist()
    indices_minus_pah34 = list(set(indices_minus_pahlp)-set(reactions_pah34))
    reactions_pah12 = (np.unique( self.SubReactionsWithMultipleSpecies(pahs12, ['RP']*len(pahs12), indices_minus_pah34, 'OR') )).tolist()
    
    return reactions_pah12, reactions_pah34, reactions_pahlp
    
    
  def SplitPAHsSphericalAggregatesReactions(self, indices):
  
    reactions_agg = (np.unique( self.SubReactionsWithMultipleSpecies(self.Group('AGG')['list'], ['RP']*len(self.Group('AGG')['list']), indices, 'OR') )).tolist()
    indices_minus_agg = list(set(indices)-set(reactions_agg))
    reactions_sp = (np.unique( self.SubReactionsWithMultipleSpecies(self.Group('SP')['list'], ['RP']*len(self.Group('SP')['list']), indices_minus_agg, 'OR') )).tolist()
    
    return reactions_sp, reactions_agg
    
    
  def SplitSphericalAggregatesReactions(self, indices):
  
    reactions_sp = (np.unique( self.SubReactionsWithMultipleSpecies(self.Group('SP')['list'], ['RP']*len(self.Group('SP')['list']), indices, 'OR') )).tolist()
    reactions_sp_agg = (np.unique( self.SubReactionsWithMultipleSpecies(self.Group('AGG')['list'], ['RP']*len(self.Group('AGG')['list']), reactions_sp, 'OR') )).tolist()
    reactions_sp_only = list(set(reactions_sp)-set(reactions_sp_agg))
    reactions_agg_only = list(set(indices)-set(reactions_sp))
    
    return reactions_sp_only, reactions_sp_agg, reactions_agg_only
    
    
  def PrintElementsSectionInCHEMKIN(self, f):
    
    f.write('\n')
    f.write('ELEMENTS' + '\n\n')
    for i in range(self.ne):
        f.write(self.elements[i] + '\n')
    f.write('\n')
    f.write('END' + '\n')
    f.write('\n\n')
    
    
  def PrintSpeciesSectionInCHEMKIN(self, f, species):
    
    # Force inclusion of inert species
    inerts = []
    if ( ('HE' in species) == False and ('HE' in self.species) == True ): inerts.append('HE')
    if ( ('AR' in species) == False and ('AR' in self.species) == True ): inerts.append('AR')
    if ( ('N2' in species) == False and ('N2' in self.species) == True ): inerts.append('N2')
    species = species + inerts
    ntot = len(species)
    
    # Split the species in groups
    species_gas, species_pahs12, species_pahs34, species_pahslp, species_sp, species_agg, species_carbon, species_inerts = self.SortAndSplitSpecies(species, '')
    
   
    # Write SPECIES section
    f.write('SPECIES' + '\n')
    f.write('! Total number of species: ' + str(ntot) + '\n\n')
    
    # Gas-phase species
    f.write('! Gas-phase species (' + str(len(species_gas)) + ')\n\n')
    for i in range(len(species_gas)):
        f.write(species_gas[i] + '\n')
    f.write('\n\n')
    
    # PAHs12
    if (len(species_pahs12) != 0):
        f.write('! Soot precursors (PAHs 1-2 aromatic rings) (' + str(len(species_pahs12)) + ')\n\n')
        for i in range(len(species_pahs12)):
            f.write(species_pahs12[i] + '\n')
        f.write('\n\n')
    
    # PAHs34
    if (len(species_pahs34) != 0):
        f.write('! Soot precursors (PAHs 3-4 aromatic rings) (' + str(len(species_pahs34)) + ')\n\n')
        for i in range(len(species_pahs34)):
            f.write(species_pahs34[i] + '\n')
        f.write('\n\n')
        
    # PAHsLP
    if (len(species_pahslp) != 0):
        f.write('! Soot precursors (PAHs more than 4 aromatic rings, up to C160) (' + str(len(species_pahslp)) + ')\n\n')
        for i in range(len(species_pahslp)):
            f.write(species_pahslp[i] + '\n')
        f.write('\n\n') 

    # Spherical particles
    if (len(species_sp) != 0):
        f.write('! Soot spherical particles (' + str(len(species_sp)) + ')\n\n')
        for i in range(len(species_sp)):
            f.write(species_sp[i] + '\n')
        f.write('\n\n')           

    # Soot aggregates
    if (len(species_agg) != 0):
        f.write('! Soot aggregates (fractal diameter 1.8) (' + str(len(species_agg)) + ')\n\n')
        for i in range(len(species_agg)):
            f.write(species_agg[i] + '\n')
        f.write('\n\n') 

    # Carbon
    if (len(species_carbon) != 0):
        f.write('! Solid carbon (' + str(len(species_carbon)) + ')\n\n')
        for i in range(len(species_carbon)):
            f.write(species_carbon[i] + '\n')
        f.write('\n\n')  

    # Inerts
    if (len(species_inerts) != 0):
        f.write('! Inerts (' + str(len(species_inerts)) + ')\n\n')
        for i in range(len(species_inerts)):
            f.write(species_inerts[i] + '\n')
        f.write('\n\n')           
        
    f.write('END' + '\n')
    f.write('\n\n')
    
           
  def PrintKineticMechanismByClasses(self, file_name, by_groups, species, reaction_indices, reaction_groups, reaction_comments):  

    ngroups = len(reaction_indices)
    nc = len(self.reaction_class_name)
    
    nr_tot = 0
    for i in range(ngroups):
        nr_tot = nr_tot + len(reaction_indices[i])
        
    f = open(file_name, "w")
    
    # ---------------------------------------------------------------------------------------------- #   
    # Write Summary
    # ---------------------------------------------------------------------------------------------- #    
    f.write('! CRECK Modeling Lab @ Politecnico di Milano \n')
    f.write('! Please visit our web-site:  http://creckmodeling.chem.polimi.it/ \n')
    f.write('\n')
    f.write('! Total number of reactions: ' + str(nr_tot) + '\n')
 

    # In case reaction groups are available
    f.write('! --------------------------------------------------------------------------- !\n')
    f.write('! Kinetic mechanism structure by Groups\n')
    f.write('! --------------------------------------------------------------------------- !\n')
    for j in range(ngroups):
        nr = len(reaction_indices[j])
        f.write('!  * Group [' + reaction_groups[j] + ']: ' + str(nr) + '\n')
        if (nc != 0):
            classes = self.OrganizeReactionsInClasses(reaction_indices[j])          
            if (len(classes[nc]) != 0):
                f.write('!    - Class [' + 'undefined' + ']: ' + str(len(classes[nc])) + '\n')
            for k in range(nc):
                if (len(classes[k]) != 0):
                    f.write('!    - Class [' + self.reaction_class_name[k] + ']: ' + str(len(classes[k])) + '\n')
    
    
    if (nc != 0):

        f.write('\n')
        f.write('! --------------------------------------------------------------------------- !\n')
        f.write('! Kinetic mechanism structure by Classes\n')
        f.write('! --------------------------------------------------------------------------- !\n')
    
        groups = self.OrganizeReactionsInGroups(reaction_indices)
        
        # Number of reactions per group
        nrc = [0]*(nc+1)
        for k in range(nc+1):
            for j in range(len(groups)):
                nrc[k] = nrc[k] + len(groups[j][k])        
        
        f.write('!  * Class [' + 'undefined' + ']: ' + str(nrc[nc]) + '\n')
        for j in range(len(groups)):
            if (len(groups[j][nc])!=0): f.write('!    - Group [' + reaction_groups[j] + ']: ' + str(len(groups[j][nc])) + '\n')
        
        for k in range(nc):
            f.write('!  * Class [' + self.reaction_class_name[k] + ']: ' + str(nrc[k]) + '\n')
            for j in range(len(groups)):
                if (len(groups[j][k])!=0): f.write('!    - Group [' + reaction_groups[j] + ']: ' + str(len(groups[j][k])) + '\n')
            
            
    # ---------------------------------------------------------------------------------------------- #   
    # Write ELEMENTS section
    # ---------------------------------------------------------------------------------------------- #
    self.PrintElementsSectionInCHEMKIN(f)
    
    # ---------------------------------------------------------------------------------------------- #   
    # Write SPECIES section
    # ---------------------------------------------------------------------------------------------- #
    self.PrintSpeciesSectionInCHEMKIN(f, species)

    # ---------------------------------------------------------------------------------------------- #   
    # Write REACTIONS section
    # ---------------------------------------------------------------------------------------------- #
    f.write('REACTIONS' + '\n')
    f.write('! Total number of reactions: ' + str(nr_tot) + '\n\n')
    
    # No reaction classes
    if (nc == 0):
    
        for j in range(ngroups):
        
            nr = len(reaction_indices[j])
            
            f.write('\n')
            f.write('! [' + reaction_groups[j] + ']' + '\n')
            f.write('! ' + reaction_comments[j] + ' (' + str(nr) + ')\n')
            for i in range(nr):
                index = reaction_indices[j][i]
                f.write(self.reaction_lines[index])
            f.write('\n')
            f.write('\n')
    
    # Reactions by groups
    elif (by_groups == True and nc != 0):
    
        for j in range(ngroups):
        
            nr = len(reaction_indices[j])
            classes = self.OrganizeReactionsInClasses(reaction_indices[j])
    
            f.write('\n')
            f.write('! [' + reaction_groups[j] + ']' + '\n')
            f.write('! ' + reaction_comments[j] + ' (' + str(nr) + ')\n')
            
            # Undefined class
            if (len(classes[nc]) != 0):
                f.write('! Class: [' + reaction_groups[j] + '][' + 'undefined' + '] (' + str(len(classes[nc])) + ')\n\n')
                for i in range(len(classes[nc])):
                    index = classes[nc][i]
                    f.write(self.reaction_lines[index])
                f.write('\n')
                
            # Defined class
            for k in range(nc):
                if (len(classes[k]) != 0):
                    f.write('! Class: [' + reaction_groups[j] + '][' + self.reaction_class_name[k] + '] (' + str(len(classes[k])) + ')\n\n')
                    for i in range(len(classes[k])):
                        index = classes[k][i]
                        f.write(self.reaction_lines[index])
                    f.write('\n\n')
    
            f.write('\n')
            f.write('\n')
            
            
    # Reaction by classes       
    elif (by_groups == False and nc != 0):
    
        groups = self.OrganizeReactionsInGroups(reaction_indices)
        
        # Number of reactions per group
        nrc = [0]*(nc+1)
        for k in range(nc+1):
            for j in range(len(groups)):
                nrc[k] = nrc[k] + len(groups[j][k])
                
        if (nrc[nc] != 0):
            f.write('\n')
            f.write('! [CLASS] [' + 'undefined' + '] (' + str(nrc[nc]) + ')\n\n')
            
            for j in range(len(groups)):
                if (len(groups[j][nc])!=0): 
                    f.write('! Group [' + reaction_groups[j] + ']: ' + str(len(groups[j][nc])) + '\n\n')
                    for i in range(len(groups[j][nc])):
                        index = groups[j][nc][i]
                        f.write(self.reaction_lines[index])
                    f.write('\n\n')        
                
    
        for k in range(nc):
        
            if (nrc[k] != 0):
            
                f.write('\n')
                f.write('! [SOOTCLASS] [' + self.reaction_class_name[k] + '] (' + str(nrc[k]) + ')\n\n')
                
                for j in range(len(groups)):
                    if (len(groups[j][k])!=0): 
                        f.write('! Group [' + reaction_groups[j] + ']: ' + str(len(groups[j][k])) + '\n\n')
                        for i in range(len(groups[j][k])):
                            index = groups[j][k][i]
                            f.write(self.reaction_lines[index])
                        f.write('\n\n')
        
    f.write('\n\n')
    f.write('END' + '\n')
    
    f.close()
  
  
  def OrganizeReactionsInGroups(self, reaction_indices):
  
    ngroups = len(reaction_indices)
    groups = [[] for x in range(ngroups)]
    
    nc = len(self.reaction_class_name)
    for k in range(ngroups):
        classes = [[] for x in range(nc+1)]
        
        nr = len(reaction_indices[k])
        for i in range(nr):
            index = self.reaction_class_belonging[reaction_indices[k][i]]
            if (index == -1): classes[nc].append(reaction_indices[k][i])
            else: classes[ index ].append(reaction_indices[k][i])
            
        groups[k] = classes
        
    return groups
    
    
  def OrganizeReactionsInClasses(self, reactions):
  
    nc = len(self.reaction_class_name)
    nr = len(reactions)
    classes = [[] for x in range(nc+1)]
    
    for i in range(nr):
        index = self.reaction_class_belonging[reactions[i]]
        if (index == -1): classes[nc].append(reactions[i])
        else: classes[ index ].append(reactions[i])
        
    return classes
  
  
  def SortAccordingToMolecularWeight(self, species):

    mws = []
    for i in range(len(species)):
        mws.append(self.mws[self.species.index(species[i])])
    return [element for _, element in sorted( zip(mws, species) )]
  
  
  def SortAndSplitSpecies(self, species, criterion):
    
    species_gas = species
    
    # Inerts
    inerts = []
    if ('HE' in species_gas): inerts.append('HE')
    if ('AR' in species_gas): inerts.append('AR')
    if ('N2' in species_gas): inerts.append('N2')
    species_gas = list(set(species_gas) - set(inerts))
    
    # Carbon
    carbon = []
    if ('CSOLID' in species_gas): carbon.append('CSOLID')
    species_gas = list(set(species_gas) - set(carbon))
    
    # PAHs12
    pahs12 = []
    for i in range(len(self.Group('PAH12')['list'])):
        if (self.Group('PAH12')['list'][i] in species_gas): 
            pahs12.append(self.Group('PAH12')['list'][i])       
    species_gas = list(set(species_gas) - set(pahs12))
        
    # PAHs34
    pahs34 = []
    for i in range(len(self.Group('PAH34')['list'])):
        if (self.Group('PAH34')['list'][i] in species_gas): 
            pahs34.append(self.Group('PAH34')['list'][i])
    species_gas = list(set(species_gas) - set(pahs34))
    
    # PAHsLP
    pahslp = []
    for i in range(len(self.Group('PAHLP')['list'])):
        if (self.Group('PAHLP')['list'][i] in species_gas): 
            pahslp.append(self.Group('PAHLP')['list'][i])
    species_gas = list(set(species_gas) - set(pahslp))   

    # Spherical particles
    sp = []
    for i in range(len(self.Group('SP')['list'])):
        if (self.Group('SP')['list'][i] in species_gas): 
            sp.append(self.Group('SP')['list'][i])
    species_gas = list(set(species_gas) - set(sp))

    # Aggregates
    agg = []
    for i in range(len(self.Group('AGG')['list'])):
        if (self.Group('AGG')['list'][i] in species_gas): 
            agg.append(self.Group('AGG')['list'][i])
    species_gas = list(set(species_gas) - set(agg))
    
    # Fixed gas species
    fixed_species = []
    if ('C2H4' in species_gas): fixed_species.append('C2H4')
    if ('H2' in species_gas):   fixed_species.append('H2')
    if ('C2H2' in species_gas): fixed_species.append('C2H2')
    if ('O2' in species_gas):   fixed_species.append('O2')
    if ('CO2' in species_gas):  fixed_species.append('CO2')
    if ('H2O' in species_gas):  fixed_species.append('H2O')
    if ('CO' in species_gas):   fixed_species.append('CO')
    if ('OH' in species_gas):   fixed_species.append('OH')
    species_gas = list(set(species_gas) - set(fixed_species))
    species_gas = fixed_species + self.SortAccordingToMolecularWeight(species_gas)
            
    return  species_gas, \
            self.SortAccordingToMolecularWeight(pahs12), \
            self.SortAccordingToMolecularWeight(pahs34), \
            sorted(pahslp), \
            sorted(sp), \
            sorted(agg), \
            carbon, inerts
