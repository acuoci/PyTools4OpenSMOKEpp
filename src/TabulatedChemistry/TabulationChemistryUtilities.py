'''
MODULE: TabulationChemistryUtilities
@Authors:
    Alberto Cuoci [1]
    [1]: CRECK Modeling Lab, Department of Chemistry, Materials, and Chemical Engineering, Politecnico di Milano
@Contacts:
    alberto.cuoci@polimi.it
@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: alberto.cuoci@polimi.it
'''

import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.style.use('default')
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

import xml.etree.ElementTree as ET
from xml.etree import ElementTree
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

# Write lookup table
def PrintMainLookupTable(kinetics, flames, alpha, output_folder):
    
    # create the file structure
    opensmoke = ET.Element('opensmoke')
    comment = ET.Comment('Example of lookup table')
    opensmoke.append(comment)

    Y_definition = ET.SubElement(opensmoke, 'ProgressVariableDefinition')
    dummy = '\n'
    dummy = dummy + str(len(alpha)) + '\n'
    for i in range(len(alpha)):
        dummy = dummy + alpha[i][0] + ' ' + str(alpha[i][1]) + '\n'
    Y_definition.text = str(dummy)
    
    # Fuel composition
    YFuel = ET.SubElement(opensmoke, 'YFuel')
    dummy = '\n'
    count = 0
    for i in range(kinetics.ns):
        if (flames[0].Y[0,i] > 1.e-12):
            dummy = dummy + kinetics.species[i] + ' ' + str(flames[0].Y[0,i])
            dummy = dummy + '\n'
            count = count+1
    YFuel.text = '\n' + str(count) + dummy
    
    # Oxidizer composition
    YOx = ET.SubElement(opensmoke, 'YOx')
    dummy = '\n'
    count = 0
    for i in range(kinetics.ns):
        if (flames[0].Y[-1,i] > 1.e-12):
            dummy = dummy + kinetics.species[i] + ' ' + str(flames[0].Y[-1,i])
            dummy = dummy + '\n'
            count = count + 1
    YOx.text = '\n' + str(count) + dummy   
    
    npts_Z = len(flames[0].Z_grid)
    Z_points = ET.SubElement(opensmoke, 'Z-points')
    Z_points.text = str(npts_Z)

    npts_C = len(flames[0].y_grid)
    C_points = ET.SubElement(opensmoke, 'C-points')
    C_points.text = str(npts_C)

    Z_coordinates = ET.SubElement(opensmoke, 'Z-coordinates')
    dummy = ''
    for i in range(npts_Z):
        dummy = dummy + str(flames[0].Z_grid[i]) + ' '
    Z_coordinates.text = dummy

    C_coordinates = ET.SubElement(opensmoke, 'C-coordinates')
    dummy = ''
    for i in range(npts_C):
        dummy = dummy + str(flames[0].y_grid[i]) + ' '
    C_coordinates.text = dummy
    
    # Normalized progress variable
    Ctilde_grid = []
    for i in range(npts_C):
        Ctilde_grid.append( 0. + (1.-0.)/(npts_C-1)*i )

    # Maximum and minimum progress variable
    Cmax = []
    Cmin = []
    for k in range(npts_Z):
        dummy = []
        for i in range(len(flames)):
            dummy.append(flames[i].y_int_Z[k])   
        Cmax.append(np.max(dummy))
        Cmin.append(np.min(dummy))
    
    # Normalized progress variable
    Ctilde_coordinates = ET.SubElement(opensmoke, 'Ctilde-coordinates')
    dummy = ''
    for i in range(npts_C):
        dummy = dummy + str(Ctilde_grid[i]) + ' '
    Ctilde_coordinates.text = dummy    
    
    # Min progress variable
    Cmin_values = ET.SubElement(opensmoke, 'Cmin-values')
    dummy = ''
    for i in range(npts_C):
        dummy = dummy + str(Cmin[i]) + ' '
    Cmin_values.text = dummy     
    
    # Max progress variable
    Cmax_values = ET.SubElement(opensmoke, 'Cmax-values')
    dummy = ''
    for i in range(npts_C):
        dummy = dummy + str(Cmax[i]) + ' '
    Cmax_values.text = dummy      

    # create a new XML file with the results
    mylookuptable = prettify(opensmoke)
    lookuptable_file = open(output_folder + "lookuptable.main.xml", "w")
    lookuptable_file.write(mylookuptable)
    
    
def PlotProfile(xexact, yexact, xrecon, yrecon, title, xlabel, ylabel, output_folder, filename):
    
    fig=plt.figure()
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.plot(xexact, yexact)
    plt.plot(xrecon, yrecon)
    plt.legend(['Simulated', 'Tabulated'])
    plt.savefig(output_folder + filename)
    

def PlotScatter(exact, recon, title, output_folder, filename):
    
    lineStart = exact.min()
    lineEnd = exact.max()

    plt.figure(figsize=(4,4))
    plt.gca().set_aspect('equal', adjustable='box')

    plt.scatter(exact, recon, marker='+', color='b', s=6)
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], '-', color = 'r', linewidth=1)
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.xlabel('Simulated')
    plt.ylabel('Tabulated')
    plt.title(title)
    
    plt.savefig(output_folder + filename)
    
    
def PlotScatterComparisonCFD(exact, recon, title, output_folder, filename):
    
    lineStart = exact.min()
    lineEnd = exact.max()
    
    plt.figure(figsize=(4,4))
    plt.gca().set_aspect('equal', adjustable='box')

    plt.scatter(exact, recon, marker='+', color='b', s=6)
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], '-', color = 'r', linewidth=1)
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.xlabel('Simulated')
    plt.ylabel('Tabulated')
    plt.title(title)
    
    plt.savefig(output_folder + filename)
    
    
def nrmse(targets,predictions):
    
    sum_squared = 0.
    for i in range(len(x)):
        sum_squared = sum_squared + (targets[i]-predictions[i])*(targets[i]-predictions[i])
    
    mu = targets.mean()
    return np.sqrt( sum_squared/len(targets) ) / mu
    
    
def nrmse_total(kinetics, flames, label, predictions):
    
    x = []
    y = []
    for i in range(len(flames)):
        if (label == "T"):       x.extend(flames[i].T)
        elif (label == "PAH12"): x.extend(flames[i].pah12_Y)
        elif (label == "PAH34"): x.extend(flames[i].pah34_Y)
        elif (label == "PAHLP"): x.extend(flames[i].pahlp_Y)
        elif (label == "SP"):    x.extend(flames[i].sp_Y)
        elif (label == "AGG"):   x.extend(flames[i].agg_Y)
        else:                    x.extend(flames[i].Y[:,kinetics.species.index(label)])
        y.extend(predictions[i])
    
    x = np.array(x)
    y = np.array(y)
    
    sum_squared = 0.
    for i in range(len(x)):
        sum_squared = sum_squared + (x[i]-y[i])*(x[i]-y[i])
    
    mu = x.mean()
    return np.sqrt( sum_squared/len(x) ) / mu
    
    
def ScatterPlotMultipleFlames(kinetics, flames, label, recon, title, output_folder, filename):
    
    x = []
    y = []
    for i in range(len(flames)):
        if (label == "T"):       x.extend(flames[i].T)
        elif (label == "PAH12"): x.extend(flames[i].pah12_Y)
        elif (label == "PAH34"): x.extend(flames[i].pah34_Y)
        elif (label == "PAHLP"): x.extend(flames[i].pahlp_Y)
        elif (label == "SP"):    x.extend(flames[i].sp_Y)
        elif (label == "AGG"):   x.extend(flames[i].agg_Y)
        else:                    x.extend(flames[i].Y[:,kinetics.species.index(label)])
        y.extend(recon[i])
    PlotScatter(np.array(x),np.array(y), title, output_folder, filename)
    
    
def PlotMultipleProfiles(kinetics, flames, label, yrecon, title, xlabel, ylabel, output_folder, filename):
    
    fig=plt.figure()
    
    plt.xlabel('Simulated')
    plt.ylabel('Tabulated')
    plt.title(title)
    
    for i in range(len(flames)):
        if (label == "T"):       plt.plot(flames[i].Z, flames[i].T)
        elif (label == "PAH12"): plt.plot(flames[i].Z, flames[i].pah12_Y)
        elif (label == "PAH34"): plt.plot(flames[i].Z, flames[i].pah34_Y)
        elif (label == "PAHLP"): plt.plot(flames[i].Z, flames[i].pahlp_Y)   
        elif (label == "SP"):    plt.plot(flames[i].Z, flames[i].sp_Y) 
        elif (label == "AGG"):   plt.plot(flames[i].Z, flames[i].agg_Y) 
        else:                    plt.plot(flames[i].Z, flames[i].Y[:,kinetics.species.index(label)])
        plt.plot(flames[i].Z, yrecon[i])
    
    plt.savefig(output_folder + filename)
    

def PlotMultipleScatteredProfiles(kinetics, flames, label, yrecon, title, xlabel, ylabel, output_folder, filename):
    
    fig=plt.figure()
    
    plt.xlabel('Simulated')
    plt.ylabel('Tabulated')
    plt.title(title)
    
    for i in range(len(flames)):
        if (label == "T"):       plt.scatter(flames[i].Z, flames[i].T, color='b')
        elif (label == "PAH12"): plt.scatter(flames[i].Z, flames[i].pah12_Y, color='b')
        elif (label == "PAH34"): plt.scatter(flames[i].Z, flames[i].pah34_Y, color='b')
        elif (label == "PAHLP"): plt.scatter(flames[i].Z, flames[i].pahlp_Y, color='b')  
        elif (label == "SP"):    plt.scatter(flames[i].Z, flames[i].sp_Y, color='b') 
        elif (label == "AGG"):   plt.scatter(flames[i].Z, flames[i].agg_Y, color='b') 
        else:                    plt.scatter(flames[i].Z, flames[i].Y[:,kinetics.species.index(label)], color='b')
        plt.scatter(flames[i].Z, yrecon[i], color='r')
    
    plt.savefig(output_folder + filename)
