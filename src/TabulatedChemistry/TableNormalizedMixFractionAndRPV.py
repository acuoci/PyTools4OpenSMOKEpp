'''
MODULE: TableNormalizedMixFractionAndRPV
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
from scipy import interpolate
from time import process_time

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.style.use('default')
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

class TableNormalizedMixFractionAndRPV:
    
    def __init__(self, kinetics, flames, label):
    
        # Start time
        tstart = process_time() 

        nflames = len(flames)

        # Un-normalized grids
        Z_grid_abs = np.copy(flames[0].Z_grid)
        C_grid_abs = np.copy(flames[0].y_grid)

        # Grid points
        npts_Z = len(Z_grid_abs)
        npts_C = len(C_grid_abs) 

        # Normalized grids
        Ztilde_grid = []
        for i in range(npts_Z):
            Ztilde_grid.append( 0. + (1.-0.)/(npts_Z-1)*i )

        Ctilde_grid = []
        for i in range(npts_C):
            Ctilde_grid.append( 0. + (1.-0.)/(npts_C-1)*i )

        # Table
        table = np.empty((npts_Z, npts_C))
        table.fill(np.nan)

        # Maximum and minimum progress variable
        Cmax = []
        Cmin = []
        for k in range(npts_Z):
            dummy = []
            for i in range(len(flames)):
                dummy.append(flames[i].y_int_Z[k])   
            Cmax.append(np.max(dummy))
            Cmin.append(np.min(dummy))

        # Loop over all the mixture fraction points
        for k in range(npts_Z):
            
            border = 0
            
            if (Cmax[k]-Cmin[k] < 0.):
                if ( abs(Cmax[k]-Cmin[k]) > 1.e-12):
                    print("WARNING: monotonicity condition is not verified in Z=", Z_grid_abs[k], "Cmin=", Cmin[k], "Cmax=",Cmax[k])
                
            if (Cmax[k]-Cmin[k] <= 0.):
                border = 1
                #print("Border point: ", Z_grid_abs[k])

            
            # Basic grid (also for boundaries)
            Ctilde_grid_local = np.empty(nflames)
            for i in range(nflames):
                Ctilde_grid_local[i] = 0. + (1.-0.)/(nflames-1)*i
            
            # Internal points
            if (border == 0):
                for i in range(nflames):
                    Ctilde = 0.
                    if (Cmax[k]-Cmin[k] > 0.):
                        Ctilde = (flames[i].y_int_Z[k]-Cmin[k])/(Cmax[k]-Cmin[k])
                    if (Ctilde<0.): Ctilde = 0.
                    elif (Ctilde>1.): Ctilde = 1.
                    Ctilde_grid_local[i] = Ctilde

            # Local profile
            local = np.empty(nflames)
            if (label == 'T'):
                for i in range(nflames):
                    local[i] = flames[i].T_int_Z[k]
            elif (label == 'Omegay'):
                for i in range(nflames):
                    local[i] = flames[i].Omegay_int_Z[k]
            elif (label == 'rho'):
                for i in range(nflames):
                    local[i] = flames[i].rho_int_Z[k]
            elif (label == 'PAH12'):
                for i in range(nflames):
                    local[i] = flames[i].pah12_int_Z[k]
            elif (label == 'PAH34'):
                for i in range(nflames):
                    local[i] = flames[i].pah34_int_Z[k]
            elif (label == 'PAHLP'):
                for i in range(nflames):
                    local[i] = flames[i].pahlp_int_Z[k]  
            elif (label == 'SP'):
                for i in range(nflames):
                    local[i] = flames[i].sp_int_Z[k]  
            elif (label == 'AGG'):
                for i in range(nflames):
                    local[i] = flames[i].agg_int_Z[k]  
            else:
                index = kinetics.species.index(label)
                for i in range(nflames):
                    local[i] = flames[i].Y_int_Z[k,index]

            # Create interpolation objects
            f = interpolate.interp1d(Ctilde_grid_local, local)

            # Interpolations
            table[k,:] = f(Ctilde_grid)

        # Assign class values
        self.label = label
        self.table = table
        self.Cmin = Cmin
        self.Cmax = Cmax
        self.Ztilde_grid = Ztilde_grid
        self.Ctilde_grid = Ctilde_grid
        
        self.flame_eq = flames[0]
        #self.flame_q = flames[nflames_steady]

        # Stop the stopwatch / counter
        tstop = process_time()

        print("CPU time (s): ", tstop-tstart)
        
    
    def BuildTableForGraphicalPurposes(self, kinetics, flames):
    
        # Start time
        tstart = process_time() 

        nflames = len(flames)

        Z_grid_abs = np.copy(flames[0].Z_grid)
        C_grid_abs = np.copy(flames[0].y_grid)

        npts_Z = len(Z_grid_abs)
        npts_C = len(C_grid_abs) 

        tablegp = np.empty((npts_Z, npts_C))
        tablegp.fill(np.nan)


        # Loop over all the mixture fraction points
        for k in range(npts_Z):

            # Range of progress variable at the given mixture fraction
            C_grid_local = []
            for i in range(nflames):
                C_grid_local.append(flames[i].y_int_Z[k])
            min_C_local = C_grid_local[-1]
            max_C_local = C_grid_local[0]

            # Be sure that interpolation query is well defined

            j_min=0;
            for j in range(npts_C):
                if (C_grid_abs[j] >= min_C_local):
                    j_min = j;
                    break;

            j_max=0;
            for j in range(npts_C):
                if (C_grid_abs[j] > max_C_local):
                    j_max = j;
                    break;

            C_grid_corrected = np.copy(C_grid_abs[j_min:j_max])


            # Local profile
            local = np.empty(nflames)
            if (self.label == 'T'):
                for i in range(nflames):
                    local[i] = flames[i].T_int_Z[k]
            elif (self.label == 'Omegay'):
                for i in range(nflames):
                    local[i] = flames[i].Omegay_int_Z[k]
            elif (self.label == 'rho'):
                for i in range(nflames):
                    local[i] = flames[i].rho_int_Z[k]
            elif (self.label == 'PAH12'):
                for i in range(nflames):
                    local[i] = flames[i].pah12_int_Z[k]
            elif (self.label == 'PAH34'):
                for i in range(nflames):
                    local[i] = flames[i].pah34_int_Z[k]
            elif (self.label == 'PAHLP'):
                for i in range(nflames):
                    local[i] = flames[i].pahlp_int_Z[k]    
            elif (self.label == 'SP'):
                for i in range(nflames):
                    local[i] = flames[i].sp_int_Z[k]  
            elif (self.label == 'AGG'):
                for i in range(nflames):
                    local[i] = flames[i].agg_int_Z[k] 
            else:
                index = kinetics.species.index(self.label)
                for i in range(nflames):
                    local[i] = flames[i].Y_int_Z[k,index]

            # Create interpolation objects
            f = interpolate.interp1d(C_grid_local, local)

            # Interpolations
            tablegp[k,j_min:j_max] = f(C_grid_corrected)


        # Stop the stopwatch / counter
        tstop = process_time()

        print("CPU time (s): ", tstop-tstart)

        self.tablegp = tablegp
        
    
    def PlotMap(self, title, output_folder, filename):
    
        extent = np.min(self.flame_eq.Z_grid), np.max(self.flame_eq.Z_grid), np.min(self.flame_eq.y_grid), np.max(self.flame_eq.y_grid)
        c = plt.imshow(self.tablegp.transpose(), cmap='jet', aspect = 'auto', origin ='lower', extent=extent)
        plt.colorbar(c)
        plt.title(title, fontweight ="bold")
        plt.xlabel('mixture fraction Z [-]')
        plt.ylabel('progress variable C [-]')
        plt.plot(self.flame_eq.Z, self.flame_eq.y, color='black', linewidth=1)
        #plt.plot(self.flame_q.Z, self.flame_q.y, color='black', linewidth=1, linestyle='--')
        plt.ylim([0, 1.05*np.max(self.flame_eq.y_grid)])
        plt.savefig(output_folder + filename)
        
        
    def PlotMapTilde(self, title, output_folder, filename):
    
        extent = 0, 1, 0, 1
        c = plt.imshow(self.table.transpose(), cmap='jet', aspect = 'auto', origin ='lower', extent=extent)
        plt.colorbar(c)
        plt.title(title, fontweight ="bold")
        plt.xlabel('mixture fraction Z [-]')
        plt.ylabel('progress variable Ctilde [-]')
        plt.ylim([0, 1.05])
        plt.savefig(output_folder + filename)
    
    
    def PrintLookupTable(self, output_folder, name_extended):
    
        npts_Z = len(self.Ztilde_grid)
        npts_C = len(self.Ctilde_grid)
                
        print('Lookup table: ', self.label)
        f = open(output_folder + "lookuptable." + self.label + ".xml", "w")
        f.write("<?xml version=\"1.0\" ?>\n")
        f.write("<opensmoke>\n")    

        f.write("<Table" + name_extended + ">\n")    
        for i in range(npts_C):
            for j in range(npts_Z):
                f.write( "{:e} ".format(self.table[j,i]) )
            f.write('\n')         
        f.write("</Table" + name_extended + ">\n") 

        f.write("</opensmoke>\n")
        f.close()
        
        
    def Reconstruct(self, Z, C):

        npts_Z = len(self.Ztilde_grid)
        npts_C = len(self.Ctilde_grid)

        dZ = (1.-0.)/(npts_Z-1)
        dC = (1.-0.)/(npts_C-1)
        
        fcmin = interpolate.interp1d(self.Ztilde_grid, self.Cmin)
        fcmax = interpolate.interp1d(self.Ztilde_grid, self.Cmax)

        recon = np.empty((len(Z),1))
        recon.fill(np.nan)

        for i in range(len(Z)):
            
            Cmin = fcmin( min( max(Z[i],0.), 1.) )
            Cmax = fcmax( max( min(Z[i],1.), 0.) )
            
            #if (Cmax-Cmin <= 0.):
            #    print("WARNING: Cmax-Cmin=0")
            
            if (Cmax-Cmin <= 0.):
            
                iz = max( min( math.floor((Z[i]-0.)/dZ), npts_Z-1), 0)
                recon[i] = self.table[iz,0]
            
            else:
                
                # Normal Ctilde
                Ctilde = (C[i]-Cmin)/(Cmax-Cmin)
                #if (Ctilde<0.): Ctilde = 0.
                #elif (Ctilde>1.): Ctilde = 1.

                iz = math.floor((Z[i]-0.)/dZ)
                ic = math.floor((Ctilde-0.)/dC)

                if ( (iz < npts_Z-1) and (ic < npts_C-1) ):

                    Q11 = self.table[iz,ic]
                    Q21 = self.table[iz+1,ic]
                    Q12 = self.table[iz,ic+1]
                    Q22 = self.table[iz+1,ic+1]

                    z1 = self.Ztilde_grid[iz]
                    z2 = self.Ztilde_grid[iz+1]
                    c1 = self.Ctilde_grid[ic]
                    c2 = self.Ctilde_grid[ic+1]

                    coeff  = 1./dZ/dC

                    f = coeff * ( (z2-Z[i]) * ( Q11*(c2-Ctilde) + Q12 *(Ctilde-c1) ) + (Z[i]-z1) * (Q21 * (c2-Ctilde) + Q22 * (Ctilde-c1) ) );
                    recon[i] = f
                
                elif ( (ic >= npts_C-1) and (iz < npts_Z-1) ):
                    
                    #print("WARNING (topC): ", "point: ", i, "Z=", Z[i], "C=", C[i], "Ctilde=", Ctilde, "iz=", iz)
                    
                    Q1 = self.table[iz,-1]
                    Q2 = self.table[iz+1,-1]
                    z1 = self.Ztilde_grid[iz]
                    
                    f = Q1 + (Q2-Q1)/dZ*(Z[i]-z1)
                    recon[i] = f
                    
                elif ( (ic < npts_C-1) and (iz >= npts_Z-1) ):
                    
                    #print("WARNING (topZ): ", "point: ", i, "Z=", Z[i], "C=", C[i], "Ctilde=", Ctilde, "iz=", iz)
                    
                    Q1 = self.table[-1,ic]
                    Q2 = self.table[-1,ic+1]
                    c1 = self.Ctilde_grid[ic]
                    
                    f = Q1 + (Q2-Q1)/dC*(Ctilde-c1)
                    recon[i] = f 
                    
                elif ( (ic >= npts_C-1) and (iz >= npts_Z-1) ):
                    
                    #print("WARNING (topZC): ", "point: ", i, "Z=", Z[i], "C=", C[i], "Ctilde=", Ctilde, "iz=", iz)
                    
                    recon[i] = self.table[-1,-1]                           
                
                else:
                    
                    nothing_to_do = 0.
                    #print("Unreconstructed point: ", i, "z=", Z[i], "C=", C[i], "Ctilde=", Ctilde, "iz=", iz, "ic=", ic)

        return recon    

