# WavePy is a Wave Optics Simulation for Atmospheric Optics Modeling
# Authors: Jeff Beck, Celina Bekins, Jeremy Bos
# Michigan Technological University (c) 2016
# WavePy 0.1 Initial Release
# Contact: jpbos@mtu.edi
# Released under BSD attribution license please reference 
# Maintained at www.github.com/jpbos/WavePy
# Developed under Python 2.7

import numpy as np
from math import pi, gamma, cos, sin

class wavepy:
    def __init__(self,simOption=0,N=256,SideLen=1.28,NumScr=10,DRx=0.1,dx=5e-3,
                 wvl=1e-6,PropDist=10e3,Cn2=1e-16,loon=1,aniso=1.0,Rdx=5e-3):
        self.N = N                      # number of grid points per side
        self.SideLen = SideLen          # Length of one side of square phase secreen [m]
        self.dx = dx                    # Sampling interval at source plane
        self.Rdx = Rdx                  # Sampling interval at receiver plane
        self.L0 = 1e3                   # Turbulence outer scale [m]
        self.l0 = 1e-3                  # Turbulence inner scale [m]
        self.NumScr = NumScr            # Number of screens Turn into input variable
        self.DRx = DRx                  # Diameter of aperture [m]
        self.wvl = wvl                  # Wavelength [m]
        self.PropDist = PropDist        # Propagation distance(Path Length) [m]
        self.Cn2 = Cn2                  
        self.simOption = simOption      # Simulation type (i.e. spherical, plane)
        self.theta = 0                  # Angle of anisotropy [deg]
        self.aniso = aniso              # Anisotropy magnitude
        self.alpha = 22.0               # Power Law exponent 22 = 11/3 (Kolmogorov)
        self.k = 2*pi / self.wvl        # Optical wavenumber [rad/m]
        self.NumSubHarmonics = 5;       # Number of subharmonics 
        self.DTx = 0.1;                  # Transmitting aperture size for Gauss [m]
        self.w0 = (np.exp(-1) * self.DTx); 

     
        # Include sub-harmonic compensation?
        self.loon = loon
        # Simulation output
        self.Output = np.zeros((N,N))

        # Place holders for geometry/source variables
        self.Source = None
        self.r1 = None
        self.x1 = None
        self.y1 = None
        self.rR = None
        self.xR = None
        self.yR = None
        self.Uout = None

        x = np.linspace(-self.N/2, (self.N/2)-1, self.N) * self.dx
        y = np.linspace(-self.N/2, (self.N/2)-1, self.N) * self.dx 
        self.x1, self.y1 = np.meshgrid(x, y)
        self.r1 = np.sqrt(self.x1**2 + self.y1**2)        
        
        if simOption == 0:
            # Plane Wave source (default)
             self.Source = np.ones([self.N,self.N])                       


        elif simOption == 1:
            # Spherical Wave Propagation
            self.Source = self.pointSource() 


        elif simOption == 2:
            #Collimated Gaussian source
            self.source = self.gaussianBeam()
        
        x = np.linspace(-self.N/2, (self.N/2)-1, self.N) * self.Rdx
        y = np.linspace(-self.N/2, (self.N/2)-1, self.N) * self.Rdx 
        self.xR, self.yR = np.meshgrid(x, y)
        self.rR = np.sqrt(self.xR**2 + self.yR**2)
        
        
        # Set Propagation Geometry / Screen placement       
        self.dzProps = np.ones(self.NumScr+2)*(self.PropDist/self.NumScr)
        self.dzProps[0:2] = 0.5*(self.PropDist/self.NumScr)
        self.dzProps[self.NumScr:self.NumScr+2] = 0.5*(self.PropDist/self.NumScr)
        
        self.PropLocs = np.zeros(self.NumScr+3)
        

        for zval in range(0,self.NumScr+2):
            self.PropLocs[ zval+1 ] = self.PropLocs[zval]+self.dzProps[zval]
            
        self.ScrnLoc = np.concatenate((self.PropLocs[1:self.NumScr],
                                       np.array([self.PropLocs[self.NumScr+1]])),axis=0)
        
        self.FracPropDist = self.PropLocs/self.PropDist
        
        self.PropSampling = (self.Rdx - self.dx)*self.FracPropDist + self.dx
        
        self.SamplingRatioBetweenScreen = \
        self.PropSampling[1:len(self.PropSampling)] \
        /self.PropSampling[0:len(self.PropSampling)-1]
        
        
        # Set derived values 
        self.r0 = (0.423 * (self.k)**2 * self.Cn2 * self.PropDist)**(-3.0/5.0)
        self.r0scrn = (0.423 * ((self.k)**2) * self.Cn2 * (self.PropDist/self.NumScr))**(-3.0/5.0)
        self.log_ampl_var = 0.3075 * ((self.k)**2) * ((self.PropDist)**(11.0/6.0)) * self.Cn2
        self.phase_var = 0.78*(self.Cn2)*(self.k**2)*self.PropDist*(self.L0**(-5.0/3.0))
        self.rho_0 = (1.46 * self.Cn2 * self.k**2 * self.PropDist)**(-5.0/3.0)
        self.rytovNum = np.sqrt(1.23 * self.Cn2 * (self.k**(7/6)) * (self.PropDist**(11/6)) )
        self.rytovVar = self.rytovNum**2

        
        
    def PlaneSource(self):
        #Uniform plane wave
        plane = np.ones([self.N,self.N])  
        
        return plane
        
        
    
    def PointSource(self):
        #Schmidt Point Source
        DROI = 4.0 *self.DRx                 #Observation plane region [m]
        D1 = self.wvl * self.PropDist / DROI #Central Lobe width [m]
        R = self.PropDist                         #Radius of curvature at wavefront [m
        temp = np.exp(-1j*self.k/(2*R) * (self.r1**2)) / (D1**2)
        pt = temp * np.sinc((self.x1/D1)) * np.sinc((self.y1/D1)) * np.exp(-(self.r1/(4.0 * D1))**2)        
        return pt
        
    def FlattePointSource(self):
    
        fpt = np.exp(-(self.r1**2) / (10*( self.dx**2)) ) \
        * np.cos(-(self.r1**2) / (10*(self.dx**2)) )
        
        return fpt
        
        
    def CollimatedGaussian(self):
       
        source = np.exp(-(self.r1**2 / self.w0**2))
        
        source = source * self.MakePupil(self.DTx)
        
        #Source return
        return source

        
    def MakeSGB(self):
        #Construction of Super Gaussian Boundary
        rad = self.r1*(self.N);   
        w = 0.55*self.N
        sg = np.exp(- ((rad / w)**16.0) )
        
        return sg
        
        
  
    def MakePupil(self,D_eval):
        #Target pupil creation
        boundary1 = -(self.SideLen / 2) #sets negative coord of sidelength
        boundary2 = self.SideLen / 2 #sets positive coord of sidelength
    
        A = np.linspace(boundary1, boundary2, self.N) #creates a series of numbers evenly spaced between
        #positive and negative boundary
        A = np.array([A] * self.N) #horizontal distance map created
        
        base = np.linspace(boundary1, boundary2, self.N) #creates another linspace of numbers
        set_ones = np.ones(self.N) #builds array of length N filled with ones
        B = np.array([set_ones] * self.N) 
        for i in range(0, len(base)):
            B[i] = B[i] * base[i] #vertical distance map created
        
        A = A.reshape(self.N,self.N)
        B = B.reshape(self.N,self.N) #arrays reshaped into matrices
    
        x_coord = A**2
        y_coord = B**2
    
        rad_dist = np.sqrt(x_coord + y_coord) #now radial distance has been defined
    
        mask = []
        for row in rad_dist:
            for val in row:
                if val < D_eval:
                    mask.append(1.0)
                elif val > D_eval:
                    mask.append(0.0)
                elif val == D_eval:
                    mask.append(0.5)
        mask = np.array([mask])
        mask = mask.reshape(self.N,self.N) #mask created and reshaped into a matrix
    
        return mask #returns the pupil mask as the whole function's output
            
    def PhaseScreen(self):
        #Generate phase screens
        #potentially change generation to be 1 screen/1 km
        b = self.aniso
        c = 1.0
        
        thetar = (pi/180.0)*self.theta
        delta = self.dx #Spatial sampling rate
        
        del_f = 1.0/(self.N * delta) #Frequency grid spacing(1/m)
        
        cen = np.floor(self.N/2)
        
        na = self.alpha/6.0 #Normalized alpha value
        Bnum = gamma(na/2.0)
        Bdenom = 2.0**(2.0-na)*pi*na*gamma(-na/2.0)
        
        #c1 Striblings Consistency parameter. Evaluates to 6.88 in Kolmogorov turb.
        cone = (2.0* (8.0/(na-2.0) *gamma(2.0/(na-2.0)))**((na-2.0)/2.0))
        
        #Charnotskii/Bos generalized phase consistency parameter
        Bfac = (2.0*pi)**(2.0-na) * (Bnum/Bdenom)
        a = gamma(na-1.0)*cos(na*pi/2.0)/(4.0*pi**2.0)
        # Toselli's inner-scale intertial range consistency parameter 
        c_a = (gamma(0.5*(5.0-na))*a*2.0*pi/3.0)**(1.0/(na-5.0))

        fm = c_a/self.l0   # Inner scale frequency(1/m) 
        f0 = 1.0/self.L0   # Outer scale frequency
        # Create frequency sample grid
        fx = np.arange(-self.N/2.0, self.N/2.0) * del_f;
        fx, fy = np.meshgrid(fx,-1*fx)
        
        # Apply affine transform
        tx = fx*cos(thetar) + fy*sin(thetar)
        ty = -1.0*fx*sin(thetar) + fy*cos(thetar);
        
        # Scalar frequency grid 
        f = np.sqrt((tx**2.0)/(b**2.0) + (ty**2.0)/(c**2.0));

        # Sample Turbulence PSD
        PSD_phi = (cone * Bfac * ((b*c)**(-na/2.0)) * (self.r0scrn**(2.0-na)) * np.exp(-(f/fm)**2.0) \
        /((f**2.0 + f0**2.0)**(na/2.0)));
   
        #PSD_phi = cone*Bfac* (r0**(2-na)) * f**(-na/2)  # Kolmogorov PSD
        PSD_phi[cen,cen]=0.0;

        # Create a random field that is circular complex Guassian
        cn = (np.random.randn(self.N,self.N) + 1j*np.random.randn(self.N,self.N) )

        # Filter by turbulence PSD
        cn = cn * np.sqrt(PSD_phi)*del_f;

        # Inverse FFT
        phz_temp  = np.fft.ifft2(np.fft.fftshift(cn))*(self.N**2);

        # Phase screens 
       # phz1 = np.real(phz_temp)-np.mean(np.real(phz_temp));
        phz1 = np.real(phz_temp);

        return phz1
        
  
    def SubHarmonicComp(self,nsub):
        #Sub-Harmonic Phase screen production
        dq = 1/self.SideLen
        na = self.alpha/6.0
        
        #Weiner spectrum consistency parameter
        Bnum = gamma(na/2.0)
        Bdenom = (2**(2-na)) * pi * na * gamma(-na/2)
        Bfac = (2*pi)**(2-na) * (Bnum/Bdenom)
        
        # c1 Striblings Consistency parameter. Evaluates to 6.88 in Kolmogorov turb.
        cone = (2* (8/(na-2) * gamma(2/(na-2)))**((na-2)/2))        
        
        #Anisotropy factors
        b = self.aniso
        c=1
        f0 = 1/self.L0
        lof_phz = np.zeros((self.N,self.N))

        for Np in range(1,nsub+1):
            
            temp = np.zeros((self.N,self.N))
            
            #Subharmonic frequency
            dqp = dq/(3.0**Np)
            #Set samples
            qf = np.arange(-1,2) * dqp
            qx, qy = np.meshgrid(qf, -1*qf)

            qxp = qx * cos((pi/180)*self.theta) + qy * sin((pi/180)*self.theta)
            qyp = -1*qx*sin((pi/180)*self.theta) + qy*cos((pi/180)*self.theta)

            f = np.sqrt((qxp**2)/(b**2) + (qyp**2)/(c**2))
            #Sample PSD
            PSD_fi = cone*Bfac*((b*c)**(-na/2))*(self.r0scrn)**(2-na)*(f**2 + f0**2)**(-na/2)
            PSD_fi[1,1] = 0;
            
            #Generate normal circ complex RV
            w = np.random.randn(3,3) + 1j*np.random.randn(3,3)
            #Covariances
            cv = w * np.sqrt(PSD_fi)*dqp
            #Sum over subharmonic components
            temp_shape = np.shape(cv)
            for n in range(0, temp_shape[0]):
                for m in range(0,temp_shape[1]):
                    temp = temp + cv[m][n] * np.exp(1j*2*pi*((qxp[m][n])*self.x1 + (qyp[m][n])*self.y1))
    
            
            #Accumulate components to phase screen
            lof_phz = lof_phz + temp
        lof_phz = np.real(lof_phz) - np.mean(np.real(lof_phz))
        return lof_phz    
        
    def VacuumProp(self):
        # Vacuum propagation (included for source valiation)
        sg = self.MakeSGB() #Generates SGB
        
        
        SamplingRatio = self.SamplingRatioBetweenScreen
        
        a = self.N/2
        
        nx, ny = np.meshgrid(range(-a,a), range(-a, a))
        
        # Initial Propagation from source plane to first screen location
        P0 = np.exp(1j* (self.k/ (2*self.dzProps[0]) )  * (self.r1**2) * (1-SamplingRatio[0]) )
        
        Uin = P0 * self.Source 
        
        for pcount in range(1,len(self.PropLocs)-2):
                                
            
            UinSpec = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Uin)))  

            #Set spatial frequencies at propagation plane
            deltaf = 1/(self.N * self.PropSampling[pcount])
            fX = nx * deltaf
            fY = ny * deltaf
            fsq = fX**2 + fY**2
            
            
            #Quadratic Phase Factor
            QuadPhaseFac = np.exp( -1j * np.pi * self.wvl * self.dzProps[pcount] \
            * SamplingRatio[pcount] * fsq)
            
            Uin = np.fft.ifftshift(np.fft.ifft2( \
            np.fft.ifftshift(UinSpec * QuadPhaseFac)) ) 
            
            Uin = Uin * sg
                
        PF = np.exp(1j* ( self.k/ (2*self.dzProps[-1]) ) * (self.rR**2) * (SamplingRatio[-1]))
        
        Uout = PF * Uin

        return Uout    
        
        
        
    def SplitStepProp(self,Uin,PhaseScreenStack):
        #Propagation/Fresnel Diffraction Integral
        sg = self.MakeSGB() #Generates SGB
        
        SamplingRatio = self.SamplingRatioBetweenScreen
        
        a = self.N/2
        
        nx, ny = np.meshgrid(range(-a,a), range(-a, a))
        
        # Initial Propagation from source plane to first screen location
        P0 = np.exp(1j* (self.k/ (2*self.dzProps[0]) )  * (self.r1**2) * (1-SamplingRatio[0]) )
        
        Uin = P0 * self.Source * np.exp(1j * PhaseScreenStack[:,:,0])
        
        for pcount in range(1,len(self.PropLocs)-2):
                                
            
            UinSpec = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Uin)))  

            #Set spatial frequencies at propagation plane
            deltaf = 1/(self.N * self.PropSampling[pcount])
            fX = nx * deltaf
            fY = ny * deltaf
            fsq = fX**2 + fY**2
            
            
            #Quadratic Phase Factor
            QuadPhaseFac = np.exp( -1j * np.pi * self.wvl * self.dzProps[pcount] \
            * SamplingRatio[pcount] * fsq)
            
            Uin = np.fft.ifftshift(np.fft.ifft2( \
            np.fft.ifftshift(UinSpec * QuadPhaseFac)) ) 
            
            Uin = Uin * sg * np.exp(1j * PhaseScreenStack[:,:,pcount-1])
                
        
        PF = np.exp(1j* ( self.k/ (2*self.dzProps[-1]) ) * (self.rR**2) * (SamplingRatio[-1]))
        
        Uout = PF * Uin
        return Uout
        
    
    def TurbSim(self):       
        #initialize phase screen array
        phz = np.zeros(shape=(self.N,self.N,self.NumScr))
        phz_lo = np.zeros(shape=(self.N,self.N,self.NumScr))
        phz_hi = np.zeros(shape=(self.N,self.N,self.NumScr))      
              
        for idxscr in range(0,self.NumScr,1):
            phz_hi[:,:,idxscr] = self.PhaseScreen()
            #FFT-based phase screens

            phz_lo[:,:,idxscr] = self.SubHarmonicComp(self.NumSubHarmonics)
            #sub harmonics
           
            phz[:,:,idxscr] = self.loon * phz_lo[:,:,idxscr] + phz_hi[:,:,idxscr]
            #subharmonic compensated phase screens 
            
        #Simulating propagation
        self.Output = self.SplitStepProp(self.Source, np.exp(1j*phz))
       
    
    def SetCn2Rytov(self,UserRytov):
        # Change rytov number and variance to user specified value        
        self.rytovNum = UserRytov
        self.rytov = self.rytovNum**2;
    
        rytov_denom = 1.23*(self.k)**(7.0/6.0)*(self.PropDist)**(11.0/6.0)

        # Find Cn2
        self.Cn2 = self.rytov/rytov_denom
        
        # Set derived values 
        self.r0 = (0.423 * (self.k)**2 * self.Cn2 * self.PropDist)**(-3.0/5.0)
        self.r0scrn = (0.423 * ((self.k)**2) * self.Cn2 * (self.PropDist/self.NumScr))**(-3.0/5.0)
        self.log_ampl_var = 0.3075 * ((self.k)**2) * ((self.PropDist)**(11.0/6.0)) * self.Cn2
        self.phase_var = 0.78*(self.Cn2)*(self.k**2)*self.PropDist*(self.L0**(-5.0/3.0))
        self.rho_0 = (1.46 * self.Cn2 * self.k**2 * self.PropDist)**(-5.0/3.0)
    
    def EvalSI(self):
    
        temp_s = (np.abs(self.Output)**2) * self.makePupil(self.DRx)
        temp_s = temp_s.ravel()[np.flatnonzero(temp_s)]
        s_i = (np.mean( temp_s**2 )/ (np.mean(temp_s)**2) ) - 1 

        return s_i
