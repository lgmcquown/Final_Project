import numpy as np
import scipy.constants as constants
import math

#############################################################

class ChargedObject(object):
    
    def __init__(self, center):
        self.c = center
        
##############################################################

class ChargedSphere(ChargedObject):

        
    def __init__(self,center,radius,distribution):
        self.c = center
        self.rad = radius
        self.rho = distribution
    
            
    def E(self,x):
        if np.linalg.norm(x-self.c) == 0:
            E_field = 0
        if 0 < np.linalg.norm(x-self.c) < self.rad :
            E_Field = (self.rho/constants.epsilon_0)*(np.linalg.norm(x-self.c)/3)
        if np.linalg.norm(x-self.c) >= self.rad:
            E_Field = (self.rho/constants.epsilon_0)*((self.rad**3)/(3*(np.linalg.norm(x-self.c)**2)))
        
        E_hat = (x-self.c)/np.linalg.norm(x-self.c)
        
        return E_Field*E_hat
    
    def V(self,x):
        pi = 3.14159
        q = self.rho*(4/3)*pi*((self.rad)^3)
        r_perp = np.linalg.norm(x-self.c)
        if r_perp == 0:
            raise ZeroDivisionError
        if r_perp < self.rad :
            V = (q/(8*pi*constants.epsilon_0*self.rad))*(3-(r_perp/self.rad)^2)
        if r_perp >= self.rad:
            V = q/(4*pi*constants.epsilon_0*r_perp)         
        return V    
    
################################################################

class ChargedSphereShell(ChargedObject):
    
    def __init__(self,center,radius,distribution):
        self.c = center
        self.rad = radius
        self.sigma = distribution
        
                
    def E(self,x):
        if np.linalg.norm(x-self.c) < self.rad :
            E_Field = 0
        if np.linalg.norm(x-self.c) >= self.rad:
            E_Field = (self.sigma/constants.epsilon_0)*((self.rad**2)/((np.linalg.norm(x-self.c)**2)))
        
        E_hat = (x-self.c)/np.linalg.norm(x-self.c)
        return E_Field*E_hat
        
    def V(self,x):
        pi = 3.14159
        q = self.sigma*(4)*pi*((self.rad)^2)
        r_perp = np.linalg.norm(x-self.c)
        if np.linalg.norm(x-self.c) == 0:
            raise ZeroDivisionError
        if np.linalg.norm(x-self.c) < self.rad :
            V = q/(4*pi*constants.epsilon_0*self.rad)
        if np.linalg.norm(x-self.c) >= self.rad:
            V = q/(4*pi*constants.epsilon_0*r_perp)         
        return V

    
###################################################################

class ChargedCylinderShell(ChargedObject):
    def __init__(self, center, axis, radius, distribution):
        self.c = center
        self.axis = axis
        self.rad = radius
        self.sigma = distribution

        
    def E(self,x):
        axis_hat = self.axis/np.linalg.norm(self.axis)
        r_vec = x-self.c
        r_parallel = np.dot(axis_hat,r_vec)
        E_not_hat = (r_vec - (axis_hat*r_parallel))
        E_hat = E_not_hat/np.linalg.norm(E_not_hat)
        r_perp = np.linalg.norm(np.cross(axis_hat,r_vec))
        
        if r_perp <= self.rad :
            E_Field = 0
        if r_perp > self.rad:
            E_Field = (self.sigma/constants.epsilon_0)*((self.rad)/((r_perp)))
        return E_Field*E_hat
    
    def V(self,x):
        axis_hat = self.axis/np.linalg.norm(self.axis)
        r_vec = x-self.c
        r_parallel = np.dot(axis_hat,r_vec)
        E_not_hat = (r_vec - (axis_hat*r_parallel))
        E_hat = E_not_hat/np.linalg.norm(E_not_hat)
        r_perp = np.linalg.norm(np.cross(axis_hat,r_vec))
        pi = 3.14159
        a = 1000 #This is an arbitrary choice
        
        if r_perp <= self.rad :
            V_field = (self.sigma/4*constants.epsilon_0)*(self.rad**2 -r_perp**2)
        if r_perp > self.rad :
            V_field = (self.sigma*self.rad/2*pi*constants.epsilon_0)*math.log(r_perp/a)
            
        return V_field

    
#######################################################################

class ChargedCylinder(ChargedObject):
    
    def __init__(self,center, axis, radius, distribution) :
        self.c = center
        self.rad = radius
        self.axis = axis
        self.rho = distribution
                
    def E(self,x):
        axis_hat = self.axis/np.linalg.norm(self.axis)
        r_vec = x-self.c
        r_parallel = np.dot(axis_hat,r_vec)
        E_not_hat = (r_vec - (axis_hat*r_parallel))
        E_hat = E_not_hat/np.linalg.norm(E_not_hat)
        r_perp = np.linalg.norm(np.cross(axis_hat,r_vec))
  
        if r_perp == 0 :
            return np.array([0,0,0])
        if 0 < r_perp <= self.rad :
            E_Field = (self.rho/constants.epsilon_0)*(r_perp/2)
        if r_perp > self.rad :
            E_Field = (self.rho/constants.epsilon_0)*((self.rad**2)/(2*r_perp))
        
        return E_Field*E_hat
    
    def V(self,x):
        axis_hat = self.axis/np.linalg.norm(self.axis)
        r_vec = x-self.c
        r_parallel = np.dot(axis_hat,r_vec)
        E_not_hat = (r_vec - (axis_hat*r_parallel))
        E_hat = E_not_hat/np.linalg.norm(E_not_hat)
        r_perp = np.linalg.norm(np.cross(axis_hat,r_vec))
        pi = 3.14159
        q = self.rho*pi*((self.rad)^2) 
        a = 1000 #this is an arbitrary reference choice and either 0 or infinity would lead to problems
        
        if r_perp <= self.rad:
            V_field = (self.rho/4*constants.epsilon_0)*(self.rad**2 - r_perp**2)
        if r_perp > self.rad:
            V_field = q/(2*pi*constants.epsilon_0)*math.log(r_perp/a)
        return V_field 

    
############################################################################

class ChargedPlane(ChargedObject):
    
    def __init__(self,center, distribution,normal):
        class Infinite_Plane(object):
    
            self.normal = normal
            self.sigma = distribution
            self.c = center
        
    def E(self,x):
        E_Field = self.sigma/(2*constants.epsilon_0)
        if np.dot(x-self.c,self.normal) == 0:
            E_hat = np.array([0,0,0])
        if np.dot(x-self.c,self.normal) > 0:
            E_hat = self.normal/np.linalg.norm(self.normal)
        if np.dot(x-self.c,self.normal) < 0:
            E_hat = -self.normal/np.linalg.norm(self.normal)
        return E_Field*E_hat        
        
    def V(self,x):
        a = 1000 #this is an arbitrary choice as infinity would cause problems.
        r_vec = (x -self.c)
        perp_distance = np.dot(self.normal,r_vec)
        V_field = np.linalg.norm(self.E(x))*(perp_distance-a)
        return V_field
        
##############################################################################

def E_Superposition(List_of_objects,x):
    Total_E = 0
    for obj in List_of_objects:
        Total_E += obj.E(x)
    return Total_E

##############################################################################

def V_Superposition(List_of_objects,x):
    Total_V = 0
    for obj in List_of_objects:
        Total_V += obj.V(x)
    return Total_V

##############################################################################

def Sweep_Points(List_of_Objects,x_range,y_range,z_range):
    import Charged_Objects
    Xpoints = list(range(x_range[0],x_range[1]+1))
    Ypoints = list(range(y_range[0],y_range[1]+1))
    Zpoints = list(range(z_range[0],z_range[1]+1))
    I = len(Xpoints)
    J = len(Ypoints)
    K = len(Zpoints)
    NumElements = I*J*K*4
    Array = np.array(range(NumElements))
    DataTensor = Array.reshape(I,J,K,4)
    for i in range(0,I):
        x = Xpoints[i]
        for j in range(0,J):
            y = Ypoints[j]
            for k in range(0,K):
                z = Zpoints[k]
                E = E_Superposition(List_of_Objects,[x,y,z])
                V = V_Superposition(List_of_Objects,[x,y,z])
                DataTensor[i,j,k,:] = [E[0],E[1],E[2],V]
    return DataTensor
