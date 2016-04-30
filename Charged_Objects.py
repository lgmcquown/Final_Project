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
        if np.linalg.norm(x-self.c) == 0:
            raise ZeroDivisionError
        if np.linalg.norm(x-self.c) < self.rad :
            V = (self.rho/6*constants.epsilon_0)*(3 - ((np.linalg.norm(x-self.c)**2)/self.rad**2))
        if np.linalg.norm(x-self.c) >= self.rad:
            V = (self.rho*((self.rad)^3)/((3*constants.epsilon_0)*(np.linalg.norm(x-self.c))))         
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
        if np.linalg.norm(x-self.c) == 0:
            raise ZeroDivisionError
        if np.linalg.norm(x-self.c) < self.rad :
            V = (self.sigma/6*constants.epsilon_0)*(3 - ((np.linalg.norm(x-self.c)**2)/self.rad**2))
        if np.linalg.norm(x-self.c) >= self.rad:
            V = (self.sigma*((self.rad)^3)/((3*constants.epsilon_0)*(np.linalg.norm(x-self.c))))         
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
        
        if r_perp <= self.rad :
            V_field = 0
        if r_perp > self.rad :
            V_field = -(self.sigma*self.rad/constants.epsilon_0)*math.log(r_perp/self.rad)
            
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
        
        
        if 0 == r_perp:
            V_field = 0
        if 0 < r_perp <= self.rad:
            V_field = -(self.rho/4*constants.epsilon_0)*((r_perp)^2)
        if r_perp > self.rad:
            V_field = -(self.rho/(2*constants.epsilon_0))*((self.rho**2))*math.log(r_perp/self.rad)
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
        r_vec = (x -self.c)
        perp_distance = np.dot(self.normal,r_vec)
        V_field = -np.linalg.norm(self.E(self.sigma))*perp_distance
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
