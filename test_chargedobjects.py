import nose
import Charged_Objects as obj
import numpy as np


##########################################

def test_xdirection_sphere():
    A = obj.ChargedSphere(np.array([0,0,0]),1,.00000001)
    E = A.E(np.array([2,0,0]))
    obs = E/np.linalg.norm(E)
    exp = np.array([1,0,0])
    assert np.array_equal(obs,exp)  == True 

def test_ydirection_sphere():
    A = obj.ChargedSphere(np.array([0,0,0]),1,.00000001)
    E = A.E(np.array([0,0,2]))
    obs = E/np.linalg.norm(E)
    exp = np.array([0,0,1])
    assert np.array_equal(obs,exp)  == True


def test_zdirection_sphere():
    A = obj.ChargedSphere(np.array([0,0,0]),1,.00000001)
    E = A.E(np.array([0,0,2]))
    obs = E/np.linalg.norm(E)
    exp = np.array([0,0,1])
    assert np.array_equal(obs,exp)  == True


def test_arbitrary_direction_sphere():
    A = obj.ChargedSphere(np.array([0,0,0]),1,.00000001)
    E = A.E(np.array([1,1,1]))
    obs = E/np.linalg.norm(E)
    exp = np.array([1,1,1])/np.linalg.norm([1,1,1])
    assert np.array_equal(obs,exp) == True


#######################################################################

def test_xdirection_cylinder() :
    C = obj.ChargedCylinder(np.array([0,0,0]), np.array([0,0,1]),1,.000000001)    
    E = C.E(np.array([2,0,0]))
    obs = E/np.linalg.norm(E)
    exp = np.array([1,0,0])
    assert np.array_equal(obs,exp)  == True


def test_ydirection_cylinder() :
    C = obj.ChargedCylinder(np.array([0,0,0]), np.array([0,0,1]),1,.000000001)
    E = C.E(np.array([0,2,0]))
    obs = E/np.linalg.norm(E)
    exp = np.array([0,1,0])
    assert np.array_equal(obs,exp)  == True

def test_zdirection_cylinder() :
    C = obj.ChargedCylinder(np.array([0,0,0]), np.array([0,0,1]),1,.000000001)
    E = C.E(np.array([0,0,2]))
    obs = E
    exp = np.array([0,0,0])
    assert np.array_equal(obs,exp)  == True

def test_arbdirection_cylinder() :
    C = obj.ChargedCylinder(np.array([0,0,0]), np.array([1,1,0]),1,.000000001)
    E = C.E(np.array([1,1,2]))
    obs = E/np.linalg.norm(E)
    exp = np.array([0,0,1])
    assert np.array_equal(obs,exp)  == True

#If you run this code, you'll find that it returns [1e-16,1e-16,1], which is essentially [0,0,1]

#############################################################################################

def test_normal_direction_plane() :
    C = obj.ChargedPlane(np.array([0,0,0]),.000000000001,np.array([1,0,0]))
    E = C.E(np.array([1,0,0]))
    obs = E/np.linalg.norm(E)
    exp = np.array([1,0,0])
    assert np.array_equal(obs,exp) == True

def test_negnormal_direction_plane() :
    C = obj.ChargedPlane(np.array([0,0,0]),.000000000001,np.array([1,0,0]))
    E = C.E(np.array([-1,0,0]))
    obs = E/np.linalg.norm(E)
    exp = np.array([-1,0,0])
    assert np.array_equal(obs,exp) == True

def test_arb_orientation_plane() :
    C = obj.ChargedPlane(np.array([0,0,0]),.000000000001,np.array([1,1,1]))
    E = C.E(np.array([1,0,0]))
    obs = E/np.linalg.norm(E)
    exp = np.array([1,1,1])/np.linalg.norm(np.array([1,1,1]))
    assert np.array_equal(obs,exp) == True

def test_arb_position_plane() :
    C = obj.ChargedPlane(np.array([0,0,0]),.000000000001,np.array([1,0,0]))
    E = C.E(np.array([1,7,2]))
    obs = E/np.linalg.norm(E)
    exp = np.array([1,0,0])
    assert np.array_equal(obs,exp) == True

#################################################################
#the codes for the shell objects are all very similar, so I won't bother testing them

