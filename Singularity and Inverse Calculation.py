import numpy as np
import math
import sympy as sp 

# link lengths in mm
a1 = float(input("a1 = "))
a2 = float(input("a2 = "))
a3 = float(input("a3 = "))


# joint variables: is mm if f, is degrees if theta
T1 = float(input("t1 = "))
d2 = float(input("d2 = "))
d3 = float(input("d3 = "))

# degree to radian
T1 = (T1/180.0)*np.pi

# Parametic Table (theta, alpha, r, d)
PT = [[T1, (0.0/180.0)*np.pi, 0, a1],
      [(270.0/180.0)*np.pi, (270.0/180.0)*np.pi, 0, a2+d2],
      [(0.0/180.0)*np.pi, (0.0/180.0)*np.pi, 0, a3+d3]]


# HTM formulae
i = 0
H0_1 = [[np.cos(PT[i][0]),-np.sin(PT[i][0])*np.cos(PT[i][1]),np.sin(PT[i][0])*np.sin(PT[i][1]),PT[i][2]*np.cos(PT[i][0])],
        [np.sin(PT[i][0]),np.cos(PT[i][0])*np.cos(PT[i][1]),-np.cos(PT[i][0])*np.sin(PT[i][1]),PT[i][2]*np.sin(PT[i][0])],
        [0,np.sin(PT[i][1]),np.cos(PT[i][1]),PT[i][3]],
        [0,0,0,1]]

i = 1
H1_2 = [[np.cos(PT[i][0]),-np.sin(PT[i][0])*np.cos(PT[i][1]),np.sin(PT[i][0])*np.sin(PT[i][1]),PT[i][2]*np.cos(PT[i][0])],
        [np.sin(PT[i][0]),np.cos(PT[i][0])*np.cos(PT[i][1]),-np.cos(PT[i][0])*np.sin(PT[i][1]),PT[i][2]*np.sin(PT[i][0])],
        [0,np.sin(PT[i][1]),np.cos(PT[i][1]),PT[i][3]],
        [0,0,0,1]]

i = 2
H2_3 = [[np.cos(PT[i][0]),-np.sin(PT[i][0])*np.cos(PT[i][1]),np.sin(PT[i][0])*np.sin(PT[i][1]),PT[i][2]*np.cos(PT[i][0])],
        [np.sin(PT[i][0]),np.cos(PT[i][0])*np.cos(PT[i][1]),-np.cos(PT[i][0])*np.sin(PT[i][1]),PT[i][2]*np.sin(PT[i][0])],
        [0,np.sin(PT[i][1]),np.cos(PT[i][1]),PT[i][3]],
        [0,0,0,1]]

H0_1 = np.matrix(H0_1)
#print("H0_1= ")
#print(H0_1)

H1_2 = np.matrix(H1_2)
#rint("H1_2= ")
#print(H1_2)

H2_3 = np.matrix(H2_3)
#print("H2_3= ")
#print(H2_3)

H0_2 = np.dot(H0_1,H1_2)
H0_3 = np.dot(H0_2,H2_3)
#print("H0_3= ")
#print(np.matrix(np.around(H0_3,3)))

## Jacobian Matrix

#1. Linear / Prismatic Vectors
Z_1 = [[0],[0],[1]] # The [0,0,1] vector

#Row 1 to 3, Column 1
J1a = [[1,0,0],
      [0,1,0],
      [0,0,1]] #R0_0
J1a = np.dot(J1a,Z_1)
J1a = np.matrix(J1a)

J1b_1 = H0_3[0:3,3]
J1b_1 = np.matrix(J1b_1)

J1b_2 = [[0],
         [0],
         [0]]
J1b_2 = np.matrix(J1b_2)

J1b = J1b_1-J1b_2

J1 = [[(J1a[1,0]*J1b[2,0])-(J1a[2,0]*J1b[1,0])],
      [(J1a[2,0]*J1b[0,0])-(J1a[0,0]*J1b[2,0])],
      [(J1a[0,0]*J1b[1,0])-(J1a[1,0]*J1b[0,0])]]

print("J1 = ")
print(np.matrix(J1))

#Row 1 to 3, Column 2
J2 = H0_1[0:3,0:3]
J2 = np.dot(J2,Z_1)
J2 = np.matrix(J2)

#Row 1 to 3, Column 3
J3 = H0_2[0:3,0:3]
J3 = np.dot(J3,Z_1)
J3 = np.matrix(J3)

#2. Rotational / Orientation Vectors

#Row 4 to 6, Column 1
J4 = J1a
J4 = np.matrix(J4)

#Row 4 to 6, Column 2
J5 = [[0],
      [0],
      [0]]
J5 = np.matrix(J5)

#Row 4 to 6, Column 1
J6 = [[0],
      [0],
      [0]]
J6 = np.matrix(J6)

#3. Concatenated Jacobian Matrix
JM1 = np.concatenate((J1,J2,J3),1)
JM2 = np.concatenate((J4,J5,J6),1)

J = np.concatenate((JM1,JM2),0)
print("J = ")
print(J)

#4. Differential Equations
xp, yp, zp = sp.symbols('x* y* z*')
ωx, ωy, ωz = sp.symbols('ωx ωy ωz')
t1_p,d2_p,d3_p = sp.symbols('θ1* d2* d3*')

q = [[t1_p],[d2_p],[d3_p]]

E = np.dot(J,q)
E = np.matrix(E)

#print("E = ")
#print(E)

xp = E[0,0]
yp = E[1,0]
zp = E[2,0]
ωx = E[3,0]
ωy = E[4,0]
ωz = E[5,0]

print("xp = ",xp)
print("yp = ",yp)
print("zp = ",zp)
print("ωx = ",ωx)
print("ωy = ",ωy)
print("ωz = ",ωz)

## Singularity
D_J = np.linalg.det(JM1)
print("D_J = ",D_J)

## Inverse Velocity
I_V = np.linalg.inv(JM1)
print("I_V = ",I_V)

# Force-Torque Analysis
F_T = np.transpose(JM1)
print("F_T = ")
print(F_T)
