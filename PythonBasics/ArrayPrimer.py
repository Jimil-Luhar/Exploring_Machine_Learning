import numpy as np #short for Numerical Python.
Array=np.array([1,2,3,4,5])
Matrix=np.array([[1,2,3],[4,5,9],[7,8,9]])

from numpy.linalg import inv,matrix_rank,det,eig
Trace=np.trace(Matrix)
Rank=matrix_rank(Matrix)
Transpose=np.transpose(Matrix)
Inverse=inv(Matrix)
Determinant=det(Matrix)
Eg,Ev=eig(Matrix)
#Print to verify the result. e.g, print('Trace=',Trace)

print(Matrix[1,2]) # Extract a particular element from matrix
print(Matrix[:,0]) # Python will automatically count up to the extreme for whitespaces or ':'
