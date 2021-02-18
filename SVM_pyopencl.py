
# coding: utf-8

# In[1]:


#this code implements SVM bindary classification model
#it contains code for GPU and CPU processing
#the functions utilizing pyOpenCL are labeled with 'GPU'
#the gradient, predict and score functions each have pyOpenCL code 

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import scipy.spatial
import functools
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler

write_train = '/Users/gregscanlon/Documents/NYU/adv_python/final_project/DSGA3001_Adv_Python/train_ro.parquet'
write_test = '/Users/gregscanlon/Documents/NYU/adv_python/final_project/DSGA3001_Adv_Python/test_ru.parquet'
ylabel = 'Crime'

train = pd.read_parquet(write_train)
test = pd.read_parquet(write_test)


# In[2]:


y_train = train['Crime']
x_train = train.drop(columns=['Crime'])
x_train = np.asmatrix(x_train)
y_train = np.asarray(y_train)


# In[3]:


y_test = test['Crime']
x_test = test.drop(columns=['Crime'])
x_test = np.asmatrix(x_test)
y_test = np.asarray(y_test)


# In[4]:


#create smaller sets because the GPU cache is being overloaded according to error statements

x_test_mini = x_test[:10000]
y_test_mini = y_test[:10000]

x_test_50 = x_test[:50000]
y_test_50 = y_test[:50000]

x_test_100 = x_test[:100000]
y_test_100 = y_test[:100000]

x_test_200 = x_test[:200000]
y_test_200 = y_test[:200000]


# In[5]:


### Kernel function generators
def linear_kernel(W, X):
    """
    Computes the linear kernel between two sets of vectors.
    Args:
        W, X - two matrices of dimensions n1xd and n2xd
    Returns:
        matrix of size n1xn2, with w_i^T x_j in position i,j
    """
    return np.dot(W,np.transpose(X))

def polynomial_kernel(W, X, offset, degree):
    """
    Computes the inhomogeneous polynomial kernel between two sets of vectors
    Args:
        W, X - two matrices of dimensions n1xd and n2xd
        offset, degree - two parameters for the kernel
    Returns:
        matrix of size n1xn2, with (offset + <w_i,x_j>)^degree in position i,j
    """
    return (offset + linear_kernel(W,X)) ** degree

            
def RBF_kernel(W,X,sigma):
    """
    Computes the RBF kernel between two sets of vectors   
    Args:
        W, X - two matrices of dimensions n1xd and n2xd
        sigma - the bandwidth (i.e. standard deviation) for the RBF/Gaussian kernel
    Returns:
        matrix of size n1xn2, with exp(-||w_i-x_j||^2/(2 sigma^2)) in position i,j
    """
    V = scipy.spatial.distance.cdist(W,X,'sqeuclidean')
    return np.exp((-V)/(2*sigma**2))



# In[6]:


import pyopencl as cl
import pyopencl.array as pycl_array
from pyopencl.reduction import ReductionKernel
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
platform = cl.get_platforms()[0]  # Select the first platform [0]
device = platform.get_devices()[1]  # Select the second device on this platform [1]
context = cl.Context([device])  # Create a context with your device
queue = cl.CommandQueue(context)
#gpu_proto = pycl_array.to_device(queue, prototype_points.astype(np.float32))
                         
# Create two random pyopencl arrays
#gpu_proto = pycl_array.to_device(queue, prototype_points.astype(np.float32))

class Kernel_Machine(object):
    def __init__(self, kernel, prototype_points, weights):
        """
        Args:
            kernel(W,X) - a function return the cross-kernel matrix between rows of W and rows of X for kernel k
            prototype_points - an Rxd matrix with rows mu_1,...,mu_R
            weights - a vector of length R
        """

        self.kernel = kernel
        self.prototype_points = prototype_points
        self.weights = weights
        
    def predict(self, X):
        """
        Evaluates the kernel machine on the points given by the rows of X
        Args:
            X - an nxd matrix with inputs x_1,...,x_n in the rows
        Returns:
            Vector of kernel machine evaluations on the n points in X.  Specifically, jth entry of return vector is
                Sum_{i=1}^R w_i k(x_j, mu_i)
        """
        cross_kernel_matrix = self.kernel(X, self.prototype_points)
        #print('cross val',cross_kernel_matrix,'weight val',self.weights)
        answer = np.dot(cross_kernel_matrix,self.weights)
        #print('cpu shapes: cross_k', cross_kernel_matrix.shape,'weights:',self.weights.shape,'dot:', answer.shape)
        return answer
        
    def GPU_predict(self, X):
        """
        Evaluates the kernel machine on the points given by the rows of X
        Args:
            X - an nxd matrix with inputs x_1,...,x_n in the rows
        Returns:
            Vector of kernel machine evaluations on the n points in X.  Specifically, jth entry of return vector is
                Sum_{i=1}^R w_i k(x_j, mu_i)
        """
        cross_kernel_matrix = self.kernel(X, self.prototype_points)
        gpu_cross_kernel = pycl_array.to_device(queue, cross_kernel_matrix.astype(np.float32))
        gpu_weights = pycl_array.to_device(queue, np.ones(len(self.weights)).astype(np.float32))

        
        #dot = ReductionKernel (context, dtype_out=np.float32, neutral="0",
                                reduce_expr="a+b" , map_expr="gpu_cross_kernel[i]*gpu_weights[i]" ,
                                arguments="__global const float *gpu_cross_kernel, __global const float *gpu_weights")
                
        #my_dot = cl.array.dot(gpu_cross_kernel, gpu_weights, dtype=None, queue=None, slice=None)
        
        mid_dot = pycl_array.empty_like(gpu_cross_kernel)
        just_for_dot = pycl_array.to_device(queue, np.zeros(cross_kernel_matrix.shape[0]).astype(np.float32))
        dot = pycl_array.empty_like(just_for_dot)
        sgd_prg = cl.Program(context, """
            __kernel void the_math(__global const float *gpu_cross_kernel, __global const float *gpu_weights, __global float *mid_dot, __global float *dot)
            {
              int i = get_global_id(0);
              int j = get_global_id(0);
              mid_dot[i,j] = gpu_cross_kernel[i,j] * gpu_weights[j];
              dot[i] = dot[i] + mid_dot[i,j];
            }""").build()  # Create the OpenCL program

        sgd_prg.the_math(queue, dot.shape, None, gpu_cross_kernel.data, gpu_weights.data, mid_dot.data, dot.data)
            
        


        return dot
        #return np.dot(cross_kernel_matrix,self.weights)
    


# In[7]:


# train_soft_svm takes the numpy arrays containing 
# the measurements of x in R^{2 x d} and y in R^d, 
# the kernel, the maximum number of SGD steps T and 
# the regularization parameter Lambda. It returns 
# the corresponding Kernel_Machine  
def train_soft_svm(X, y, kernel, T, Lambda):
    
    # sgd_for_soft_svm implements the above-mentioned method from SSBD.
    # It takes the numpy arrays containing the measurements y in {-1,+1},
    # the kernel K in R^{d x d} the maximum number of steps T and 
    # the regularization parameter lambda. It returns the coefficients alpha_bar
    # of the solution w_bar represented as a linear combination  
    def sgd_for_soft_svm (y, K, T, Lambda): 
        sample_size = len(y)
        beta=[] 
        beta.append(np.zeros(sample_size))
        alpha = []
    
        for t in range(T):
            #print('beta[t] type:', type(beta[t]))
            alpha_temp= (beta[t] / (Lambda*(t+1) ))
            i = np.random.randint(sample_size)
            beta_temp = beta[t]
            temp=0
            for j in range (sample_size): 
                #print('alpha_temp[j] type:',type(alpha_temp[j]))
                #print('K[j,i] type:',type(K[j,i]))
                #print('temp type:',type(temp))
                temp = temp+ alpha_temp[j]* K[j,i]
            if ((y[i]*temp)<1):
                beta_temp[i]= beta_temp[i]+y[i]         
            beta.append(beta_temp)
            alpha.append(alpha_temp)
        alpha_bar = 1.0 / T * np.sum(alpha, axis=0)                  
        return alpha_bar
    
    K = kernel(X,X)
    #print('K type:',type(K))
    alpha= sgd_for_soft_svm (y, K, T, Lambda)
    #print('alpha type:',type(alpha),'alpha value:',alpha)
    return Kernel_Machine(kernel, X, alpha)


# In[8]:


# train_soft_svm takes the numpy arrays containing 
# the measurements of x in R^{2 x d} and y in R^d, 
# the kernel, the maximum number of SGD steps T and 
# the regularization parameter Lambda. It returns 
# the corresponding Kernel_Machine  
def train_soft_svm_fast(X, y, kernel, T, Lambda):
    
    # sgd_for_soft_svm implements the above-mentioned method from SSBD.
    # It takes the numpy arrays containing the measurements y in {-1,+1},
    # the kernel K in R^{d x d} the maximum number of steps T and 
    # the regularization parameter lambda. It returns the coefficients alpha_bar
    # of the solution w_bar represented as a linear combination  
    def sgd_for_soft_svm (y, K, T, Lambda): 
        sample_size = len(y)
        beta=[] 
        beta.append(np.zeros(sample_size))
        alpha = []
    
        for t in range(T):
            alpha_temp= (beta[t] / (Lambda*(t+1) )) #np float64
            i = np.random.randint(sample_size) #np array
            beta_temp = beta[t] #np array
            temp=0
            
            gpu_a_temp = pycl_array.to_device(queue, alpha_temp.astype(np.float32))
            gpu_k = pycl_array.to_device(queue, K[:,i].astype(np.float32))
            mid_temp = pycl_array.empty_like(gpu_a_temp)
            
            sgd_prg = cl.Program(context, """
            __kernel void the_math(__global const float *gpu_a_temp, __global const float *gpu_k, __global float *temp)
            {
              int i = get_global_id(0);
              temp[i] = gpu_a_temp[i] + gpu_k[i];
            }""").build()  # Create the OpenCL program

            sgd_prg.the_math(queue, gpu_a_temp.shape, None, gpu_a_temp.data, gpu_k.data, mid_temp.data)  

            gpu_temp = cl.array.sum(mid_temp, dtype=None, queue=None, slice=None)
            gpu_np_temp = gpu_temp.map_to_host()
            for j in range (sample_size): 
                temp = temp+ alpha_temp[j]* K[j,i] #temp np float64, alpha_temp[j] np float 64, K[j,i] np float64
            
            if ((y[i]*temp)<1):
                beta_temp[i]= beta_temp[i]+y[i]         
            beta.append(beta_temp)
            alpha.append(alpha_temp)
        alpha_bar = 1.0 / T * np.sum(alpha, axis=0)                  
        return alpha_bar
    
    K = kernel(X,X)
    alpha= sgd_for_soft_svm (y, K, T, Lambda)
    return Kernel_Machine(kernel, X, alpha)


# In[9]:


from sklearn.base import BaseEstimator, RegressorMixin
class KernelSoftSVM(BaseEstimator, RegressorMixin):  
    """sklearn wrapper for our kernel soft SVM"""
     
    def __init__(self, kernel="RBF", sigma=1, degree=2, offset=1, T = 100, l2reg=0.2):        
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.offset = offset
        self.l2reg = l2reg 
        self.T = T

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.
        """
        if (self.kernel == "linear"):
            self.k = linear_kernel
        elif (self.kernel == "RBF"):
            self.k = functools.partial(RBF_kernel, sigma=self.sigma)
        elif (self.kernel == "polynomial"):
            self.k = functools.partial(polynomial_kernel, offset=self.offset, degree=self.degree)
        else:
            raise ValueError('Unrecognized kernel type requested.')
        
        self.kernel_machine_ = train_soft_svm(X, y, self.k, self.T, self.l2reg)

        return self
    
    def GPU_fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.
        """
        if (self.kernel == "linear"):
            self.k = linear_kernel
        elif (self.kernel == "RBF"):
            self.k = functools.partial(RBF_kernel, sigma=self.sigma)
        elif (self.kernel == "polynomial"):
            self.k = functools.partial(polynomial_kernel, offset=self.offset, degree=self.degree)
        else:
            raise ValueError('Unrecognized kernel type requested.')
        
        self.kernel_machine_ = train_soft_svm_fast(X, y, self.k, self.T, self.l2reg)

        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "kernel_machine_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        
        return(self.kernel_machine_.predict(X))
        
    def score(self, X, y=None):
        # get the average square error
        return (self.predict(X)-y).mean()
    
    def GPU_predict(self, X, y=None):
        try:
            getattr(self, "kernel_machine_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return(self.kernel_machine_.GPU_predict(X))
        
    def GPU_score(self, X, y=None):
        # get the average square error
        n = len(y)
        gpu_pred_x = self.GPU_predict(X)
        #gpu_pred_x = pycl_array.to_device(queue, pred_x.astype(np.float32))
        gpu_pred_y = pycl_array.to_device(queue, y.astype(np.float32))
        diff = pycl_array.empty_like(gpu_pred_y)  # Create an empty pyopencl destination array
        sub_prg = cl.Program(context, """
        __kernel void diff(__global const float *gpu_pred_x, __global const float *gpu_pred_y, 
        __global float *diff)
        {
          int i = get_global_id(0);
          diff[i] = (gpu_pred_x[i] - gpu_pred_y[i]);
        }""").build()  # Create the OpenCL program


        sub_prg.diff(queue, gpu_pred_y.shape, None, gpu_pred_x.data, gpu_pred_y.data, diff.data)
        
        total_diff = cl.array.sum(diff, dtype=None, queue=None, slice=None)
        return total_diff.map_to_host()
        #return (self.predict(X)-y).mean()


# In[10]:


def average_zero_one_loss(y_true, y_pred):
    return np.mean((y_true * y_pred) <= 0)  

kernel_soft_SVM_estimator = KernelSoftSVM()


# In[ ]:


import cProfile
import re
#cProfile.run('kernel_soft_SVM_estimator.fit(x_train,y_train)')


# In[ ]:


ts = time.time()
kernel_soft_SVM_estimator.fit(x_train,y_train)
td=time.time()
print('Time: ',(td-ts))


# In[ ]:


ts = time.time()
kernel_soft_SVM_estimator.GPU_fit(x_train,y_train)
td=time.time()
print('Time: ',(td-ts)/60)


# In[334]:


ts = time.time()
s = kernel_soft_SVM_estimator.GPU_score(x_test_mini,y_test_mini)
td=time.time()
print('Time: ',(td-ts))
print(s)



# In[335]:


ts = time.time()
s = kernel_soft_SVM_estimator.score(x_test_mini,y_test_mini)
td=time.time()
print('Time: ',(td-ts))
print(s)

