import torch
import torchvision.models as models
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import cv2 as cv

import PIL
import PIL.Image

import scipy
from scipy import io
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from scipy.optimize import minimize
from numba import jit, float64, int64
import numpy as np
import matplotlib.pyplot as plt

IT_multi=io.loadmat('Downloads/DiCarlo2014Project/PLoSCB2014_data_20141216/NeuralData_IT_multiunits.mat')
IT_single=io.loadmat('Downloads/DiCarlo2014Project/PLoSCB2014_data_20141216/NeuralData_IT_singleunits.mat')
V4_multi=io.loadmat('Downloads/DiCarlo2014Project/PLoSCB2014_data_20141216/NeuralData_V4_multiunits.mat')
V4_single=io.loadmat('Downloads/DiCarlo2014Project/PLoSCB2014_data_20141216/NeuralData_V4_singleunits.mat')

AlexNet = io.loadmat('Downloads/DiCarlo2014Project/20150218/Models_Krizhevsky2012.mat')
ZFNet = io.loadmat('Downloads/DiCarlo2014Project/20150218/Models_ZeilerFergus2013.mat')
HMO = io.loadmat('Downloads/DiCarlo2014Project/20150218/Models_HMO.mat')

resnext101 = models.resnext101_32x8d(pretrained=True)
inception = models.inception_v3(pretrained=True)

avgpool_layer_resnext101 = resnext101._modules.get('avgpool')
avgpool_layer_inception = inception._modules.get('avgpool')

resnext101.eval()
inception.eval()

preprocess = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def copy_data_res(m, i, o):
    my_embedding_res.copy_((o.data.reshape(o.data.size(1))))
def copy_data_inc(m, i, o):
    my_embedding_inc.copy_((o.data.reshape(o.data.size(1))))

my_embedding_res = torch.zeros([2048])
my_embedding_inc = torch.zeros([2048])
ResNext_X = torch.zeros([1960, 2048])
Inception_X = torch.zeros([1960, 2048])

for i in range(1960):
    
    img_str = 'Downloads/DiCarlo2014Project/PLoSCB2014_data_20141216/' + str(IT_multi['meta'][i])[:51]
    img = PIL.Image.open(img_str)
    input_tensor = preprocess(img)
    input_batch = Variable(input_tensor.unsqueeze(0))

    h_res = avgpool_layer_resnext101.register_forward_hook(copy_data_res)
    h_inc = avgpool_layer_inception.register_forward_hook(copy_data_inc)
       
    resnext101(input_batch)
    inception(input_batch)

    h_res.remove()
    h_inc.remove()
    
    ResNext_X[i, ] = my_embedding_res
    Inception_X[i, ] = my_embedding_inc
    print(i)

np.savetxt('Downloads/DiCarlo2014Project/Results/ResNext_X.txt', ResNext_X)
np.savetxt('Downloads/DiCarlo2014Project/Results/Inception_X.txt', Inception_X)

ResNext = np.loadtxt('Downloads/DiCarlo2014Project/Results/ResNext_X.txt')
Inception = np.loadtxt('Downloads/DiCarlo2014Project/Results/Inception_X.txt')

def gaussian_kernel(X, sigma, n_components, PCA = False): # potentially can be used with PCA
    
    """
    Gaussian kernel implementation.    
    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_examples, n_features]  
    lambda, sigma: float
        Tuning parameter of the kernel    
    n_components: int
        Number of principal components to return    
    Returns
    ------------
    X_pc: {NumPy ndarray}, shape = [n_examples, k_features]
        Projected dataset   
    """

    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')    
    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)    
    # Compute the symmetric kernel matrix.
    K = np.exp(-mat_sq_dists/(2*sigma**2))    

    if PCA:
        # Center the kernel matrix.
        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)    
        # Obtaining eigenpairs from the centered kernel matrix
        # scipy.linalg.eigh returns them in ascending order
        eigvals, eigvecs = eigh(K)
        eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]    
        # Collect the top k eigenvectors (projected examples)
        X_pc = np.column_stack([eigvecs[:, i]
                            for i in range(n_components)])    
        return X_pc
    else:
        return K


def Theta(K, lmbda, Y):

    return np.matmul(np.linalg.inv(K + lmbda * np.identity(K.shape[0])), Y)

def looe(lmbda, K, Y):

    theta = Theta(K, lmbda, Y)
    loo_error = theta / np.diagonal(np.linalg.inv(K+lmbda * np.identity(K.shape[0])))

    return loo_error

def max_precision(X, Y, lmbda):

    # This is too slow! Even though this is used in the paper.
    sq_dists = pdist(X, 'sqeuclidean')    
    mat_sq_dists = squareform(sq_dists)
    sigma_m = mat_sq_dists.std()
    bnds = ((.01*sigma_m, sigma_m),) #[.1*sigma_m], [10*sigma_m])

    def min_looe(sigma):

        K = gaussian_kernel(X, sigma, 1)
        LOOE = looe(lmbda, K, Y)

        return (LOOE**2).mean()

    def grad_ml(sigma):
        h = 1e-9
        grad_sig = np.zeros_like(sigma)
        
        f_0 = min_looe(sigma)
        x_d = np.copy(sigma)
        x_d += h
        f_d = min_looe(x_d)
        grad_sig = (f_d - f_0) / h
        
        return grad_sig

    min_res = minimize(min_looe, sigma_m, method='SLSQP', options={'maxiter': 1e4}, jac=grad_ml, bounds=((0.001*sigma_m, .1*sigma_m),))
    max_prec = 1-min_res.fun

    return max_prec


# here we sub-sampling the images 10 times.
categories = ['Animals','Cars','Chairs','Faces','Fruits','Planes','Tables']
lmbda = np.logspace(np.log10(.001),np.log10(10000), 28) 
prec_ResNext = np.zeros((10, 28))
prec_Inception = np.zeros((10, 28))
prec_AlexNet = np.zeros((10, 28))
prec_ZFNet = np.zeros((10, 28))
prec_HMO = np.zeros((10, 28))

for s in range(10): 

    features = 0
    f_ResNext = np.zeros((1568, 2048)) # 80% of the whole imageset.
    f_Inception = np.zeros((1568, 2048))
    f_AlexNet = np.zeros((1568, 4096))
    f_ZFNet = np.zeros((1568, 4096))
    f_HMO = np.zeros((1568, 1250))

    Y_c = np.zeros((7, 1568))

    for i in range(1960):
        img_cat_sample = str(IT_multi['meta'][i])[52:]
        
        for c in range(7):
            if categories[c] in img_cat_sample:
                if img_cat_sample[len(categories[c])+1+2*s] == '1':
                    
                    f_ResNext[features,:] = ResNext[i,]
                    f_Inception[features,:] = Inception[i,]
                    f_AlexNet[features,:] = AlexNet['features'][i,]
                    f_ZFNet[features,:] = ZFNet['features'][i,]
                    f_HMO[features,:] = HMO['features'][i,]
                    
                    Y_c[c,features] = 1
                    
                    features += 1
                    
    # Normalize the labels.
    # This is super important!
    for c in range(7):
        Y_c[c,] = Y_c[c,] * np.sqrt(1568./Y_c[c,].sum())

    print('features gathered')        

    for l in range(28):
        for c in range(7):

            prec_ResNext[s, l] += max_precision(f_ResNext, Y_c[c,], lmbda[l])
            prec_Inception[s, l] += max_precision(f_Inception, Y_c[c,], lmbda[l])
            prec_AlexNet[s, l] += max_precision(f_AlexNet, Y_c[c,], lmbda[l])
            prec_ZFNet[s, l] += max_precision(f_ZFNet, Y_c[c,], lmbda[l])
            prec_HMO[s, l] += max_precision(f_HMO, Y_c[c,], lmbda[l])

            print('category', c)
        
        # average over 7 categories.
        prec_ResNext[s, l] /= 7.
        prec_Inception[s, l] /= 7.
        prec_AlexNet[s, l] /= 7.
        prec_ZFNet[s, l] /= 7.
        prec_HMO[s, l] /= 7.

        print('lambda', l)

np.savetxt('Downloads/DiCarlo2014Project/Results/sampling_ResNext_0-3.txt', prec_ResNext)
np.savetxt('Downloads/DiCarlo2014Project/Results/sampling_Inception_0-3.txt', prec_Inception)
np.savetxt('Downloads/DiCarlo2014Project/Results/sampling_AlexNet_0-3.txt', prec_AlexNet)
np.savetxt('Downloads/DiCarlo2014Project/Results/sampling_ZFNet_0-3.txt', prec_ZFNet)
np.savetxt('Downloads/DiCarlo2014Project/Results/sampling_HMO_0-3.txt', prec_HMO)

# For Neuraonal Data.

# here we sub-sampling the images 10 times.
categories = ['Animals','Cars','Chairs','Faces','Fruits','Planes','Tables']
lmbda = np.logspace(np.log10(.001),np.log10(10000), 28) 
prec_IT = np.zeros((10, 28))
prec_V4 = np.zeros((10, 28))

for s in range(10): 

    features = 0
    f_ResNext = np.zeros((1568, 2048)) # 80% of the whole imageset.
    f_Inception = np.zeros((1568, 2048))

    Y_c = np.zeros((7, 1568))

    for i in range(1960):
        img_cat_sample = str(IT_multi['meta'][i])[52:]
        
        for c in range(7):
            if categories[c] in img_cat_sample:
                if img_cat_sample[len(categories[c])+1+2*s] == '1':
                    
                    f_ResNext[features,:] = ResNext[i,]
                    f_Inception[features,:] = Inception[i,]
                    
                    Y_c[c,features] = 1
                    
                    features += 1

    print('features gathered')
              

    for l in range(28):
        for c in range(7):

            prec_ResNext[s, l] += max_precision(f_ResNext, Y_c[c,], lmbda[l])
            prec_Inception[s, l] += max_precision(f_Inception, Y_c[c,], lmbda[l])

            print('category', c)
        
        # average over 7 categories.
        prec_ResNext[s, l] /= 7.
        prec_Inception[s, l] /= 7.

        print('lambda', l)


Alex_Prec = np.zeros((10, 28))
ZF_Prec = np.zeros((10, 28))
HMO_Prec = np.zeros((10, 28))

ResNext_Prec = np.zeros((10, 28))
Inception_Prec = np.zeros((10, 28))

Alex_Prec = np.loadtxt('Downloads/DiCarlo2014Project/Results/sampling_AlexNet_0-3.txt')
ZF_Prec = np.loadtxt('Downloads/DiCarlo2014Project/Results/sampling_ZFNet_0-3.txt')
HMO_Prec = np.loadtxt('Downloads/DiCarlo2014Project/Results/sampling_HMO_0-3.txt')
ResNext_Prec = np.loadtxt('Downloads/DiCarlo2014Project/Results/sampling_ResNext_0-3.txt')
Inception_Prec = np.loadtxt('Downloads/DiCarlo2014Project/Results/sampling_Inception_0-3.txt')

IT_Prec = np.zeros((10, 28))
V4_Prec = np.zeros((10, 28))

IT_Prec = np.loadtxt('Downloads/DiCarlo2014Project/Results/sampling_IT_multi_4-6.txt')
V4_Prec = np.loadtxt('Downloads/DiCarlo2014Project/Results/sampling_V4_multi_4-6.txt')

lmbda = np.logspace(np.log10(.001),np.log10(10000), 28)
complexity = 1./lmbda

plt.plot(complexity, ResNext_Prec.mean(axis=0), 'b--', label='ResNext')
plt.plot(complexity, Inception_Prec.mean(axis=0), 'c--', label='Inception')
plt.fill_between(complexity, ResNext_Prec.mean(axis=0) - 3*ResNext_Prec.std(axis=0), ResNext_Prec.mean(axis=0) + 3*ResNext_Prec.std(axis=0), color='#888888')
plt.fill_between(complexity, Inception_Prec.mean(axis=0) - 3*Inception_Prec.std(axis=0), Inception_Prec.mean(axis=0) + 3*Inception_Prec.std(axis=0), color='#888888')

plt.plot(complexity, Alex_Prec.mean(axis=0), 'r--', label='Krizhevsky et al. 2012')
plt.plot(complexity, ZF_Prec.mean(axis=0), 'g--', label='Zeiler & Fergus 2013')
plt.plot(complexity, HMO_Prec.mean(axis=0), 'y--', label='HMO')
plt.fill_between(complexity, Alex_Prec.mean(axis=0) - 3*Alex_Prec.std(axis=0), Alex_Prec.mean(axis=0) + 3*Alex_Prec.std(axis=0), color='#888888')
plt.fill_between(complexity, ZF_Prec.mean(axis=0) - 3*ZF_Prec.std(axis=0), ZF_Prec.mean(axis=0) + 3*ZF_Prec.std(axis=0), color='#888888')
plt.fill_between(complexity, HMO_Prec.mean(axis=0) - 3*HMO_Prec.std(axis=0), HMO_Prec.mean(axis=0) + 3*HMO_Prec.std(axis=0), color='#888888')

plt.plot(complexity, IT_Prec.mean(axis=0), 'k--', label='IT Multi')
plt.plot(complexity, V4_Prec.mean(axis=0), 'p--', label='V4 Multi')
plt.fill_between(complexity, IT_Prec.mean(axis=0) - 3*IT_Prec.std(axis=0), IT_Prec.mean(axis=0) + 3*IT_Prec.std(axis=0), color='#888888')
plt.fill_between(complexity, V4_Prec.mean(axis=0) - 3*V4_Prec.std(axis=0), V4_Prec.mean(axis=0) + 3*V4_Prec.std(axis=0), color='#888888')
plt.legend(loc='upper left')

plt.xscale('log')
plt.xlabel('complexity')
plt.ylabel('Precision')

normalize = 1
sklearn.preprocessing.normalize(ResNext, axis=normalize, copy=False)
sklearn.preprocessing.normalize(Inception, axis=normalize, copy=False)
AlexNet = sklearn.preprocessing.normalize(AlexNet['features'], axis=normalize)
ZFNet = sklearn.preprocessing.normalize(ZFNet['features'], axis=normalize)
HMO = sklearn.preprocessing.normalize(HMO['features'], axis=normalize)
V4 = sklearn.preprocessing.normalize(V4_multi['features'], axis=normalize)

IT_multi = sklearn.preprocessing.normalize(IT_multi['features'], axis=normalize)

ZF_explained_v = np.zeros((10, 168))

for i in range(10):
    # one split.
    
    # fit to IT Multi.
    X_train, X_test, y_train, y_test = train_test_split(ZFNet, IT_multi,test_size=0.2,random_state=42+i)
    
    for site in range(168):
        ridge_ZF = linear_model.Ridge(1.) # linear_model.SGDRegressor()
        ridge_ZF.fit(X_train, y_train[:,site])

        ZF_explained_v[i,site] = explained_variance_score(y_test[:,site], ridge_ZF.predict(X_test))
        
    print('done!')

Alex_explained_v = np.zeros((10, 168))

for i in range(10):
    # one split.
    
    # fit to IT Multi.
    X_train, X_test, y_train, y_test = train_test_split(AlexNet, IT_multi,test_size=0.2,random_state=42+i)
    
    for site in range(168):
        ridge_Alex = linear_model.Ridge(1.) # linear_model.SGDRegressor()
        ridge_Alex.fit(X_train, y_train[:,site])

        Alex_explained_v[i,site] = explained_variance_score(y_test[:,site], ridge_Alex.predict(X_test))
        
    print('done!')

HMO_explained_v = np.zeros((10, 168))

for i in range(10):
    # one split.
    
    # fit to IT Multi.
    X_train, X_test, y_train, y_test = train_test_split(HMO, IT_multi,test_size=0.2,random_state=42+i)
    
    for site in range(168):
        ridge_HMO = linear_model.Ridge(1.) # linear_model.SGDRegressor()
        ridge_HMO.fit(X_train, y_train[:,site])

        HMO_explained_v[i,site] = explained_variance_score(y_test[:,site], ridge_HMO.predict(X_test))
        
    print('done!')

ResNext_explained_v = np.zeros((10, 168))

for i in range(10):
    # one split.
    
    # fit to IT Multi.
    X_train, X_test, y_train, y_test = train_test_split(ResNext, IT_multi,test_size=0.2,random_state=42+i)
    
    for site in range(168):
        ridge_ResNext = linear_model.Ridge(1.) # linear_model.SGDRegressor()
        ridge_ResNext.fit(X_train, y_train[:,site])

        ResNext_explained_v[i,site] = explained_variance_score(y_test[:,site], ridge_ResNext.predict(X_test))
        
    print('done!')

Inception_explained_v = np.zeros((10, 168))

for i in range(10):
    # one split.
    
    # fit to IT Multi.
    X_train, X_test, y_train, y_test = train_test_split(Inception, IT_multi,test_size=0.2,random_state=42+i)
    
    for site in range(168):
        ridge_Inception = linear_model.Ridge(1.) # linear_model.SGDRegressor()
        ridge_Inception.fit(X_train, y_train[:,site])

        Inception_explained_v[i,site] = explained_variance_score(y_test[:,site], ridge_Inception.predict(X_test))
        
    print('done!')

V4_explained_v = np.zeros((10, 168))

for i in range(10):
    # one split.
    
    # fit to IT Multi.
    X_train, X_test, y_train, y_test = train_test_split(V4, IT_multi,test_size=0.2,random_state=42+i)
    
    for site in range(168):
        ridge_V4 = linear_model.Ridge(1.) # linear_model.SGDRegressor()
        ridge_V4.fit(X_train, y_train[:,site])

        V4_explained_v[i,site] = explained_variance_score(y_test[:,site], ridge_V4.predict(X_test))
        
    print('done!')

zf_ = np.sqrt(np.abs(ZF_explained_v))
zf_.sort(axis=0)
alex_ = np.sqrt(np.abs(Alex_explained_v))
alex_.sort(axis=0)
hmo_ = np.sqrt(np.abs(HMO_explained_v))
hmo_.sort(axis=0)
resnext_ = np.sqrt(np.abs(ResNext_explained_v))
resnext_.sort(axis=0)
incep_ = np.sqrt(np.abs(Inception_explained_v))
incep_.sort(axis=0)
v4_ = np.sqrt(np.abs(V4_explained_v))
v4_.sort(axis=0)

df = pd.DataFrame(data = np.array([incep_[:,84], resnext_[:,84], hmo_[:,84], alex_[:,84], zf_[:,84], v4_[:,84]]).transpose(), columns = ['Inception','ResNext','HMO','Krizhevsky et al. 2012','Zeiler & Fergus 2013','V4 Multi'])

ax=sns.boxplot(x="variable", y="value", data=pd.melt(df))
ax.set(xlabel=' ',ylabel='Median IT Multi-Unit Explained Variance (%)')
ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
plt.ylim(0.1,.7)
plt.show()
