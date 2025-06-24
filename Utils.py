from tensorflow.keras.layers import add, Dense, MultiHeadAttention, Reshape, Input, Flatten,LeakyReLU, Add, BatchNormalization, concatenate
import numpy as np
from tensorflow.keras.models import Model
import h5py

# To measure timing
#' http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")
        


def yc_patch(A,l1,l2,o1,o2):

    n1,n2=np.shape(A);
    tmp=np.mod(n1-l1,o1)
    if tmp!=0:
        ##print(np.shape(A), o1-tmp, n2)
        A=np.concatenate([A,np.zeros((o1-tmp,n2))],axis=0)

    tmp=np.mod(n2-l2,o2);
    if tmp!=0:
        A=np.concatenate([A,np.zeros((A.shape[0],o2-tmp))],axis=-1); 


    N1,N2 = np.shape(A)
    X=[]
    for i1 in range (0,N1-l1+1, o1):
        for i2 in range (0,N2-l2+1,o2):
            tmp=np.reshape(A[i1:i1+l1,i2:i2+l2],(l1*l2,1));
            X.append(tmp);  
    X = np.array(X)
    return X[:,:,0]


def yc_snr(g,f):
    psnr = 20.*np.log10(np.linalg.norm(g)/np.linalg.norm(g-f))
    return psnr

def yc_patch_inv(X1,n1,n2,l1,l2,o1,o2):
    
    tmp1=np.mod(n1-l1,o1)
    tmp2=np.mod(n2-l2,o2)
    if (tmp1!=0) and (tmp2!=0):
        A     = np.zeros((n1+o1-tmp1,n2+o2-tmp2))
        mask  = np.zeros((n1+o1-tmp1,n2+o2-tmp2)) 

    if (tmp1!=0) and (tmp2==0): 
        A   = np.zeros((n1+o1-tmp1,n2))
        mask= np.zeros((n1+o1-tmp1,n2))


    if (tmp1==0) and (tmp2!=0):
        A    = np.zeros((n1,n2+o2-tmp2))   
        mask = np.zeros((n1,n2+o2-tmp2))   


    if (tmp1==0) and (tmp2==0):
        A    = np.zeros((n1,n2))
        mask = np.zeros((n1,n2))

    N1,N2= np.shape(A)
    ids=0
    for i1 in range(0,N1-l1+1,o1):
        for i2 in range(0,N2-l2+1,o2):
            ##print(i1,i2)
    #       [i1,i2,ids]
            A[i1:i1+l1,i2:i2+l2]=A[i1:i1+l1,i2:i2+l2]+np.reshape(X1[:,ids],(l1,l2))
            mask[i1:i1+l1,i2:i2+l2]=mask[i1:i1+l1,i2:i2+l2]+ np.ones((l1,l2))
            ids=ids+1


    A=A/mask;  
    A=A[0:n1,0:n2]

    return A


# Define the network
def msa(inp1, D1, projection_dim):

    num_heads = 8
    # MSMHA Network
    inp2 = Flatten()(inp1)

    x = Dense(D1)(inp2)
    x = LeakyReLU(0.1)(x)
    x = Dense(D1)(x)
    x = LeakyReLU(0.1)(x)

    x1 = Reshape((int(np.sqrt(D1)),int(np.sqrt(D1))))(x)
    #x1 = Reshape((50,20))(x)

    x1 = MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=0
    )(x1, x1)
    x1 = Flatten()(x1)
  
    x2 = Add()([x1,x])
    
    
    return x2

def MSMHA(ac,cc,w1,w2,modw):
    
    input_shape = (w1, w1,1)
    image_size = w1  # We'll resize input images to this size
    #projection_dim = int(100)
    

    inp1 = Input(shape=(w1,w2),name='input_layer')

    # Strong Denoiser
    y1 = msa(inp1,ac, int(np.sqrt(ac)))
    e = Dense(w1*w2,activation='linear')(y1)
    e = Reshape((w1,w2))(e)

    # Weak Denoiser
    y2 = msa(e,cc, int(np.sqrt(cc)))
    e1 = Dense(w1*w2,activation='linear')(y2)
    e1 = Reshape((w1,w2))(e1)


    model = Model(inputs=[inp1], outputs=[e,e1])

    model.summary()
    
    return model


def MSMHA_Multi(D1,D2,D3,D4,w1,w2,modw):
    
    input_shape = (w1, w1,1)
    image_size = w1  # We'll resize input images to this size
    #projection_dim = int(100)
    

    inp1 = Input(shape=(w1,w2),name='input_layer')

    # Strong Denoiser
    y1 = msa(inp1,D1, int(np.sqrt(D1)))
    e = Dense(w1*w2,activation='linear')(y1)
    e = Reshape((w1,w2))(e)

    # Weak Denoiser
    y2 = msa(e,D2, int(np.sqrt(D2)))
    e1 = Dense(w1*w2,activation='linear')(y2)
    e1 = Reshape((w1,w2))(e1)


    # Weak Denoiser
    y3 = msa(e1,D3, int(np.sqrt(D3)))
    e2 = Dense(w1*w2,activation='linear')(y3)
    e2 = Reshape((w1,w2))(e2)
    
     # Weak Denoiser
    y4 = msa(e2,D4, int(np.sqrt(D4)))
    e3 = Dense(w1*w2,activation='linear')(y4)
    e3 = Reshape((w1,w2))(e3)
    
    
    
    model = Model(inputs=[inp1], outputs=[e,e1,e2,e3])

    model.summary()
    
    return model
