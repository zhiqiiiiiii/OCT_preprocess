import numpy as np
import scipy.misc
from scipy import ndimage as nd
from scipy import signal as sg
import matplotlib.pyplot as plt
import pdb
from scipy import interpolate as ip
import inpaint_nan3 as inan

def retinaFlatten(img_data, shifts, interp):
    """
    Flattening by shifting A-scans up or down depending on the value in 'shifts'
    Extrapolation just adds zeros to the top or bottom of the A-scan
    Args:
    vol: image volume [img_rows, img_cols, Bscans]
    shifts: [img_rows, Bscans]
    interp: interp method. 'linear' or 'nearest'
    Return: 
    shiftImg: [img_rows, img_cols, Bscans]
    """
    img_rows, img_cols, B = img_data.shape
    types = img_data.dtype
    assert interp == 'linear' or interp == 'nearest'
    xx = np.arange(img_cols)
    yy = np.arange(img_rows)
    [X,Y] = np.meshgrid(xx,yy)
    Ym = Y[:,:,np.newaxis] + shifts[np.newaxis,:,:]
    shiftImg = np.zeros(img_data.shape)
    for i in range(B):
        f = ip.RegularGridInterpolator((yy,xx),img_data[:,:,i],bounds_error=False,method=interp)
        z = np.stack((Ym[:,:,i],X),-1)
        shiftImg[:,:,i] = f(z) 
    shiftImg[np.isnan(shiftImg)] = 0
    return shiftImg.astype(types)

def detect_retina(img_vol, pixelsX=200, numFrames=200, pixelsZ=1024, xWidth=6, yWidth=6, zWidth=2, normalize=True, outputRetinaMask=True, denoised=True, flattern=True):
    '''
    xWidth, yWidth, zWidth: mm actual scanning area
    '''
    
    # microns/pixel
    xScale = xWidth*1000/pixelsX      
    yScale = yWidth*1000/numFrames
    zScale = zWidth*1000/pixelsZ
    
    
    # Cirrus parameters
    params = dict()
    params['sigma_lat'] = 2*16.67
    params['sigma_ax'] = 0.5*11.6
    params['distconst'] = 96.68
    params['sigma_lat_ilm'] = 55.56
    params['sigma_lat_isos'] =55.56
    params['sigma_lat_bm'] = 111.13
    params['maxdist'] = 386.73 # ~100 pixels in spectralis
    params['bsc_indep'] = True # process each B scan independently

    maxdist = params['maxdist'] # maximum distance from ILM to ISOS
    maxdist_bm = 116.02 # maximum distance from ISOS to BM
    isosThresh = 20 # Minimum distance from ISOS to BM
    #Median filter outlier threshold distance and kernel
    dc_thresh = 5
    mf_k = 140

    # sigma values for smoothing final surfaces
    sigma_tp_ilm = 91.62
    sigma_tp_isos = 91.62
    sigma_tp_bm = 244.32
    # lateral direction
    sigma_lat_ilm = params['sigma_lat_ilm']
    sigma_lat_isos = params['sigma_lat_isos']
    sigma_lat_bm = params['sigma_lat_bm']

    # convert all values frmo micron to pixel
    sigma_lat = params['sigma_lat']/(xScale)
    sigma_ax = params['sigma_ax']/(zScale)
    distConst = np.round(params['distconst']/(zScale))
    maxdist = np.round(maxdist/(zScale))
    maxdist_bm = np.round(maxdist_bm/(zScale))
    isosThresh = np.round(isosThresh/(zScale))
    dc_thresh = np.round(dc_thresh/(zScale))

    temp = np.round(np.array([(mf_k/xScale),(mf_k/yScale)]))  #?????
    mf_k = (temp*2 +1).reshape((1,2))    #?????
    sigma_tp_ilm = sigma_tp_ilm/yScale
    sigma_tp_isos = sigma_tp_isos/yScale
    sigma_tp_bm = sigma_tp_bm/yScale
    sigma_lat_ilm = sigma_lat_ilm/(xScale)
    sigma_lat_isos = sigma_lat_isos/(xScale)
    sigma_lat_bm = sigma_lat_bm/(xScale)

    
    # gaussian denoise
    sigma_ax = float(sigma_ax)
    sigma_lat = float(sigma_lat)
    grad = nd.gaussian_filter(img_vol, sigma = (sigma_ax,0, 1), mode='nearest', 
                              order=0,truncate=2*np.round(2*sigma_ax) + 1) 
    grad = nd.gaussian_filter(grad, sigma = (0,sigma_lat,1), mode='nearest',
                              order=0,truncate=2*np.round(2*sigma_lat) + 1)
    img_denoised = grad
    
    
    # normalize
    if normalize:
        grad = (grad - grad.min())/(grad.max() - grad.min())
        
    
    # compute gradient along Z direction
    grad = nd.sobel(grad, mode='nearest', axis =0)
    
    
    # find the two largest gradient corresponds to ILM & IS/OS
    grad_o = grad.copy()
    max1pos = np.argmax(grad, axis =0) # find largest gradient for every A scan, pixelsX x numFrames

    #Find the largest gradient to the max gradient at distance of
    #at least distCount away but not more than maxdist away
    for i in range(grad.shape[1]):
        for j in range(grad.shape[2]):
            dc = distConst
            if (max1pos[i,j] - distConst) < 1: # minimum distance go beyond upper boundary
                dc = max1pos[i,j] -1
            elif (max1pos[i,j] + distConst) > grad.shape[0]: # minimum distance go beyond lower boundary
                dc = grad.shape[0] - max1pos[i,j]
            grad[int(max1pos[i,j]-dc):int(max1pos[i,j]+dc), i,j] = 0

            #max distance
            if (max1pos[i,j] - maxdist) > 0:  # minimum distance go beyond upper boundary
                grad[:int(max1pos[i,j]-maxdist),i,j] = 0
            if (max1pos[i,j] + maxdist) <= grad.shape[0]:  # minimum distance go beyond lower boundary
                grad[int(max1pos[i,j]+maxdist):,i,j] = 0           
    
    max2pos = np.argmax(grad, axis =0)

    ilm = np.minimum(max1pos, max2pos)   # upper gradient corresponds to ILM
    isos = np.maximum(max1pos, max2pos)  # lower gradient corresponds to IS/OS 
    
    
    # Find BM, BM is largest negative gradient below the ISOS
    grad = grad_o
    # isosThresh: Minimum distance from ISOS to BM
    # maxdist_bm: maximum distance from ISOS to BM
    for i in range(grad.shape[1]):
        for j in range(grad.shape[2]):
            grad[:int(isos[i,j]+isosThresh), i ,j] = 0
            if (isos[i,j]+maxdist_bm) <= grad.shape[0]:
                grad[int(isos[i,j]+maxdist_bm):,i,j] = 0
    # To encourage boundary points closer to the top of the image, weight linearly by depth
    isos_temp = (grad.shape[0] - (isos[np.newaxis,:,:] + maxdist_bm))
    lin = np.transpose(np.arange(grad.shape[0])).reshape(1024,1,1) + isos_temp
    lin = -0.5/grad.shape[0] * lin + 1
    grad = grad*lin
    bm = np.argmin(grad, axis = 0) #no need to squeeze for python
    
    
    # detect outliers
    th = bm - ilm
    th = th.astype(float)
    th_med = sg.medfilt2d(th, (11, 11))
    bpt = (abs(th - th_med) > dc_thresh)

    ilm = ilm.astype(float)
    isos = isos.astype(float)
    bm = bm.astype(float)
    ilm[bpt] = np.nan  #find correspondance of nan
    isos[bpt] = np.nan
    bm[bpt] = np.nan
    nbpt = 0

    # Fill in outlier points:
    if np.any(np.any(bpt)): #since bpt is 2-D
        nbpt = np.sum(bpt)
        ilm = inan.inpaint_nans(ilm)
        isos = inan.inpaint_nans(isos)
        bm = inan.inpaint_nans(bm) 

    #Get final boundaries by smoothing
    #smooth surfaces
    sigma_tp_ilm = float(sigma_tp_ilm)
    sigma_tp_isos = float(sigma_tp_isos)
    sigma_tp_bm = float(sigma_tp_bm)
    sigma_lat_ilm = float(sigma_lat_ilm)
    sigma_lat_isos = float(sigma_lat_isos)
    sigma_lat_bm = float(sigma_lat_bm)

    ilm = nd.gaussian_filter(ilm, sigma = (sigma_tp_ilm, 0), mode='nearest', order=0,
                             truncate=2*np.round(3*sigma_tp_ilm) + 1)
    isos = nd.gaussian_filter(isos, sigma = (sigma_tp_isos, 0), mode='nearest', 
                              order=0,truncate=2*np.round(3*sigma_tp_isos) + 1)
    bm = nd.gaussian_filter(bm, sigma = (sigma_tp_bm, 0), mode='nearest', order=0, 
                            truncate=2*np.round(3*sigma_tp_bm) + 1)
    bm = nd.gaussian_filter(bm, sigma = (0, sigma_lat_bm), mode='nearest', order=0, 
                            truncate=2*np.round(3*sigma_lat_bm) + 1)
    ilm = nd.gaussian_filter(ilm, sigma = (0, sigma_lat_ilm), mode='nearest', order=0, 
                             truncate=2*np.round(3*sigma_lat_ilm) + 1)
    isos = nd.gaussian_filter(isos, sigma = (0, sigma_lat_isos), mode='nearest', order=0, 
                              truncate=2*np.round(3*sigma_lat_isos) + 1)
    #need to transfer all the image to filter function
    #Enforce ordering and a very small minimum thickness

    bmilm = (bm -ilm)*zScale <100
    ilm[bmilm] = bm[bmilm] - 100/zScale
    bmisos = (bm -isos)*zScale <10
    isos[bmisos] = bm[bmisos] - 10/zScale
    isosilm = (isos-ilm)*zScale < 90
    isos[isosilm] = ilm[isosilm] + 90/zScale

    # Make sure that we are not out of the volumn
    ilm[ilm <1] = 1
    ilm[ilm> img_vol.shape[0]] = img_vol.shape[0]
    isos[isos <1] = 1
    isos[isos > img_vol.shape[0]] = img_vol.shape[0]
    bm[bm<1] = 1
    bm[bm>img_vol.shape[0]] = img_vol.shape[0]

    # create mask volume
    if outputRetinaMask:
        retinaMask = np.zeros(img_vol.shape)
        for i in range(img_vol.shape[1]):
            for j in range(grad.shape[2]):
                retinaMask[max((0, int(np.round(ilm[i,j]))-100)):int(np.round(ilm[i,j])), i, j] = 0.5
                retinaMask[int(np.round(ilm[i,j])):int(np.round(isos[i,j])), i, j] = 1
                retinaMask[int(np.round(isos[i,j])):int(np.round(bm[i,j])), i, j] = 2
                retinaMask[int(np.round(bm[i,j])):min((1024, int(np.round(bm[i,j]))+200)), i, j]
    ilm_cat = ilm.reshape(ilm.shape[0], ilm.shape[1], 1)
    isos_cat = isos.reshape(isos.shape[0], isos.shape[1], 1)
    bm_cat = bm.reshape(bm.shape[0], bm.shape[1], 1)

    boundaries = np.concatenate((ilm_cat, isos_cat, bm_cat), axis= 2)
    bds = boundaries
    # define the shift amount here
    stemp = np.mean(bm, axis=0) + np.round(img_vol.shape[0]/2) - np.mean(bm, axis=0)
    shifts = bm - stemp.reshape((1,-1))
    
    tb = bds[:,:,0] - shifts
    if np.any(tb <0): #For the example case, won't get in
        shifts = shifts + np.amin(tb)
        #center
        d = np.amin(img_vol.shape[0] - (bds[:,:,-1] - shifts))
        shifts = shifts - np.amin(d)/2
    print('Flattening data')


    try:
        if flattern:
            if denoised:
                img_vol = retinaFlatten(img_denoised, shifts, 'linear')
            else:
                img_vol = retinaFlatten(img_vol, shifts, 'linear')
        else:
            img_vol = None
        if outputRetinaMask:
            retinaMask = retinaFlatten(retinaMask, shifts, 'nearest')
        else:
            retinaMask = None   
        print('done!\n')
        upperBound = ilm - shifts
        lowerBound = bm - shifts
    except Exception as e:
        print(str(e))
        img_vol = []
        quit()
        
    return img_vol, retinaMask, upperBound, lowerBound
    
    
