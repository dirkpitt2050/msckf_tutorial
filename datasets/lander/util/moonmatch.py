import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from osgeo import gdal, osr
import spiceypy as cspice
from scipy.spatial.transform import Rotation
import os

imgs = [
'frames/screenshot-000001.png',
'frames/screenshot-000002.png',
'frames/screenshot-000003.png',
'frames/screenshot-000004.png',
'frames/screenshot-000005.png',
'frames/screenshot-000006.png'
]
# quaternions x,y,z,w from Moon-fixed to LVLH
# NOTE: roll (x) and pitch (y) are swapped
# and w signs are opposite
attitude = [
[-0.190918719132316,0,0,-0.981605848945938],
[-0.220668639412306,0,0,-0.975348835842809],
[-0.262791063501651,0,0,-0.964852764386189],
[-0.327293441083984,0,0,-0.944922749976634],
[ 0.435113014579987,0,0, 0.900375846268166],
[ 0.61784681949002, 0,0, 0.786298485084428]
]
# radius [m], lon [deg], lat [deg]
lonlat = [
[1740697.36330253,-19.52,44.0875351805348],
[1740194.68651941,-19.52,44.1252677267488],
[1739711.90236342,-19.52,44.1553536752533],
[1739252.16200680,-19.52,44.1777574671437],
[1738818.74489095,-19.52,44.1924445782873],
[1738415.06031979,-19.52,44.1993816673008]
]

# Only SIFT is scale invariant (needed for this example)
# Since initial altitude should be known, could be used to scale and crop image
# to same scale as query image. This is not done here however.
#detect = cv.ORB_create()
#detect = cv.BRISK_create(octaves=6)
detect = cv.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)

ds = gdal.Open('map/NAC_DTM_CHANGE3_SHADE2.TIF')
ds_dem = gdal.Open('map/NAC_DTM_CHANGE3.TIF')
geotransform = ds.GetGeoTransform()
invgeotransform = gdal.InvGeoTransform(geotransform)
xoffset, px_w, rot1, yoffset, rot2, px_h = geotransform
img2 = ds.ReadAsArray()
crs = osr.SpatialReference()
crs.ImportFromWkt(ds.GetProjectionRef())
crsGeo = osr.SpatialReference()
crsGeo.ImportFromProj4('+proj=longlat +a=1737400 +b=1737400 +no_defs')
transform2Latlon = osr.CoordinateTransformation(crs, crsGeo)
transformFromLatlon = osr.CoordinateTransformation(crsGeo, crs)

cspice.furnsh('data/latest_leapseconds.tls');
cspice.furnsh('data/de421.bsp');
cspice.furnsh('data/moon_pa_de421_1900_2050.bpc');
cspice.furnsh('data/moon_080317.tf');


def latlon2XYZSpice(lat, lon, alt):
    return cspice.latrec(alt + 1737400., lon, lat);

def blocknormalize(img,blockSide=21):
    I=np.float32(img)/255.0

    for i in range(0,img.shape[0],blockSide):           
        for j in range(0,img.shape[1],blockSide):        
            patch=I[i:i+blockSide,j:j+blockSide]
            m,s=cv.meanStdDev(patch)
            patch-=m[0]
            patch/=s[0]

    return (np.clip(I*127.+127., 0., 255.)).astype('uint8')

def readimage(fname, att, lonlat):
    attr = Rotation.from_quat(att)
    R = attr.as_matrix()
    alt = lonlat[0] - 1737400. - readatlonlat(lonlat[1], lonlat[2])
    print(alt)

    img1raw = cv.imread(fname, cv.IMREAD_GRAYSCALE)          # queryImage
    img1 = (np.clip(img1raw*4.0-190., 0, 255)).astype('uint8')

    # Homography correction for better image matching
    w = img1.shape[0]
    h = img1.shape[1]
    fov = 19*np.pi/180
    f = 0.5 * w / np.tan(fov*0.5)
    # Camera intrinsic matrix
    K = np.array([[f, 0, w/2],
                  [0, f, h/2],
                  [0, 0, 1]])
    # Homography matrix
    # (Note: since image space x axis is horizontal, pitch is actually "roll" in
    # image space)
    # Need to set roll = roll - 90 (ground plane relative to S/C)
    # Can multiply by a roll -90deg to achieve desired effect
    pitch = attr.as_euler('xyz')[0]
    print(f'pitch={pitch}')
    RG = np.array([[1.0,  0.0,  0.0],
                   [0.0,  0.0,  1.0],
                   [0.0, -1.0,  0.0]]) # roll -90
    RT = RG @ R
    # Normally, RT is a 3x4 matrix like this:
    # [[r11, r12, r13, tx],
    #  [r21, r22, r23, ty],
    #  [r31, r32, r33, tz]]
    # But since z = 0 at camera plane, the column [r13, r23, r33].T = 0
    # and omitting z from [x,y,z,1] we can just use [x,y,1] and RT =
    # [[r11, r12, tx],
    #  [r21, r22, ty],
    #  [r31, r32, tz]]
    T_sc = np.array([0.1, 0.18, 1.2])
    RT[:, 2] = -T_sc
    # Additional transform to bird's eye view camera
    T_bev = np.array([0.1, 0.2, 1.2])
    dx = h / f * T_bev[2]
    dy = w / f * T_bev[2]
    pxPerM = (w / dy, h / dx)
    shift = (w / 2., h / 2.)
    shift = shift[0] + T_bev[0] * pxPerM[0], shift[1] + T_bev[1] * pxPerM[1]
    M = np.array([[-1.0 / pxPerM[0], 0.0, shift[0]/pxPerM[0]],
                  [0.0, -1.0 / pxPerM[1], shift[1]/pxPerM[1]],
                  [0.0, 0.0, 1.0]])
    H = (K @ RT) @ M

    # warpPerspective() includes perspective projection (divide by z)
    warp1 = cv.warpPerspective(img1, H, img1.shape, flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP)
    return warp1

def spatialref(px):
    posX = px_w * px[0] + rot1 * px[1] + xoffset
    posY = rot2 * px[0] + px_h * px[1] + yoffset

    # shift to the center of the pixel
    posX += px_w / 2.0
    posY += px_h / 2.0

    return transform2Latlon.TransformPoint(posX, posY)

def readatlonlat(lon, lat):
    mapx, mapy, _ = transformFromLatlon.TransformPoint(lon, lat)
    x, y = gdal.ApplyGeoTransform(invgeotransform, mapx, mapy)
    alt = np.squeeze(ds_dem.ReadAsArray(x, y, 1, 1))
    return alt

def matchimage(queryimg, mapimg, ulx, uly, lrx, lry):
    latlon = []
    querymatches = []
    # find keypoints and descriptors
    kp1, des1 = detect.detectAndCompute(img1,None)
    kp2, des2 = detect.detectAndCompute(mapimg[uly:lry,ulx:lrx],None)

#    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
#    matches = bf.match(des1,des2)
#    matches = sorted(matches, key = lambda x:x.distance)[:10]
#    for i,m in enumerate(matches):
#            xy = np.array(kp2[m.trainIdx].pt) + [ulx, uly]
#            (lon, lat, _) = spatialref(xy)
#            # Sample altitude from DEM
#            alt = np.squeeze(ds_dem.ReadAsArray(xy[0], xy[1], 1, 1))
#            latlon.append([lat, lon, alt])
#            querymatches.append(kp1[m.queryIdx].pt)
#    img3 = cv.drawMatches(img1,kp1,mapimg[uly:lry,ulx:lrx],kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    matches = flann.knnMatch(des1,des2,k=2)     # Note: Flann is not deterministic (randomly changes every run)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75*n.distance:
            matchesMask[i]=[1,0]
            xy = np.array(kp2[m.trainIdx].pt) + [ulx, uly]
            (lon, lat, _) = spatialref(xy)
            # Sample altitude from DEM
            alt = np.squeeze(ds_dem.ReadAsArray(xy[0], xy[1], 1, 1))
            latlon.append([lat, lon, alt])
            querymatches.append(kp1[m.queryIdx].pt)

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1,kp1,mapimg[uly:lry,ulx:lrx],kp2,matches,None,**draw_params)
    return (img3, np.squeeze(querymatches), np.squeeze(latlon))

plt.ion()

for j, fname in enumerate(imgs):
#    if j != 0:
#        continue
    print(j)
    img1 = readimage(fname, attitude[j], lonlat[j])
#    img1 = blocknormalize(warp1,148)
    ulx = 0
    uly = 6000+j*400
    lrx = 2000
    lry = 7500+j*400
    (img3, querymatches, latlon) = matchimage(img1, img2, ulx, uly, lrx, lry)
    print(latlon)

    if len(latlon) > 0:
        # Convert lat, lon, alt to xyz
        if latlon.ndim < 2:
            xyz = np.array(latlon2XYZSpice(*latlon)).T
        else:
            xyz = [latlon2XYZSpice(*row) for row in latlon]

        xyzpairs = np.hstack((querymatches, xyz))
        print(xyzpairs)
    fig = plt.figure()
    ax = fig.subplots()
    plt.imshow(img3)
    plt.axis('off')
#    fig.savefig(fname + '.plot.png', format='png', dpi=300, pad_inches=0, bbox_inches='tight')
    np.savetxt(fname + '.matches.csv', xyzpairs, fmt='%.10lf', delimiter=',',
               header='query match x [px],query match y [px],map planet fixed x [m], map planet fixed y [m], map planet fixed z [m]')
