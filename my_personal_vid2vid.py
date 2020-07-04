
import cv2
import numpy as np
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import interpolate,linalg
import matplotlib.path as mpltPath
import random

"""
Author: Tal Daniel
Tal wrote the following 2 functions that break a video to frames and reasemble a video from frames
"""
def image_seq_to_video(imgs_path, output_path='./video.mp4', fps=15.0):
    output = output_path
    img_array = []
    for filename in glob.glob(os.path.join(imgs_path, '*.jpg')):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        # img = cv2.resize(img, (width // 2, height // 2))
        img = cv2.resize(img, (width, height))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    print(size)
    print("writing video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, fps, size)
    # out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("saved video @ ", output)

def video_to_image_seq(vid_path, output_path='./datasets/OTB/img/Custom/'):
    os.makedirs(output_path, exist_ok=True)
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 0
    print("converting video to frames...")
    while success:
        fname = str(count).zfill(4)
        cv2.imwrite(os.path.join(output_path, fname + ".jpg"), image)  # save frame as JPEG file
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1
    print("total frames: ", count)


# from this part onward are the main functions

# compute H uses SVD to calculate an affine homographie between sets of matching points on different images
def computeH(p1, p2):
    p1 = [(round(x[0],2),round(x[1],2)) for x in p1]
    p2 = [(round(x[0],2),round(x[1],2)) for x in p2]
    N=len(p1)
    A=np.zeros((2*N,9))
    for i in range(N):
        A[2*i,0]=p2[i][0]
        A[2*i, 1] = p2[i][1]
        A[2*i,2] = 1
        A[2 * i, 6] = -1*p2[i][0]*p1[i][0]
        A[2 * i, 7] = -1*p2[i][1]*p1[i][0]
        A[2 * i, 8] = -1*p1[i][0]
        A[2 * i +1, 3] = p2[i][0]
        A[2 * i +1, 4] = p2[i][1]
        A[2 * i +1, 5] = 1
        A[2 * i +1, 6] = -1 * p2[i][0] * p1[i][1]
        A[2 * i +1, 7] = -1 * p2[i][1] * p1[i][1]
        A[2 * i +1, 8] = -1 * p1[i][1]
    #print(A)
    # now A matrix is built and we use svd
    (U, D, V) = np.linalg.svd(A, False)
    # left most column of svd is the solution to the argmin problem
    V=V.T
    H2to1 = V[:, -1]
    # reshape to matrix
    H2to1=np.reshape(H2to1,((3,3)))
    # normalize by 3,3 element to create homographie
    H2to1/=H2to1[2,2]
    return H2to1

# inputs: y,x coordinates on the original grid, and a homographie
# outputs: y,x coordinates on the new grid
def applyH(y,x,H):
    vec = np.zeros((3, 1))
    vec[0, 0] = x
    vec[1, 0] = y
    vec[2, 0] = 1
    mat = np.matmul(H, vec)
    mat=mat / mat[2, 0]
    newX=mat[0,0]
    newY=mat[1,0]
    return [newY,newX]


# this function gets a path to an image, asks you to mark the book's 4 corners and middels of the long axis returns a ref image
def create_ref(im_path):
    im= cv2.imread(im_path)
    im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    width = 250
    Hight=int(width*1.5)
    plt.figure()
    plt.imshow(im)
    # mark the 4 corners and centers of the long axis of the book, starting from top left and going clock-wise
    points = plt.ginput(6, timeout=60)
    plt.close()
    points_ref = [(0, 0), (width, 0), (width,Hight/2), (width, Hight), (0, Hight), (0,Hight/2)]
    H=computeH(points_ref, points)
    ref_image=np.zeros((Hight,width,3))
    # black out pixels out of the polygon
    path = mpltPath.Path(points)
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            coordinates = applyH(y, x, H)
            if coordinates[0]>0 and coordinates[0]<Hight and coordinates[1]>0 and coordinates[1]<width:
                # if point is part of the book we warp it
                ref_image[int(coordinates[0]),int(coordinates[1]),:]=im[y,x,:]

    # ref_image=warpH(im,H,(150,100))
    # ref_image = cv2.warpPerspective(im, H, (100, 150))
    return np.uint8(ref_image)

## this piece of code creates a reference model
# refIm=create_ref('.\\my_data\\Hobbit.jpeg')
# plt.figure()
# plt.imshow(refIm)
# plt.show()
# cv2.imwrite('.\\my_data\\refHobbit.jpg', cv2.cvtColor(refIm, cv2.COLOR_RGB2BGR))

# this function gets 2 images, applys SIFT to find matching points, preforms a second match test and returns
# p1,p2 sets of coordinates of matching points H will be calculates based on
def getPoints_SIFT(im1, im2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(im2,cv2.COLOR_RGB2GRAY), None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    p1=[]
    p2=[]
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])
            p1.append(kp1[m.queryIdx].pt)
            p2.append(kp2[m.trainIdx].pt)
    return p1,p2

# ransac gets sets of matching points, number of iterations and tolerance
# output: a homographie based on random X (=28, could differ) matching points from the images who's slopes differences
# are smaller than the tolerance - basically enforcing a more 'correct' transform
def ransacH(p1, p2, nIter=15, tol=0.15):
    p1_selected=[]
    p2_selected=[]
    for iter in range(nIter):
        # first we sample 8  random matching points
        p1_new=[]
        p2_new=[]
        # print('iteration '+str(iter))
        for i in range(28):
            x = random.randint(0,len(p1))
            p1_new.append(p1[x-1])
            p2_new.append(p2[x-1])
        # vectors is a list of all the slopes
        vectors=[]
        for i in range(28):
            vectors.append((p2_new[i][1]-p1_new[i][1])/(p2_new[i][0]-p1_new[i][0]))
        # check if vectors' spread is smaller than the best
        if np.max(vectors)-np.min(vectors) < tol:
            p1_selected = p1_new
            p2_selected = p2_new
            break
    if len(p1_selected)==0:
        p1_selected=p1
        p2_selected=p2
        print('ransac fail')
    bestH=computeH(p1_selected, p2_selected)
    return bestH

# this function gets a warped image and cropps it
def cropWar(im):
    width=im.shape[1]
    hight=im.shape[0]
    for x in range(10,im.shape[1]):
        if np.array_equal(im[:,x,:].reshape((im.shape[0],1,3)),np.zeros((im.shape[0],1,3))): #first all black column
            width=x
            break
    for y in range(10,im.shape[0]):
        if np.array_equal(im[y,:,:].reshape((1,im.shape[1],3)),np.zeros((1,im.shape[1],3))): #first all black row
            hight=y
            break
    # croppedIm=np.zeros((hight,width,3),dtype=np.uint8)
    croppedIm=im[0:hight,0:width,:]
    return croppedIm


# this function is a version of im2im that smoths H's for clips, we are assuming H doesnt change much between frames
# so we mostly use the stable H
def im2imMov(image, refrence_model, new_image, LastH, tolerance=0.2, isLastStable=False):
    repImage=image
    # now we sift between the reference and the scene image
    p1,p2=getPoints_SIFT(image,refrence_model)
    # calculate the homographie
    # H = computeH(p1, p2)
    H=ransacH(p1,p2,100,tolerance)
    # now we calculate the difference bewteen this H and the last one,
    # we assume they need to be close so the determinant of mul of H and lastH inverse should be around 1
    if isLastStable==True: # H larger than 1
        diffVal=np.abs(np.linalg.det(H@linalg.pinv(LastH)))
        det=np.linalg.det(H)
        print('diff val:'+str(diffVal))
        print('current det: '+str(det))
        if diffVal>1.1 or diffVal < 0.9:
            print('correcting to last H')
            # H's are two different, take the last one
            H=LastH

    repImage=image[:,:,:]
    # now we go pixel by pixel of the new image and put it in its place in the new image
    im=cv2.resize(new_image, (refrence_model.shape[1],refrence_model.shape[0]))
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            newImageCoordinates=applyH(y,x,H)
            if newImageCoordinates[0]>0 and newImageCoordinates[0]<image.shape[0] and newImageCoordinates[1]>0 and newImageCoordinates[1]<image.shape[1]:
                repImage[int(newImageCoordinates[0]),int(newImageCoordinates[1]),:]=im[y,x,:]
    return repImage,H

# video_to_image_seq('.\\VidData\\dancing_man_model.mp4','.\\VidData\\DancingManFrames\\')
# for name in os.listdir('.\\VidData\\DancingManFrames'):
#   newName=str(int(name.split('.')[0])+242).zfill(4)
#   os.rename('.\\VidData\\DancingManFrames\\'+name,'.\\VidData\\DancingManFrames\\'+newName+'.jpg')

ref= cv2.imread('.\\VidData\\refHobbit.jpg')
ref=cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
for name in os.listdir('.\\VidData\\OGframes'):
  frameScene = cv2.imread('.\\VidData\\OGframes\\'+name)
  if int(name.split('.')[0])<75:
      cv2.imwrite('.\\VidData\\NewFrames\\' + name, frameScene)
      continue
  elif int(name.split('.')[0])>74 and int(name.split('.')[0])<242:
      smallFrame = cv2.imread('.\\VidData\\Logo.jpg')
  else:
      smallFrame = cv2.imread('.\\VidData\\DancingManFrames\\'+name)
  print(name)
  # implent the first frame
  if int(name.split('.')[0])==75:
    im,H = im2imMov(frameScene, ref, smallFrame,0, 0.4,False)
  else:
      # here we use an im2im version that "smooths" jumps in H
      im, H = im2imMov(frameScene, ref, smallFrame, H, 0.4, True)

  cv2.imwrite('.\\VidData\\NewFrames\\'+name, im)

image_seq_to_video('.\\VidData\\NewFrames\\','.\\VidData\\finalVid.mp4',25) # we slowed it from 29 to 25 fps so that its longer...