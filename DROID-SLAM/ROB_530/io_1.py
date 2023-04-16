# Programmer: Junkai Zhang, April 1st 2023
# Description: This file transfers the image, depth image, semantic information
# and the pose of the robot to the semanticKITTI dataset format we want

import os
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# Read .bin file from given inputDataStr
def readBinFile(inputDataStr):
    # Read the .bin file
    inputData = np.fromfile(inputDataStr, dtype=np.float32)
    # Reshape the data
    inputData = inputData.reshape((-1, 4))

    # Return the data
    return inputData

# Read .label file from given inputDataStr

# TODO: The label file is not read correctly. There is some decoding process
def readLabelFile(inputDataStr):
    # Read the .label file
    inputData = np.fromfile(inputDataStr, dtype=np.uint32)
    # Reshape the data
    inputData = inputData.reshape((-1))

    # Return the data
    return inputData

# Define a wrapper function to read label and bin files from the given path and
# return a list of label or bin files. 
def readFiles(inputDataStr, fileFormat):
    # Initialize the list
    fileList = []
    # Get the length of the file
    fileLen = len(os.listdir(inputDataStr))
    # Read the files
    for i in range(fileLen):
        fileStr = inputDataStr + str(i).zfill(6) + fileFormat
        if fileFormat == '.bin':
            fileData = readBinFile(fileStr)
        elif fileFormat == '.label':
            fileData = readLabelFile(fileStr)
        fileList.append(fileData)
    # Return the list
    return fileList

# Read from the .npy file
def readNpyFile(inputDataStr):
    # Read the .npy file
    inputData = np.load(inputDataStr)
    # Return the data
    return inputData

# Define a DepthSemanVideoclass to store the images, depth images, semantic 
# information and the pose of the robot. The input of this class is npy files
class DepthSemanVideo:
    def __init__(self, tstamp, intrinsic, depthImage, semanticImage, videoImage, pose):
        self.tstamp = tstamp
        self.timeLen = tstamp.shape[0]
        self.depthMask = None
        self.depthImage = self.__setDepth(depthImage)
        
        self.semanticImage = semanticImage
        self.videoImage = videoImage
        self.pose = self.__setPose(pose)
        self.intrinsic = self.__initIntrinsic(intrinsic)
        self.pointCloud = None
        

        
        self.__setPointCloud()


    def __initIntrinsic(self, intrinsic): 
        fx, fy, cx, cy = intrinsic
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return intrinsic


    # # This function builds point cloud matrix from depth matrix, using inverse
    # # of the intrinsic matrix
    # def __setPointCloud(self):

    #     # Get the inverse intrinsic
    #     depths = self.depthImage # N*H*W
    #     invIntrinsic = np.linalg.inv(self.intrinsic)

    #     # Concantenate the grid_x, grid_y, and ones
    #     self.pointCloud = np.zeros((depths.shape[0], 3, depths.shape[1], depths.shape[2]))
    #     H, W = depths.shape[1:]
    #     grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    #     ones = np.ones_like(depths[0])

    #     pixel_coords = np.stack([grid_x, grid_y, ones], axis=-1).reshape(-1, 3)
    #     pixel_coords = np.expand_dims(pixel_coords, axis=0)
    #     pixel_coords = np.repeat(pixel_coords, depths.shape[0], axis=0)
    #     pixel_coords = pixel_coords.reshape(-1, H, W, 3)

    #     depths_pixel_coords = np.expand_dims(depths, axis = 3) * pixel_coords
    #     depths_pixel_coords = np.expand_dims(depths_pixel_coords, axis=4)

    #     invIntrinsic = invIntrinsic.reshape((1,1,1) + invIntrinsic.shape)
    #     self.pointCloud = np.matmul(invIntrinsic, depths_pixel_coords).squeeze()


    # # This function builds point cloud matrix from depth matrix, using inverse
    # # of the intrinsic matrix
    def __setPointCloud(self):

        # Get the inverse intrinsic
        depths = self.depthImage # N*H*W
        fx = self.intrinsic[0,0]
        fy = self.intrinsic[1,1]
        cx = self.intrinsic[0,2]
        cy = self.intrinsic[1,2]
        # Concantenate the grid_x, grid_y, and ones
        self.pointCloud = np.zeros((depths.shape[0], 3, depths.shape[1], depths.shape[2]))
        H, W = depths.shape[1:]
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        print(grid_x.shape)
        ones = np.ones_like(depths[0])

        grid_x = (grid_x - cx) / fx
        grid_y = (grid_y - cy) / fy

        pixel_coords = np.stack([grid_x, grid_y, ones], axis=-1).reshape(H, W, 3)
        pointCloud = pixel_coords[None,:] * depths[:,:,:,None]
        self.pointCloud = pointCloud



    def __setDepth(self, depthImage):
        max_depth = 1
        # depthImage = 1 / np.clip(depthImage, 1 / max_depth, None)

        depthImage = 1 / depthImage
        self.depthMask = depthImage < max_depth
        return depthImage
    
    # Write a private function to write the pointCloud to .bin file
    def __writeBin(self, outputFolderPath):
        pointCloud = self.pointCloud
        for frame in range(self.timeLen):
            curPC = pointCloud[frame][self.depthMask[frame]].reshape(-1, 3)
            curPC = np.concatenate((curPC, np.zeros((curPC.shape[0], 1))), axis=1)
            curPC = curPC.astype(np.float32)
            curPC.tofile(outputFolderPath + str(frame).zfill(6) + '.bin')

    def __writeLabel(self, outputFolderPath):
        semanticImage = self.semanticImage
        for frame in range(self.timeLen):
            # curLabel = semanticImage[frame].reshape(-1)
            curLabel = semanticImage[frame][self.depthMask[frame]].reshape(-1)
            curLabel = curLabel.astype(np.uint32)
            curLabel.tofile(outputFolderPath + str(frame).zfill(6) + '.label')
    
    # Write the pose to .txt file. self.pose has shape (329, 7)
    # The first 3 elements are the translation, the last 4 elements are the quaternion
    # This function saves the pose in the format of translation matrix
    def __writePose(self, outputFolderPath):
        np.savetxt(outputFolderPath+"poses.txt", self.pose, delimiter=" ", newline="\n")

    def __setPose(self, pose):
        poseMatrix = np.zeros((pose.shape[0], 12))
        for i in range(pose.shape[0]):
            currentPose = pose[i]
            # currentPose.shape = (7,)
            # The first 3 elements are the translation, the last 4 elements are the quaternion
            currentPose = currentPose.astype(np.float32)
            # R = self.__quaternion_matrix(currentPose[3:])
            r = R.from_quat(currentPose[3:])
            r = r.as_matrix()
            T = currentPose[0:3]
            poseMatrix[i] = np.concatenate((r, T[:, None]), axis=1).reshape(1, 12)

        return poseMatrix

    def __quaternion_matrix(self, quaternionArr):
        qx, qy, qz, qw = quaternionArr
        R = np.array([
            [1-2*qy**2-2*qz**2, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
            [2*qx*qy+2*qz*qw, 1-2*qx**2-2*qz**2, 2*qy*qz-2*qx*qw],
            [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx**2-2*qy**2]
        ])

        return R
        
    # Output the attribute in class to the semanticKITTI dataset format.
    def outputSemanticKITTI(self, outputFolderPath):
        # Output the depthImage to .bin file under the outputFolderPath
        # depthImage has shape (time, height, width)
        self.__writeBin(outputFolderPath + 'velodyne/')
        self.__writeLabel(outputFolderPath + 'predictions/')
        self.__writePose(outputFolderPath)


    # Plot the point cloud, image and depth image
    def plotPCL(self, frameNum):
        point_cloud = self.pointCloud[frameNum][self.depthMask[frameNum]][None,:]
        # point_cloud = point_cloud[::5, ::5,:]
        point_cloud = point_cloud[:, ::25,:]
        fig = plt.figure()
        ax = fig.add_subplot(141, projection='3d')
        ax.scatter(point_cloud[:,:,0], point_cloud[:,:,1], point_cloud[:,:,2], s=0.1)

        # Add axis label for plotPCL
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        
    def plotCompletePCL(self):
        total_PC = np.zeros((0,3))
        for i in range (self.timeLen):
            point_cloud = self.pointCloud[i][self.depthMask[i]]
            rot_matrix = self.pose[i].reshape(3,4)
            trans_matrix = np.concatenate((rot_matrix, np.array([[0,0,0,1]])), axis=0) 
            # print(point_cloud.shape)
            point_cloud = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1))), axis=1)
            point_cloud = np.linalg.inv(trans_matrix) @ point_cloud[:,:,None]
            point_cloud = point_cloud[::1000,:3].squeeze()
            total_PC = np.concatenate((total_PC, point_cloud), axis=0)

        # total_PC = total_PC[::5,:3]
        fig = plt.figure()
        ax = fig.add_subplot(141, projection='3d')
        ax.scatter(total_PC[:,0], total_PC[:,1], total_PC[:,2], s=0.1)

        # Add axis label for plotPCL
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()


    def plotImage(self, frameNum):
        plt.subplot(142)
        plt.imshow(self.videoImage[frameNum].transpose((1, 2, 0)))

    def plotDepthImage(self, frameNum):
        plt.subplot(143)
        depthOutput = np.log(self.depthImage[frameNum])
        plt.imshow(depthOutput)
    
    def plotSemanticImage(self, frameNum):
        plt.subplot(144)
        plt.imshow(self.semanticImage[frameNum])

    def plot(self, frameNum):
        self.plotPCL(frameNum)
        self.plotImage(frameNum)
        self.plotDepthImage(frameNum)
        self.plotSemanticImage(frameNum)
        plt.show()

    
        


        

        
def main():
    # Read kitti dataset
    # inputLabelPath = 'semantickitti_04/labels/'
    # inputBinPath = 'BenchKitti/velodyne/'

    # labelList = readFiles(inputLabelPath, '.label')
    # binList = readFiles(inputBinPath, '.bin')
    # print(binList[0].shape)
    # print(labelList[0].shape)

    # ----------------- Read npy and Output Kitti -----------------
    # Read npy files
    npyFolder = '../reconstructions/office_reconstruction/'
    # npyFolder = 'reconstructions/'
    tstamps = readNpyFile(npyFolder + 'tstamps.npy')
    depths = readNpyFile(npyFolder + 'disps.npy')
    images = readNpyFile(npyFolder + 'images.npy')
    poses = readNpyFile(npyFolder + 'poses.npy')    
    semantics = readNpyFile(npyFolder + 'semantics.npy')
    intrinsicFromnpy = readNpyFile(npyFolder + 'intrinsics.npy')
    intrinsicDic = {'bench':np.array([726.21, 726.21, 359.2048, 202.4724]), \
                    'office':np.array([600.0, 600.0, 599.5, 339.5])}

    # intrinsicDic = {'bench':np.array([726.21, 726.21, 359.2048, 202.4724]), 'office':np.array([400.0, 300.0, 399.5, 339.5])}
    intrinsic = intrinsicDic['office']


    depthSemanVideo = DepthSemanVideo(tstamps, intrinsic, depths, semantics, images, poses)

    # Assuming you have a point cloud stored in a numpy array called `point_cloud`

    depthSemanVideo.plot(75)
    # depthSemanVideo.plotCompletePCL()
    # depthSemanVideo.outputSemanticKITTI('officekitti/')

    


if __name__ == "__main__":
    main()







