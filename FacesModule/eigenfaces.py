import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time

class EigenEmotions:
    def __init__(self, data_dir, labels_dir):

        data = sio.loadmat(data_dir)
        labels = sio.loadmat(labels_dir)

        self.train_images = data['train_images']
        self.train_data = data['train_data']
        self.train_labels = labels['train_labels']

        self.test_images = data['test_images']
        self.test_data = data['test_data']
        self.test_labels = labels['test_labels']

        self.data_prep()
        self.eigen_faces()

    def data_prep(self):
        self.train_mat = self.train_data - np.mean(self.train_data,0)
        self.train_mat = self.train_mat / np.linalg.norm(self.train_mat,2,0)

        self.test_mat = self.test_data - np.mean(self.test_data,0)
        self.test_mat = self.test_mat / np.linalg.norm(self.test_mat,2,0)

    def eigen_faces(self):
        ata = self.train_mat.T.dot(self.train_mat)
        ata_d, ata_v = np.linalg.eig(ata)
        self.v = self.train_mat.dot(ata_v)

    def classifier(self, face_num, M, singular):
        vM = self.v[:,0:M]
        w = self.train_mat.T.dot(vM)
        self.ws = w.shape
        foo = self.test_mat[:,face_num].T.dot(vM)
        self.fs = foo.shape
        comparison = np.sqrt(np.mean((w - foo)**2,1))
        face = np.nanargmin(comparison)
        test_name = self.test_labels[face_num]
        match_name = self.train_labels[face]

        # DEBUGGING LINES
        # print(face)
        # print(test_name, match_name)
        # comp_string = "The face of " + test_name + " is identified as the face of " + match_name + "."
        # print(comp_string)

        if singular == True:
            print(face)
            print(test_name, match_name)
            comp_string = "The face of " + test_name + " is identified as the face of " + match_name + "."
            print(comp_string)
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(self.test_images[:,:,face_num])
            plt.title('Test image')
            plt.xlabel(test_name)
            plt.subplot(1,2,2)
            plt.imshow(self.train_images[:,:,face])
            plt.title('Matched image')
            plt.xlabel(match_name)
            plt.show()
            
        return (test_name, match_name)
            

    def accuracy_test(self, M):
        acc = 0
        count = 0
        for j in np.arange(0,self.test_mat.shape[1],dtype=int):
            tn,mn = self.classifier(j,M, False)
            if tn == mn:
                acc = acc + 1
            count = count + 1
        return acc / count

    def num_eigen_faces(self):
        return self.v.shape


if __name__ == "__main__":
    start_time = time.time()

    data = "/home/sswaminathan/olin_share/Semester2/QEA/QEAProjects/FacesModule/DataSets/curated/curated.mat"
    labels = "/home/sswaminathan/olin_share/Semester2/QEA/QEAProjects/FacesModule/DataSets/curated/labels.mat"

    classifier = EigenEmotions(data, labels)

    qual = []
    # for i in np.arange(1,classifier.num_eigen_faces()[1]+1,dtype=int):
    for i in np.arange(1,11,dtype=int):
        qual.append([i,classifier.accuracy_test(i)])

    qual = np.array(qual)
    print("Standard EigenFaces method best accuracy: ", qual[-1])

    print("Time elapsed: ", time.time() - start_time)
    """

    start_time = time.time()

    data = "/home/sswaminathan/olin_share/Semester2/QEA/QEAProjects/FacesModule/DataSets/curated/average_data.mat"
    labels = "/home/sswaminathan/olin_share/Semester2/QEA/QEAProjects/FacesModule/DataSets/curated/average_labels.mat"

    avg_classifier = EigenEmotions(data, labels)

    avg_qual = []
    for i in np.arange(1,avg_classifier.num_eigen_faces()[1]+1,dtype=int):
        avg_qual.append([i,avg_classifier.accuracy_test(i)])

    avg_qual = np.array(avg_qual)

    print("Time elapsed: ", time.time() - start_time)

    # print("Standard EigenFaces method best accuracy: ", qual[-1])
    print("Using average faces generated via dlib and OpenCV best accuracy: ", avg_qual[-1])

    """
    plt.figure()
    plt.plot(qual[:,0],qual[:,1])
    plt.title('Eigenfaces vs. Accuracy')
    plt.xlabel('Number of Eigenfaces used')
    plt.ylabel('Accuracy')
    plt.show()
    """

    plt.figure()
    plt.plot(avg_qual[:,0],avg_qual[:,1])
    plt.title('Eigenfaces vs. Accuracy')
    plt.xlabel('Number of Eigenfaces used')
    plt.ylabel('Accuracy')
    plt.show()
    """
