import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score



add_face_dir=r'dataset/faces/' 
faces_lst = []
name_lst = []
name_id=[]
final_img=[]
f_img_list=[] #list of all the flattened images
id_count = 0

for names in os.listdir(add_face_dir):
    
    name_dir= os.path.join(add_face_dir,names)
    name_lst.append(names)
    
    for img in os.listdir(name_dir):
        image_dir=os.path.join(name_dir,img)
        img_f=cv2.imread(image_dir)

        if img_f is not None:
            f_img=cv2.cvtColor(img_f,cv2.COLOR_BGR2GRAY) #convert image to black n white
            resizedimg = cv2.resize(f_img,(200,200)) #resize image for symmentric analysis
            f=resizedimg.flatten()
            f_img_list.append(f)
            name_id.append(id_count)
    id_count+=1


X=np.array(f_img_list)
Y=np.array(name_id)
Z=np.array(name_lst)#actual name of ppl used for user convenience
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.4, random_state=55) #randomly split images in four arrays of ratio 40-60% ratio correspondingly


pca = PCA(n_components=120).fit(X_train)
X_test_pca = pca.transform(X_test)
X_train_pca=pca.transform(X_train)

#corespondingly jois the X and Y variables
lda=LinearDiscriminantAnalysis()
lda.fit_transform(X_train_pca, Y_train)
X_train_lda=lda.transform(X_train_pca)
X_test_lda=lda.transform(X_test_pca)


clf=MLPClassifier(max_iter=800,verbose=True).fit(X_train_lda,Y_train)


# Prediction the labels for the test set
Y_pred = clf.predict(X_test_lda)

accuracy = accuracy_score(Y_test, Y_pred)


# Calculate the number of correct and incorrect predictions
correct_predictions = np.sum(Y_test == Y_pred)
incorrect_predictions = np.sum(Y_test != Y_pred)

# Display the results
print(f"Accuracy of the model: {accuracy * 100:.2f}%")
print(f"Number of tests passed: {correct_predictions}")
print(f"Number of tests failed: {incorrect_predictions}")


