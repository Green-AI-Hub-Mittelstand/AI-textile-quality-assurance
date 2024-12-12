import numpy as np
import gudhi as gd
from skimage import io, color 
import cv2 as cv
import joblib
import time
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


# Load and preprocess the image
def preprocess_image(image_path):
    image = io.imread(image_path)
    gray_image = color.rgb2gray(image) * 255
    gray_image = gray_image.astype(np.uint8)
    return gray_image

def load_image(image_path):
    image = io.imread(image_path)
    return image

# Compute persistent homology
def compute_persistence(gray_image):
    #Note: flatten has to be in fortran order (='F')
    cc = gd.CubicalComplex(dimensions=gray_image.shape, top_dimensional_cells=gray_image.flatten(order='F'))
    persistence = cc.persistence()
    return persistence

# Extract features from persistence diagram
def extract_features(persistence):
    num_0d = sum(1 for p in persistence if p[0] == 0)
    num_1d = sum(1 for p in persistence if p[0] == 1)
    return [num_0d, num_1d]

#transform persistance diagram to image
def persistence_diagram_to_image(persistence, resolution=50, features_dim='all', max_value=None):
    if max_value is None:
        max_value = np.max([point[1][1] for point in persistence if point[1][1] < np.inf])
    img = np.zeros((resolution, resolution))
    #h0 and h1 eqauly handelt, it could makes sense to only consider h1, (dimension==1)
    for dimension, (birth, death) in persistence:
        if features_dim == 'all':
            if death < np.inf:
                x = int((birth / max_value) * (resolution - 1))
                y = int((death / max_value) * (resolution - 1))
                img[y, x] += 1
        else:
            if dimension == features_dim and death < np.inf:
                x = int((birth / max_value) * (resolution - 1))
                y = int((death / max_value) * (resolution - 1))    
                img[y, x] += 1
    return img


def train_model(image_paths, labels, model_path, resolution=50, features_dim='all' , n_estimators=100):
    features = []
    #check if grey or colored images, on first image
    path = image_paths[0]
    gray = (len(io.imread(path).shape) == 2)
        
    
    for path in image_paths:
        if gray:
            gray_image = io.imread(path)
        else:
            gray_image = preprocess_image(path)
        gray_image = cv.bitwise_not(gray_image)
        persistence = compute_persistence(gray_image)
        persistence_image  = persistence_diagram_to_image(persistence, resolution, features_dim, max_value=None)
        features.append(persistence_image.flatten()) 
    
    X = np.array(features)
    y = np.array(labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_path)

    # Evaluate the classifier
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")


#Parameter
feture_resolution = 50
n_estimators = 100
features_dim = 1 #all, 0, 1 are useable parameter. 0: only h0, 1: only h1, all: h0 and h1.
#Folders of images with different labels, here Ok and Error
folders = ['filepath/OK', 'filepath/Error']
save_model_path = 'model_homology.joblib'

image_paths = []
labels = []
for l in range(0,2):
    folder_dir = folders[l]
    for s in os.listdir(folder_dir):
        image_paths.append(folder_dir + '/' + s)
        labels.append(l)
#shuffle images+labels together
image_paths, labels = shuffle(np.array(image_paths), np.array(labels))

start_time = time.time()
train_model(image_paths, labels, save_model_path, feture_resolution, features_dim, n_estimators)
print("--- %s seconds ---" % (time.time() - start_time))   
