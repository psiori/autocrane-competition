import os
import imageio
import numpy as np
from sklearn.metrics import accuracy_score


    
class DataLoader:
    def _load_from_directory(self, path, label):
        images = []
        y = []
        for r, d, f in os.walk(path):
            for file in f:
                if ".png" in file:
                    img = imageio.imread(path + "/" + file)
                    images.append(img)
                    y.append(label)
        return np.array(images), np.array(y)
    
    def load_data_from_files(self,path_to_images ):
        images_not_loaded, y_not_loaded = self._load_from_directory(path_to_images+'/not_loaded', 0 )
        images_loaded, y_loaded = self._load_from_directory(path_to_images+'/loaded', 1 )

        images = np.concatenate((images_not_loaded, images_loaded), axis=0)
        y = np.concatenate((y_not_loaded, y_loaded), axis=0)

        return np.array(images), np.array(y)
    
class ValidationHelper: 
    def evaluate(self,y_truth, y_prediction):
        ''' calculate classification score 
        Args:
            y_truth: ground truth data
            y_prediction: processed predictions
         
        Return:
            score 
        '''    
        return accuracy_score(y_truth,y_prediction)
    