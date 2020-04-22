from skimage.transform import resize
import numpy as np
import tensorflow as tf
class Processing:       

    def preprocessing(self,x):
        ''' preprocess the image data 
        Args:
            x: images
         
        Return:
            x: preprocessed iamges  
        '''    
        # TODO: implement your own preprocessing method    
        return x
    
    def postprocessing(self,y):
        '''  postprocess the predicted target 
        Args:
            x: your predictions
         
        Return:
            x: he predicted class (0 / 1)
        '''    
        # TODO: implement your own postprocessing method 
        return y
