import os
import imageio
import numpy as np
import xml.etree.ElementTree as ET
    
class DataLoader:
    def xml_to_coordinates_bounding_box(self,file_to_xml_data):
        """ Read relevant Labeldata from xml-file
        Args:
            file_to_xml_data: path to XML-File
         
        Return:
            Boundigbox: ymin, xmin, ymax, xmax
        """
        root = ET.parse(file_to_xml_data).getroot()
        xmin = int(root.find(".//bndbox/xmin").text)
        ymin = int(root.find(".//bndbox/ymin").text)
        xmax = int(root.find(".//bndbox/xmax").text)
        ymax = int(root.find(".//bndbox/ymax").text)
        img_filename = root.find("filename").text
        
        return [ymin, xmin, ymax, xmax]

    def load_data_from_files(self,path_to_images, path_to_labels):
        images = []
        y = []
        for r, d, f in os.walk(path_to_images):
            for file in f:
                if ".png" in file:
                    try:                        
                        img = imageio.imread(path_to_images + "/" + file)
                        coordinateBoundingbox = self.xml_to_coordinates_bounding_box( path_to_labels + "/" + file.replace(".png", ".xml"))
                        images.append(img)
                        y.append(coordinateBoundingbox)
                    except Exception as exe:
                        print(exe)
                        
        return np.array(images), np.array(y)
    
class ValidationHelper:
    def calc_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        yA = max(boxA[0], boxB[0])
        xA = max(boxA[1], boxB[1])
        yB = min(boxA[2], boxB[2])
        xB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        #        =    ymax -  ymin            xmax    - xmin
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def _is_prediction_correct(self, iou, iou_threshold=0.8):
        return True if iou >= iou_threshold else False

    def _get_prediction_quality(self,boxesA, boxesB, iou_threshold=0.8):
        all_iou = []
        pred_res = []
        for loop_counter in range(len(boxesA)):
            iou = self.calc_intersection_over_union(
                    boxesA[loop_counter], boxesB[loop_counter]
                )
            all_iou.append(iou)
            is_correct = self._is_prediction_correct(iou, iou_threshold)
            pred_res.append(is_correct)
        return np.array(all_iou), np.array(pred_res)

    def _get_correct_prediction_count(self,y_truth, y_predictions, iou_threshold):
        res = self._get_prediction_quality(y_truth, y_predictions, iou_threshold)

        mask_res = res[1] == True
        correct_prediction_count = len(res[1][mask_res])
        wrong_predictions_count = len(res[1][~mask_res])

        return correct_prediction_count, wrong_predictions_count

    def evaluate(self,y_truth, y_prediction):
        ''' calculate detection score 
        Args:
            y_truth: ground truth data
            y_prediction: processed predictions
         
        Return:
            score 
        ''' 
        IoU_Threshold = 0.8
        correct_predictions, wrong_predictions = self._get_correct_prediction_count(y_truth, y_prediction, IoU_Threshold)
        rec = correct_predictions / len(y_truth)
        return rec
    