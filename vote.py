import numpy as np
import cv2
import os
test_dir = '/home/aistudio/data/test_image/'
sub_dir = '/home/aistudio/work/vote0805'
if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)
test_list = os.listdir(test_dir)
model_lists = ['/home/aistudio/work/output_fcn2_pl_msf_4k_0.76667_0805/result_1/results/',
               '/home/aistudio/work/outputB_fcn2_pl_0730_0.7683/result_1/results/',
                '/home/aistudio/work/output_fcn2_pl_0731/result_1/results/',
                '/home/aistudio/work/output_fcn2_pl_0805_f/result_1/results/',
                '/home/aistudio/work/output_fcn2_pl_0801/result_1/results/']
# each mask has 20 classes: 0~19
def vote_per_image():   
    for index in range(len(test_list)):
        result_list = []
        test_name = test_list[index]
        print(index,test_name)
        model_logits1 = cv2.imread(model_lists[0] + test_name.replace('.JPEG', '.jpg'),0)
        model_logits2 = cv2.imread(model_lists[1] + test_name.replace('.JPEG', '.jpg'),0)
        model_logits3 = cv2.imread(model_lists[2] + test_name.replace('.JPEG', '.jpg'),0)
        model_logits4 = cv2.imread(model_lists[3] + test_name.replace('.JPEG', '.jpg'),0)
        model_logits5 = cv2.imread(model_lists[4] + test_name.replace('.JPEG', '.jpg'),0)
        result_list.append(model_logits1)
        result_list.append(model_logits2)
        result_list.append(model_logits3)
        result_list.append(model_logits4)
        result_list.append(model_logits5)
        # each pixel
        #print(result_list[0])
        height, width = result_list[0].shape
        vote_mask = np.zeros((height, width))
        #print(result_list)

        for h in range(height):
            for w in range(width):
                record = np.zeros((1, 256))
                for n in range(len(result_list)):
                    mask = result_list[n]
                    pixel = mask[h, w]
                # print('pix:',pixel)
                    record[0, pixel] += 1
                label = record.argmax()
                # print(label)
                vote_mask[h, w] = label
        cv2.imwrite(os.path.join(sub_dir, test_name.replace('.JPEG', '.jpg')), vote_mask)
vote_per_image()
