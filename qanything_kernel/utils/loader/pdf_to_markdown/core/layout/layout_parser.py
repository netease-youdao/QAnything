import numpy as np
import cv2
import math
from layout.layout_model import YOLOV8
from layout.table_cls.infer_onnx import TableCls
from layout.table_rec.pipeline import TableParser
import torch




def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


class LayoutParser(object):
    def __init__(self,device=torch.device("cuda")):
        model_path = 'layout/best.onnx'
        self.model = YOLOV8(model_path, conf_thres=0.4, iou_thres=0.65,imgsz=(800, 800),device=device)
        self.class_names = ('Text', 'Title', 'Figure', 'Equation', 'Table','Caption', 'Header', 'Footer', 'BibInfo', 'Reference','Content', 'Code', 'Other', 'Item', 'Author',)
        
        self.table_cls = TableCls(device=device)
        self.table_parse = TableParser(device=device)


    def process_image(self,image):
        boxes, scores, class_ids = self.model(image)
        for idx, box in enumerate(boxes):
            # print(box)
            color = np.random.rand(3) * 160
            image = np.int32(image)
            image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color)
            image = cv2.putText(image, '%.3f %s' % (scores[idx], self.class_names[class_ids[idx]]), 
                (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # cv2.imwrite('tmp_res.jpg',image)

    def get_min_dis_box(self,caption_boxes,box):
        dis = 1e6
        min_box = None
        for c_box in caption_boxes:
            # print(c_box)
            c_x, c_y = (c_box[0] + c_box[2]) / 2, (c_box[1] + c_box[3]) / 2
            x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            c_dis = distance(c_x, c_y, x, y)
            # print('*********')
            # print(box)
            # print(c_box)
            # print(c_dis)
            if c_dis < dis:
                min_box = c_box
                dis = c_dis
        return min_box

    def extract_table(self,image,ocr_result):
        boxes, scores, class_ids = self.model(image.copy())
        table_dict = {}
        index = 0
        caption_boxes = []
        for idx,(box,class_id) in enumerate(zip(boxes,class_ids)):
            if self.class_names[class_ids[idx]] == 'Caption':
                caption_boxes.append(box)
        # print(len(caption_boxes))
        # print(caption_boxes)
        # vis_img = image.copy()
        # for idx, box in enumerate(boxes):
        #     color = np.random.rand(3) * 160
        #     cv2.rectangle(vis_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color)
        #     cv2.putText(vis_img, '%.3f %s' % (scores[idx], self.class_names[class_ids[idx]]), 
        #         (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # cv2.imwrite('layout_vis.jpg', vis_img)
        # print(caption_boxes)
        for idx, box in enumerate(boxes):
            if self.class_names[class_ids[idx]] == 'Table':
                # print(box)
                x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
                # print(x1,y1,x2,y2)
                table_image = image[y1:y2,x1:x2,:]

                # cv2.imwrite('/ssd8/exec/qinhaibo/code/RAG/QAnything-master/tools/TreeIndex/paddle_test/table_tmp3.jpg',table_image)
                table_type = self.table_cls.process(table_image.copy())
                # print(table_type)
                # cv2.imwrite('/ssd8/exec/qinhaibo/code/RAG/QAnything-master/tools/TreeIndex/paddle_test/table_tmp4.jpg',table_image.copy())
                # ti = cv2.imread('/ssd8/exec/qinhaibo/code/RAG/QAnything-master/tools/TreeIndex/paddle_test/table_tmp3.jpg')
                # table_html,table_markdown = self.table_parse.process(ti,table_type,convert2markdown=True)
                table_html,table_markdown = self.table_parse.process(table_image.copy(),table_type,ocr_result=ocr_result,convert2markdown=True)
                # print(table_str)
                if len(caption_boxes) == 0:
                    related_caption_box = box
                    has_caption = False
                else:
                    related_caption_box = self.get_min_dis_box(caption_boxes,box)
                    has_caption = True
                c_x1,c_y1,c_x2,c_y2 = int(related_caption_box[0]),int(related_caption_box[1]),int(related_caption_box[2]),int(related_caption_box[3])
                caption_image =  image[c_y1:c_y2, c_x1:c_x2]
                table_dict[index] = {
                    'has_caption': has_caption,
                    'table_html': table_html,
                    'table_markdown': table_markdown,
                    'table_box': box,
                    'caption_box': related_caption_box,
                    'caption_image': caption_image,
                }
                index +=1
        # print(table_dict[0]['table_content'])
        # print(table_dict[0]['caption_box'])
        # print(len(table_dict.keys()))
        #对table dict按照box的左上角顶点进行排序
        return table_dict