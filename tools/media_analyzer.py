import torch
import clip
import cv2
import json
import numpy as np
from PIL import Image
from typing import Dict, Any

classes: dict[Any, Any] = {}

with open("data\\classes.json", "r") as file:
    classes = json.load(file)


class NeuroCLIP:
    def __init__(self, classes_list=classes, model_name="ViT-B/32", priority_cuda=True):
        """
           Инициализация класса.

           Аргументы:
               classes_list (dict): словарь классов и подклассов.
               model_name (str): название модели.
               priority_cuda (bool): приоритетность CUDA-ядер. Если они есть, и параметр равен True, они буду использоваться.

           Возвращается:
               Nonr
        """
        self.device = "cuda" if (torch.cuda.is_available() and priority_cuda) else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.classes = classes_list

    def _get_similarity(self, image, class_list):
        """
           Возращает вероятности для всех объектов.

           Аргументы:
               image (PIL.open):  нужное изображение для анализ.
               class_list (list): список объектов.

           Возвращается:
               torch.Tensor: проценты (0...1).
           """
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image_tensor)

        text = clip.tokenize(class_list).to(self.device)
        text_features = self.model.encode_text(text)

        return (image_features @ text_features.T).softmax(dim=-1)

    def _get_best_result(self, similarity, class_list):
        """
           Возращает вероятности для объектов с фильтрацией по чувствительности.

           Аргументы:
               similarity (torch.Tensor): вероятности объектов.
               class_list: список оьъектов.

           Возвращается:
               dict: ключ - название объекта, значение - вероятность.
        """
        similarity_dict = {class_list[i]: similarity[0][i].item() for i in range(len(class_list))}
        similarity_sorted = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)
        result = list(filter(lambda x: x[1] >= self.tolerance, similarity_sorted))

        result = {key: value for key, value in result}

        return result

    def analyze(self, image_name="image.jpg", tolerance=0.7, is_percent=True):
        """
           Возращает самые подходящие объекты.

           Аргументы:
               image_name (str): название файла.
               tolerance (float): чувстительность (если меньше ее, то объекты не попадают в список). Допустимые значения 0..1.
               is_percent (bool): сохранять ли проценты в формате 0..100 (93.5 вместо 0.935).
           Возвращается:
               dict: {"classes": {"class1": 0.1, "class2": 0.2, ...}, "subclasses": {"subclass1": 0.1, "subclass2": 0.2, ...}}
        """
        self.tolerance = tolerance

        image = Image.open(image_name)
        classes = list(self.classes.keys())

        similarity = self._get_similarity(image, classes)
        answer_classes = self._get_best_result(similarity, classes)

        subclasses = []
        for key in answer_classes.keys():
            new = [f"{key} {i}" if key != i else f"{i}" for i in self.classes[key]]
            subclasses.extend(new)
        similarity_subclasses = self._get_similarity(image, subclasses)
        answer_subclasses = self._get_best_result(similarity_subclasses, subclasses)

        result_subclasses = {}
        for subclass, value in answer_subclasses.items():
            for class_, factor in answer_classes.items():
                if subclass.startswith(class_):
                    product = value * factor
                    if product >= tolerance:
                        result_subclasses[subclass] = product

        answer = {"class": answer_classes, "subclass": result_subclasses}
        if is_percent:
            for (key, subclass) in answer.items():
                for key_sub, value in subclass.items():
                    answer[key][key_sub] = round(value * 100, 1)

        return answer

    @property
    def name(self):
        return "CLIP"


class NeuroYOLO:
    def __init__(self, weights='data\\yolo\\yolov3.weights', config='data\\yolo\\yolov3.cfg',
                 coco='data\\yolo\\coco.names'):
        """
           Инициализация класса.

           Аргументы:
               weights (str): путь к весам.
               config  (str): путь к концифигу.
               coco    (str): путь к названиям оъектов.

           Возвращается:
               None
        """
        self.weights = weights
        self.config = config
        self.coco = coco

    def analyze(self, filename, tolerance=0.7, is_percent=True):
        """
           Возращает объекты на фото с вероятностями.

           Аргументы:
               filename (str): имя файла.
               tolerance (float): чувстительность. Допустимое значение 0..1
               is_percent (bool): сохранять ли проценты в формате 0..100 (93.5 вместо 0.935).

           Возвращается:
               dict: {"object1": 0.1, "object2": 0.2, ...}
        """
        net = cv2.dnn.readNet(self.weights, self.config)

        classes = []
        with open(self.coco, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        img = cv2.imread(filename)
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence >= tolerance:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        objects = {}
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                label = str(classes[class_ids[i]])
                objects[label] = round(confidences[i] * 100, 1) if is_percent else confidences[i]

        return objects

    @property
    def name(self):
        return "YOLO"
