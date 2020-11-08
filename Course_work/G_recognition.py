import os
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
from facenet_pytorch import MTCNN

sys.path.insert(0, './yolo-hand-detection')

from yolo import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.pool_2 = nn.MaxPool2d(2)
        self.pool_6 = nn.MaxPool2d(6)
        self.drop = torch.nn.Dropout(p=0.3)
        self.flat = nn.Flatten()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(3136, 512)  # 576 - 3136
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))
        x = self.pool_2(x)
        x = self.drop(x)

        x = F.relu(self.conv3(x))
        x = self.pool_6(x)
        x = self.drop(x)

        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))

        return x


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = Net(1, 10).to(device)

    # Создаем объект для считывания потока с веб-камеры(обычно вебкамера идет под номером 0. иногда 1)
    cap = cv2.VideoCapture(0)

    yolo = YOLO("yolo-hand-detection/models/cross-hands.cfg", "yolo-hand-detection/models/cross-hands.weights",
                ["hand"], confidence=0.5, threshold=0.3)


    # Класс детектирования и обработки лица с веб-камеры
    class FaceDetector(object):

        def __init__(self, mtcnn):
            self.mtcnn = mtcnn
            self.hand_detector = yolo
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.emodel = Net(1, 10).to(self.device)
            self.emodel.load_state_dict(torch.load('./Hand_gesture_recognition_model_100_state_10_epoch.pth'))
            self.emodel.eval()

        # Функция рисования найденных параметров на кадре
        def _draw(self, frame, boxes, probs, landmarks):
            try:
                for box, prob, ld in zip(boxes, probs, landmarks):
                    # Рисуем обрамляющий прямоугольник лица на кадре
                    cv2.rectangle(frame,
                                  (box[0], box[1]),
                                  (box[2], box[3]),
                                  (0, 0, 255),
                                  thickness=2)
            except:
                pass
                # print('Something wrong im draw function!')

            return frame

        # Функция для вырезания рук и с кадра
        @staticmethod
        def crop_hand(frame, boxes, exp=0):
            hands = []
            for i, box in enumerate(boxes):
                hands.append(frame[int(box[1]) - exp:int(box[3]) + exp,
                             int(box[0]) - exp:int(box[2]) + exp])
            return hands

        # Словарь жестов
        @staticmethod
        def digit_to_classname(digit):
            digit_to_classname_dict = {0: 'palm', 1: 'l', 2: 'fist', 3: 'fist_moved', 4: 'thumb', 5: 'index',
                                       6: 'ok', 7: 'palm_moved', 8: 'c', 9: 'down'}
            return digit_to_classname_dict[digit]

        # Функция реакции на жест
        def gesture_action(self, digit, frame):
            if digit is None:
                return "", frame

            gesture_to_show = self.digit_to_classname(digit)
            if digit in [3, 5, 7, 9]:  # Данные положения руки плохо определяются детектором, поэтому только надпись
                pass
            if digit == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if digit == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if digit == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if digit == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if digit == 6:
                pass  # Вернуть изначальный вариант
            if digit == 8:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            return gesture_to_show, frame

        # Функция в которой будет происходить процесс считывания и обработки каждого кадра
        def run(self):
            gesture = ''
            gesture_to_show = ''
            gesture_idx = None
            guess_list = np.array([0] * 10)
            x, y = [0, 0]

            # Заходим в бесконечный цикл
            while True:
                # Считываем каждый новый кадр - frame
                # ret - логическая переменая. Смысл - считали ли мы кадр с потока или нет

                ret, frame = cap.read()

                try:
                    # детектируем расположение лица на кадре, вероятности на сколько это лицо
                    # и особенные точки лица
                    boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                    # Ищем лицо и рисуем рамку
                    self._draw(frame, boxes, probs, landmarks)

                    # Если в кадре есть лицо, то считываем жест:
                    if boxes is not None:

                        # детектируем расположение рук на кадре
                        width, height, inference_time, results = self.hand_detector.inference(frame)

                        boxes_hand = []
                        for detection in results:
                            id, name, confidence, x, y, w, h = detection
                            boxes_hand = [[x, y, x + w, y + h]]

                        # Вырезаем руку из кадра с некоторым захватом области exp=30
                        hand = self.crop_hand(frame, boxes_hand, exp=30)[0]
                        # Меняем размер изображения руки для входа в нейронную сеть
                        hand = cv2.resize(hand, (100, 100))
                        # Превращаем в 1-канальное серое изображение
                        hand = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
                        # Превращаем numpy-картинку вырезанной руки в pytorch-тензор
                        torch_hand = torch.from_numpy(hand).unsqueeze(0).to(self.device).float()

                        # Загужаем наш тензор руки в нейронную сеть и получаем предсказание
                        gesture = self.emodel(torch_hand[None, ...])
                        # Заполняем таблици жестов
                        guess_list[int(gesture.argmax())] += 1

                except:
                    pass
                    # print('Something wrong im main cycle!')

                # Получаем жест с наибольшим значением
                if max(guess_list) >= 4:
                    gesture_idx = guess_list.argmax()
                    guess_list = np.array([0] * 10)

                # Реакция на наш жест (вывод названия жеста на экран и изменение параметров выводимого кадра)
                gesture_to_show, changed_frame = self.gesture_action(gesture_idx, frame)
                cv2.putText(changed_frame, gesture_to_show, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (188, 198, 48), 2,
                            cv2.LINE_AA)
                # Показываем кадр в окне, и назвываем его(окно) - 'Gesture Recognition'
                cv2.imshow('Gesture Recognition', changed_frame)

                # Функция, которая проверяет нажатие на клавишу 'q'
                # Если нажатие произошло - выход из цикла. Конец работы приложения
                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    break

            # Очищаем все объекты opencv, что мы создали
            cap.release()
            cv2.destroyAllWindows()


    # Загружаем мтцнн
    mtcnn = MTCNN()
    # Создаем объект нашего класса приложения
    fcd = FaceDetector(mtcnn)
    # Запускаем
    fcd.run()