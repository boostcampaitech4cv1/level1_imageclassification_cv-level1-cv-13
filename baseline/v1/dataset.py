import os # 파일 불러오기, 경로 설정하는 모듈
from enum import Enum # 알아봐야함
from typing import Tuple # 타입 체킹?? 

import numpy as np
from PIL import Image # 이미지처리 모듈
from torch.utils.data import Dataset, Subset, random_split # subset ??? random_split : 랜덤으로 분리
from torchvision.transforms import Resize, ToTensor, Normalize, Compose # normalize 정규화 평균화 값을 좀 고르게 

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]

# 이미지 파일인지 확인??
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    # endswith : 확장자를 추출하여 확장자가 IMG_EXTENSION 안에 있는지 확인, True, False 반환
    # 이미지 파일인지 확인

# 기본 Augmentation
class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            Resize(resize, Image.BILINEAR), # BILINEAR : 이미지 사이즈 변경시 쓰는 필터
            ToTensor(), # 이미지를 Tensor
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

# 라벨을 숫자값으로 바꾼다, Enum 을 이런식으로 쓴다. 
class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod # 클래스 메소드라고 지정
    def from_str(cls, value: str) -> int: # value 가 들어갔을 때 str 을 int 로 변환, 또는 에러 반환
        value = value.lower()
        if value == "male":
            return cls.MALE # 0 반환
        elif value == "female":
            return cls.FEMALE # 1 반환
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2): # val_ratio : validation ratio
        # mean : 평균값, std : 표준편차 값 , 괄호 값은 RGB 아마 EDA 단계에서 구하지 않았을까?
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir) # listdir : 디렉토리 내 하위 디렉토리를 리스트로 반환하는 함수

        # 폴더 내 파일이 여러 개 있어서 for 문을 돌린다
        for profile in profiles: # 상위폴더
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile) # 상대 경로에 파일명을 붙여서 경로를 반환해준다.
            for file_name in os.listdir(img_folder): # 하위 폴더
                # 여기서 파일을 읽어온다
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name] # 라벨 선언, 여기서 enum 을 쓰나??

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    # 이미지 파일의 RGB 값의 평균과 표준편차를 계산하는 함수
    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None # True, False 반환
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = [] # 루트
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255 # 평균을 최대값(255) 로 나눠준 값, 1보다 작은 값으로 맞춰준다.
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255 # 표준편차를 최대값(255) 로 나눠준 값

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index): # __getitem__ : 특정 인덱스에 있는 값을 반환해주는 매직 메소드
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        # 이미지, 이미지의 라벨을 반환
        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label # 멀티 클래스

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set



class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
