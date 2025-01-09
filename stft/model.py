import torch
import torch.nn as nn

from torchvision import models

import mlflow

import utils


def create_classifier(in_features, num_classes, dropout=0.5):
    """간단한 분류기(classifier)를 생성하는 함수
        dropout과 Linear Layer를 포함한 모델을 반환합니다.
    Args:
        in_features (int): 입력 특성의 수
        num_classes (int): 출력 class의 수
        dropout (float): dropout 비율. 과적합을 방지하고 일반화 성능 높이는 용도
    Returns:
        torch.nn.Sequential: dropout과 Linear Layer로 구성된 분류기
    Notes:
        - nn.Dropout: 주어진 비율(dropout)에 따라 입력 뉴런을 무작위로 비활성화합니다.
        - nn.Linear: 입력 특징(in_features)을 출력 클래스(num_classes)로 매핑하는 선형 계층입니다.
        - 이 함수는 간단한 구조의 분류기를 생성하며, 백본 모델(예: ResNet, EfficientNet)의 헤드로 사용될 수 있습니다.
    """
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))


class StackedDCNN(nn.Module):
    """간단한 Deep CNN 모델. 이미지 다운샘플링과 특징 추출 후, 최종 분류를 수행합니다.
    Args:
        num_classes(int): 출력 클래스의 수. 기본값 3
    Attributes:
        expected_input_size (tuple): 입력 이미지의 기대 크기 (채널 수, 높이, 너비). (3, 224, 224)로 설정.
        features (torch.nn.Sequential): 합성곱(Convolutional) 및 풀링(Pooling)을 사용하여 특징을 추출하는 부분.
        classifier (torch.nn.Sequential): 특징을 바탕으로 최종 분류를 수행하는 완전연결 레이어(FC Layer).

    Methods:
        forward(x):
            모델의 순전파(forward pass)를 정의합니다.
    """

    def __init__(self, num_classes=3):
        super(StackedDCNN, self).__init__()
        self.expected_input_size = (3, 224, 224)
        self.features = nn.Sequential(  # 이미지 다운샘플링
            nn.Conv2d(
                3, 64, kernel_size=3, padding=1
            ),  # 2D 합성곱 레이어 (저수준 특징)
            nn.ReLU(inplace=True),  # 활성화 함수
            nn.MaxPool2d(kernel_size=2, stride=2),  # 공간 크기 줄이기
            nn.Conv2d(
                64, 128, kernel_size=3, padding=1
            ),  # 2D 합성곱 레이어 (중간 수준 특징)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                128, 256, kernel_size=3, padding=1
            ),  # 2D 합성곱 레이어 (고수준 특징)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.classifier = nn.Sequential(  # 최종 분류 수행
            nn.Linear(256 * 7 * 7, 128),  # Layer 0
            nn.ReLU(),  # Layer 1
            nn.Dropout(0.5),  # Layer 2
            nn.Linear(128, num_classes),  # Layer 3
        )
        self.preprocessor = utils.ImagePreprocessor()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def predict(self, img):
        """원본 이미지를 입력받아 예측 결과(분류 라벨)를 반환합니다."""
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            x = self.preprocessor.preprocess_image(img)
            x = x.unsqueeze(0).to(device)
            pred = self.forward(x)

        return pred.argmax(dim=1).item()

    def save(self, path):
        """모델을 파일로 저장합니다."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "num_classes": self.classifier[-1].out_features,
            "preprocessor": self.preprocessor,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path):
        """모델을 파일로부터 로드합니다."""
        checkpoint = torch.load(path)
        model = cls(checkpoint["num_classes"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.preprocessor = checkpoint["preprocessor"]
        return model


class StackedDCNNWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def load_context(self, context):
        model_path = context.artifacts["model"]
        self.model = StackedDCNN.load(model_path)

    def predict(self, context, inputs):
        return self.model.predict(inputs)


class ResNet50Model(nn.Module):
    """
    ResNet-50 기반의 분류 모델을 정의하는 클래스.
    기존 ResNet-50 모델의 Fully Connected (FC) 레이어를 사용자 정의 분류기로 교체하여
    원하는 클래스 수에 맞게 최종 출력을 생성합니다.

    Args:
        num_classes (int, optional): 출력 클래스의 수. 기본값은 3.

    Attributes:
        resnet (torch.nn.Module): ResNet-50 모델 구조를 기반으로 하며,
                                  최종 FC 레이어는 사용자 정의 분류기로 교체됩니다.

    Methods:
        forward(x):
            입력 이미지 텐서를 받아 순전파(forward pass)를 수행합니다.
    """

    def __init__(self, num_classes=3):
        super(ResNet50Model, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = create_classifier(num_ftrs, num_classes)
        self.preprocessor = utils.ImagePreprocessor()
        self.num_classes = num_classes

    def forward(self, x):
        return self.resnet(x)

    def predict(self, img):
        """원본 이미지를 입력받아 예측 결과(분류 라벨)를 반환합니다."""
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            x = self.preprocessor.preprocess_image(img)
            x = x.unsqueeze(0).to(device)
            pred = self.forward(x)

        return pred.argmax(dim=1).item()

    def save(self, path):
        """모델을 파일로 저장합니다."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "num_classes": self.num_classes,
            "preprocessor": self.preprocessor,
            "architecure": "resnet50",
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path):
        """모델을 파일로부터 로드합니다."""
        checkpoint = torch.load(path)
        model = cls(checkpoint["num_classes"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.preprocessor = checkpoint["preprocessor"]
        return model


class ResNet50Wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def load_context(self, context):
        model_path = context.artifacts["model"]
        self.model = ResNet50Model.load(model_path)

    def predict(self, context, inputs):
        return self.model.predict(inputs)


class EfficientNetB0Model(nn.Module):
    """
    EfficientNet-B0 기반의 분류 모델을 정의하는 클래스.
    기존 EfficientNet-B0 모델의 분류기(FC 레이어)를 사용자 정의 분류기로 교체하여
    원하는 클래스 수에 맞게 최종 출력을 생성합니다.

    Args:
        num_classes (int, optional): 출력 클래스의 수. 기본값은 3.

    Attributes:
        efficientnet (torch.nn.Module): EfficientNet-B0 모델 구조를 기반으로 하며,
                                        최종 classifier 레이어는 사용자 정의 분류기로 교체됩니다.

    Methods:
        forward(x):
            입력 이미지 텐서를 받아 순전파(forward pass)를 수행합니다.
    """

    def __init__(self, num_classes=3):
        super(EfficientNetB0Model, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = create_classifier(num_ftrs, num_classes)
        self.preprocessor = utils.ImagePreprocessor()

    def forward(self, x):
        return self.efficientnet(x)

    def predict(self, img):
        """원본 이미지를 입력받아 예측 결과(분류 라벨)를 반환합니다."""
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            x = self.preprocessor.preprocess_image(img)
            x = x.unsqueeze(0).to(device)
            pred = self.forward(x)

        return pred.argmax(dim=1).item()

    def save(self, path):
        """모델을 파일로 저장합니다."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "num_classes": self.efficientnet.classifier[-1].out_features,
            "preprocessor": self.preprocessor,
            "architecure": "efficientnet-b0",
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path):
        """모델을 파일로부터 로드합니다."""
        checkpoint = torch.load(path)
        model = cls(checkpoint["num_classes"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.preprocessor = checkpoint["preprocessor"]
        return model


class EfficientNetB0Wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def load_context(self, context):
        model_path = context.artifacts["model"]
        self.model = EfficientNetB0Model.load(model_path)

    def predict(self, context, inputs):
        return self.model.predict(inputs)


# class EfficientNetB1Model(nn.Module):
#     def __init__(self, num_classes=3):
#         super(EfficientNetB0Model, self).__init__()
#         self.efficientnet = models.efficientnet_b1(pretrained=True)
#         num_ftrs = self.efficientnet.classifier[1].in_features
#         self.efficientnet.classifier = create_classifier(num_ftrs, num_classes)

#     def forward(self, x):
#         return self.efficientnet(x)


# class EfficientNetB2Model(nn.Module):
#     def __init__(self, num_classes=3):
#         super(EfficientNetB2Model, self).__init__()
#         self.efficientnet = models.efficientnet_b2(pretrained=True)
#         num_ftrs = self.efficientnet.classifier[1].in_features
#         self.efficientnet.classifier = create_classifier(num_ftrs, num_classes)

#     def forward(self, x):
#         return self.efficientnet(x)


class ViTModel(nn.Module):
    """
    Vision Transformer (ViT) 기반의 분류 모델을 정의하는 클래스.
    기존 Vision Transformer의 헤드(FC 레이어)를 사용자 정의 분류기로 교체하여
    원하는 클래스 수에 맞게 최종 출력을 생성합니다.

    Args:
        num_classes (int, optional): 출력 클래스의 수. 기본값은 3.

    Attributes:
        vit (torch.nn.Module): 사전 학습된 Vision Transformer (ViT-B/16) 모델로,
                               최종 heads.head 레이어가 사용자 정의 분류기로 교체됩니다.

    Methods:
        forward(x):
            입력 이미지 텐서를 받아 순전파(forward pass)를 수행합니다.
    """

    def __init__(self, num_classes=3):
        super(ViTModel, self).__init__()
        self.vit = models.vit_b_16(pretrained=True)
        num_ftrs = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.vit(x)


def get_scripted_model(model):
    """PyTorch 모델을 TorchScript 형식으로 변환하는 함수.
    TorchScript는 PyTorch 모델을 직렬화(serialization)하여 추론(inference) 성능을 최적화하고
    플랫폼 독립적으로 모델을 배포할 수 있도록 합니다
    Args:
        model (torch.nn.Module): TorchScript로 변환할 PyTorch 모델
    Returns:
        torch.jit.ScriptModule: TorchScript 형식으로 변환된 모델
    Notes:
        - `torch.jit.script`는 모델의 전체 구조를 스크립트로 변환합니다.
        - TorchScript 모델은 Python 인터프리터가 필요하지 않아 C++ 환경에서도 실행될 수 있습니다.
        - 모델이 동적 제어 흐름을 포함하면 `torch.jit.trace` 대신 `torch.jit.script`를 사용하는 것이 적합합니다.
    """
    return torch.jit.script(model)
