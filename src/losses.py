import torch
import wandb
import cv2
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as F

from facenet_pytorch import MTCNN
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as TF



import kornia.augmentation as K
import lpips

from arcface import Backbone
from utils_face.utils import extract_faces_and_landmarks

# class Loss:
#     """
#     General purpose loss class. 
#     Mainly handles dtype and visualize_every_k.
#     keeps current iteration of loss, mainly for visualization purposes.
#     """
#     def __init__(self, visualize_every_k=-1, dtype=torch.float32, accelerator=None):
#         self.visualize_every_k = visualize_every_k
#         self.iteration = -1
#         self.dtype=dtype
#         self.accelerator = accelerator
        
#     def __call__(self, **kwargs):
#         self.iteration += 1
#         return self.forward(**kwargs)

class IDLoss():
    """
    Use pretrained facenet model to extract features from the face of the predicted image and target image.
    Facenet expects 112x112 images, so we crop the face using MTCNN and resize it to 112x112.
    Then we use the cosine similarity between the features to calculate the loss. (The cosine similarity is 1 - cosine distance).
    Also notice that the outputs of facenet are normalized so the dot product is the same as cosine distance.
    """
    def __init__(self, pretrained_arcface_path: str, skip_not_found=True, accelerator=None, dtype=torch.float32):
        self.dtype = dtype
        self.accelerator = accelerator
        
        assert pretrained_arcface_path is not None, "please pass `pretrained_arcface_path` in the losses config. You can download the pretrained model from "\
            "https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing"
        self.mtcnn = MTCNN(device=self.accelerator.device)
        self.mtcnn.forward = self.mtcnn.detect
        self.facenet_input_size = 112  # Has to be 112, can't find weights for 224 size.
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(pretrained_arcface_path))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((self.facenet_input_size, self.facenet_input_size))
        self.facenet.requires_grad_(False)
        self.facenet.eval()
        self.facenet.to(device=self.accelerator.device, dtype=self.dtype)  # not implemented for half precision
        self.face_pool.to(device=self.accelerator.device, dtype=self.dtype)  # not implemented for half precision
        self.visualization_resize = transforms.Resize((self.facenet_input_size, self.facenet_input_size), interpolation=transforms.InterpolationMode.BICUBIC)
        self.reference_facial_points = np.array([[38.29459953, 51.69630051],
                                                 [72.53179932, 51.50139999],
                                                 [56.02519989, 71.73660278],
                                                 [41.54930115, 92.3655014],
                                                 [70.72990036, 92.20410156]
                                                 ])  # Original points are 112 * 96 added 8 to the x axis to make it 112 * 112
        self.facenet, self.face_pool, self.mtcnn = self.accelerator.prepare(self.facenet, self.face_pool, self.mtcnn)

        self.skip_not_found = skip_not_found
    
    def extract_feats(self, x: torch.Tensor):
        """
        Extract features from the face of the image using facenet model.
        """
        x = self.face_pool(x)
        x_feats = self.facenet(x)

        return x_feats

    def __call__(
        self, 
        predicted_pixel_values: torch.Tensor,
        encoder_pixel_values: torch.Tensor,
    ):
        encoder_pixel_values = encoder_pixel_values.to(dtype=self.dtype)
        predicted_pixel_values = predicted_pixel_values.to(dtype=self.dtype)

        predicted_pixel_values_face, predicted_invalid_indices = extract_faces_and_landmarks(predicted_pixel_values, mtcnn=self.mtcnn)
        with torch.no_grad():
            encoder_pixel_values_face, source_invalid_indices = extract_faces_and_landmarks(encoder_pixel_values, mtcnn=self.mtcnn)
        
        if self.skip_not_found:
            valid_indices = []
            for i in range(predicted_pixel_values.shape[0]):
                if i not in predicted_invalid_indices and i not in source_invalid_indices:
                    valid_indices.append(i)
        else:
            valid_indices = list(range(predicted_pixel_values))
            
        valid_indices = torch.tensor(valid_indices).to(device=predicted_pixel_values.device)

        if len(valid_indices) == 0:
            loss =  (predicted_pixel_values_face * 0.0).mean()  # It's done this way so the `backwards` will delete the computation graph of the predicted_pixel_values.
            return loss

        # with torch.no_grad():
        #     pixel_values_feats = self.extract_feats(encoder_pixel_values_face[valid_indices])

        pixel_values_feats = self.extract_feats(encoder_pixel_values_face[valid_indices])
            
        predicted_pixel_values_feats = self.extract_feats(predicted_pixel_values_face[valid_indices])
        loss = 1 - torch.einsum("bi,bi->b", pixel_values_feats, predicted_pixel_values_feats)

        return loss.mean()
    

class DiNOLoss():
    def __init__(self, device):
        self.model_dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.model_dino.eval()
        self.model_dino.to(device)
        self.device = device
        
        # self.transform = transforms.Compose([
        #     transforms.Resize(256, interpolation=3),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])

    def encode(self, image_input):
        image_features = self.model_dino(image_input)
        return image_features
    
    def process_img(self, img):
        # pil_img = to_pil_image((img * 255).clamp(0, 255).to(torch.uint8))
        # breakpoint()
        # img_input = self.transform(pil_img).unsqueeze(0).to(self.device)

        # this is to ensure gradient still exist 
        img = F.interpolate(img.unsqueeze(0), size=256, mode='bilinear', align_corners=False).squeeze(0)
        img = TF.center_crop(img, 224)  # Center crop supports tensors
        img = TF.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        img = img.unsqueeze(0).to(self.device)

        return img

    def __call__(self, ref_img, pred_img):
        # convert torch image to pil format

        bsz = ref_img.shape[0]

        ref_img_inputs = []
        pred_img_inputs = []

        for i in range(bsz):
            ref_img_inputs.append(self.process_img(ref_img[i]))
            pred_img_inputs.append(self.process_img(pred_img[i]))
        
        ref_img_inputs = torch.cat(ref_img_inputs)
        pred_img_inputs = torch.cat(pred_img_inputs)

        ref_features = self.encode(ref_img_inputs)
        pred_features = self.encode(pred_img_inputs)
        
        sim = F.cosine_similarity(pred_features, ref_features, dim=-1)
        losses = 1 - sim.mean()

        return losses