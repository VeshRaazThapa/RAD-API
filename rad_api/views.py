from django.http import JsonResponse
import cv2

# from rest_framework import viewsets
# from .serializers import ReportSerializer, ReportPageSerializer, PublicLinkSerializer
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from django.views.decorators.csrf import csrf_exempt
import os
import pickle

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
PATH = '/Users/stuff/Desktop/Research/Projects/rad_api/assets'


class CAM(nn.Module):
    def __init__(self, model_to_convert, get_fc_layer=lambda m: m.classifier, score_fn=F.softmax, resize=True):
        super().__init__()
        self.backbone = nn.Sequential(*list(model_to_convert.children())[:-1])
        self.fc = get_fc_layer(model_to_convert)
        self.conv = nn.Conv2d(self.fc.in_features, self.fc.out_features, kernel_size=1)
        self.conv.weight = nn.Parameter(self.fc.weight.data.unsqueeze(-1).unsqueeze(-1))
        self.conv.bias = self.fc.bias
        self.score_fn = score_fn
        self.resize = resize
        self.eval()

    def forward(self, x, out_size=None):
        batch_size, c, *size = x.size()
        feat = self.backbone(x)
        cmap = self.score_fn(self.conv(feat))
        if self.resize:
            if out_size is None:
                out_size = size
            cmap = F.upsample(cmap, size=out_size, mode='bilinear')
        pooled = F.adaptive_avg_pool2d(feat, output_size=1)
        flatten = pooled.view(batch_size, -1)
        cls_score = self.score_fn(self.fc(flatten))
        weighted_cmap = (cmap * cls_score.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

        return cmap, cls_score, weighted_cmap


@csrf_exempt
def get_model_prediction(request):
    if request.method == "POST":
        image_path = request.POST.get('id')
        x_ray_type = request.POST.get('type')
        # print(request.POST.get('files',None))

        saved_model_path = "{}/{}_model.pt".format(PATH, x_ray_type)
        model = models.densenet169(pretrained=False)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 2)
        model.load_state_dict(torch.load(saved_model_path, map_location=torch.device('cpu'), ))
        model = model.cpu()
        # evaluate(model, valid_loader)

        target_size = (224, 224)
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

        transform = transforms.Compose([transforms.Resize(target_size),
                                        transforms.CenterCrop(target_size),
                                        transforms.ToTensor(),
                                        normalize])

        cam = CAM(model)
        path = image_path
        img = Image.open(path).convert("RGB")
        with torch.no_grad():
            img_v = Variable(transform(img).unsqueeze(0), volatile=True, requires_grad=False).cpu()
            cmap, score, weighted_cmap = cam.forward(img_v)

            # images
            # heated map image
            color_map = weighted_cmap.data.cpu().numpy()[0]

            # background image
            background = np.array(img.resize(target_size))

            """ combining foreground and background image"""
            # foreground image
            img = Image.fromarray(color_map)
            target_size = (224, 224)
            ar_pro2d = np.array(img.resize(target_size))
            print(ar_pro2d.shape)
            plt.imsave('foreground.png', ar_pro2d)

            # background image
            plt.imsave('background.png', background)

            #combinition
            background = cv2.imread('background.png')
            overlay = cv2.imread('foreground.png')
            print(background.shape)
            print(overlay.shape)
            added_image = cv2.addWeighted(background, 1, overlay, 0.3, 1)


            # Pickle dictionary using protocol 0.
            pickle.dump(color_map, open('heated_map_image3.pickle', 'wb'))
            pickle.dump(background, open('background_image3.pickle', 'wb'))

            image_name = image_path.split('/')[-1].split('.')[0]
            print(image_name)
            label = 'Abnormal' if (float(score.numpy()[0][1]) > 0.10) else  'Normal'
            if 'patient101' in image_name:
                highlighted_image_name='patient101_highlighted_image.png'
            else:
                highlighted_image_name = image_name + '_highlighted_image.png'
                # plt.imsave('static/'+highlighted_image_name, background[:, :, 1] * 10)
                ## final image
                cv2.imwrite('static/'+highlighted_image_name, added_image)



            return JsonResponse({
                "success": True,
                "prediction": {"probabilities": [float(score.numpy()[0][0]),float(score.numpy()[0][1])], "label": label,
                               "prediction": 0}, "id": image_name,
                "highlighted_imagename": highlighted_image_name
            })

        # dst = background[:, :, 1] * 10  + color_map
        # Image.fromarray(dst.astype(np.uint8)).save('numpy_image_alpha_blend.jpg')
        # plt.imshow(background[:, :, 1] * 10)
        # plt.imshow(color_map, alpha=0.4)
