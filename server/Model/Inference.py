import torch
from torchvision import models
import torch.nn.functional as F


class Inference :
    def __init__(self, checkpoint) :
        self.checkpoint = checkpoint
    
        self.teacher_model =  models.resnet18(pretrained=True)
        for param in self.teacher_model.parameters() :
            param.requires_grad = False
        self.student_model =  models.resnet18(pretrained=False)
        self.student_model.load_state_dict(self.checkpoint)

        self.features_t = {}
        self.features_s = {}

        self.teacher_forward()
        self.student_forward()

    def teacher_features(self, layer_name) :        
        def hook(model, input, output) :
            self.features_t[layer_name] = output
        return hook

    def student_features(self, layer_name) :        
        def hook(model, input, output) :
            self.features_s[layer_name] = output
        return hook

    def teacher_forward(self) :
        self.teacher_model.layer1[-1].register_forward_hook(self.teacher_features('layer_1'))
        self.teacher_model.layer2[-1].register_forward_hook(self.teacher_features('layer_2'))
        self.teacher_model.layer3[-1].register_forward_hook(self.teacher_features('layer_3'))

    def student_forward(self) :
        self.student_model.layer1[-1].register_forward_hook(self.student_features('layer_1'))
        self.student_model.layer2[-1].register_forward_hook(self.student_features('layer_2'))
        self.student_model.layer3[-1].register_forward_hook(self.student_features('layer_3'))



    def get_features(self, img) :
        t_out = self.teacher_model(img)
        s_out = self.student_model(img)  
        return self.features_t, self.features_s  

    def anomaly_map(self, img, out_size = 256) :
        teacher_features, student_features = self.get_features(img)
        anomaly_map = torch.ones([1, 1, out_size, out_size])
        maps = []
        for key in teacher_features.keys() :
            f_1 = F.normalize(teacher_features[key], p = 2)
            f_2 = F.normalize(student_features[key], p = 2)
            a_map = 1 - F.cosine_similarity(f_1, f_2) 
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear')
            maps.append(a_map)
            anomaly_map *= a_map
        return maps, anomaly_map
