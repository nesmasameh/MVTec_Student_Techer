from torchvision import models
from torch import nn

class TS_Model(nn.Module) :
    def __init__(self) :
        super(TS_Model, self).__init__()
        
        self.teacher_model = models.resnet18(pretrained=True).eval()
    
        for param in self.teacher_model.parameters() :
            param.requires_grad = False    
       
        self.student_model = models.resnet18(pretrained=False)
   
        self.teacher_outs = {}
        self.student_outs = {}
        
        self.teacher_forward()
        self.student_forward()

    def teacher_features(self, layer_name) :        
        def hook(model, input, output) :
            self.teacher_outs[layer_name] = output
        return hook
        
    def student_features(self, layer_name) :        
        def hook(model, input, output) :
            self.student_outs[layer_name] = output
        return hook

    def teacher_forward(self) :
        self.teacher_model.layer1[-1].register_forward_hook(self.teacher_features('layer_1'))
        self.teacher_model.layer2[-1].register_forward_hook(self.teacher_features('layer_2'))
        self.teacher_model.layer3[-1].register_forward_hook(self.teacher_features('layer_3'))
        
    def student_forward(self) :
        self.student_model.layer1[-1].register_forward_hook(self.student_features('layer_1'))
        self.student_model.layer2[-1].register_forward_hook(self.student_features('layer_2'))
        self.student_model.layer3[-1].register_forward_hook(self.student_features('layer_3'))
    
    def teacher(self, img) :
        t_out = self.teacher_model(img)
        return self.teacher_outs

    def forward(self, img) :
        s_out = self.student_model(img)
        return self.student_outs       