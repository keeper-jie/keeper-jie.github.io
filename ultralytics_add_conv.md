# ultralytics框架新增conv
- 在`ultralytics-8.3.152/ultralytics/nn/modules/conv.py`中新建卷积类，例如参考[https://github.com/AlanLi1997/slim-neck-by-gsconv/blob/master/gsconv-yolov8_9_10_11/ultralytics/nn/modules/conv.py](https://github.com/AlanLi1997/slim-neck-by-gsconv/blob/master/gsconv-yolov8_9_10_11/ultralytics/nn/modules/conv.py)中`GSConvE`
```
class GSConvE(nn.Module):
    '''
    GSConv enhancement for representation learning: generate various receptive-fields and
    texture-features only in one Conv module
    # GSConvE1 https://github.com/AlanLi1997/rethinking-fpn
    '''
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, act)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, 1, bias=False),
            nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        y = torch.cat((x1, x2), dim=1)
        # shuffle
        y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])
```
- 在`conv.py`末尾添加函数测试新卷积输入和输出shape
```
def test_conv():
    """test conv"""
    input_tensor = torch.randn(1, 256, 160, 160)
    model = GSConvE(c1 = 256, c2 = 256, k=3, s=2, p=None, g=1, d=1, act=True)
    output = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)

if __name__=="__main__":
    test_conv()
```
- 运行测试脚本`python ultralytics-8.3.152/ultralytics/nn/modules/conv.py`得到如下结果
```
Input shape: torch.Size([1, 256, 160, 160])
Output shape: torch.Size([1, 256, 80, 80])
```
- 查看框架中模型配置，看看可以替换哪个算子
```
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Ultralytics YOLO11 object detection model with P3/8 - P5/32 outputs
# Model docs: https://docs.ultralytics.com/models/yolo11
# Task docs: https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo1-yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 181 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 181 layers, 9458752 parameters, 9458736 gradients, 2-7 GFLOPs
  m: [0.50, -00, 512] # summary: 231 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  l: [-00, -00, 512] # summary: 357 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  x: [-00, -50, 512] # summary: 357 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# YOLO11n backbone
backbone:
  # [from, repeats, module, args]  input:[1,3,640,640]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2  [1,64,320,320]
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4  [1,128,160,160]
  - [-1, 2, C3k2, [256, False, 0.25]]  # [1,256,160,160]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8  [1,256,80,80]
  - [-1, 2, C3k2, [512, False, 0.25]]  # [1,512,80,80]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16  [1,512,40,40]
  - [-1, 2, C3k2, [512, True]]  # [1,512,40,40]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32  [1,1024,20,20]
  - [-1, 2, C3k2, [1024, True]]  # [1,1024,20,20]
  - [-1, 1, SPPF, [1024, 5]] # 9  [1,1024,20,20]
  - [-1, 2, C2PSA, [1024]] # 10  [1,1024,20,20]

# YOLO11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # [1,1024,40,40]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4  [1,1024+512,40,40]
  - [-1, 2, C3k2, [512, False]] # 13  [1,512,40,40]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # [1,512,80,80]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3  [1,512+512,80,80]
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)  [1,256,80,80]

  - [-1, 1, Conv, [256, 3, 2]]  # [1,256,40,40]
  - [[-1, 13], 1, Concat, [1]] # cat head P4  [1,256+512,40,40]
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)  [1,512,40,40]

  - [-1, 1, Conv, [512, 3, 2]]  # [1,512,20,20]
  - [[-1, 10], 1, Concat, [1]] # cat head P5  [1,512+1024,20,20]
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)  [1,1024,20,20]

  - [[16, 19, 22], 1, Detect, [nc]]
# d.shape: torch.Size([1, 144, 80, 80])
# d.shape: torch.Size([1, 144, 40, 40])
# d.shape: torch.Size([1, 144, 20, 20])
```
- 由输出信息可以看出`GSConvE`可以替换`Conv`做下采样，不过放到哪里需要实验去验证，这里我们直接参考[https://github.com/AlanLi1997/slim-neck-by-gsconv/blob/master/gsconv-yolov8_9_10_11/ultralytics/cfg/models/v8/yolov8-gsconve.yaml](https://github.com/AlanLi1997/slim-neck-by-gsconv/blob/master/gsconv-yolov8_9_10_11/ultralytics/cfg/models/v8/yolov8-gsconve.yaml)进行替换，替换后模型配置如下
```
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Ultralytics YOLO11 object detection model with P3/8 - P5/32 outputs
# Model docs: https://docs.ultralytics.com/models/yolo11
# Task docs: https://docs.ultralytics.com/tasks/detect
# original yolo11n YOLO11_cbam summary: 181 layers, 2,624,080 parameters, 2,624,064 gradients, 6.6 GFLOPs
# 

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo1-yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 181 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 181 layers, 9458752 parameters, 9458736 gradients, 2-7 GFLOPs
  m: [0.50, -00, 512] # summary: 231 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  l: [-00, -00, 512] # summary: 357 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  x: [-00, -50, 512] # summary: 357 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# YOLO11n backbone
backbone:
  # [from, repeats, module, args]  input:[1,3,640,640]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2  [1,64,320,320]
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4  [1,128,160,160]
  - [-1, 2, C3k2, [256, False, 0.25]]  # [1,256,160,160]
  - [-1, 1, GSConvE, [256, 3, 2]] # 3-P3/8  [1,256,80,80]
  - [-1, 2, C3k2, [512, False, 0.25]]  # [1,512,80,80]
  - [-1, 1, GSConvE, [512, 3, 2]] # 5-P4/16  [1,512,40,40]
  - [-1, 2, C3k2, [512, True]]  # [1,512,40,40]
  - [-1, 1, GSConvE, [1024, 3, 2]] # 7-P5/32  [1,1024,20,20]
  - [-1, 2, C3k2, [1024, True]]  # [1,1024,20,20]
  - [-1, 1, SPPF, [1024, 5]] # 9  [1,1024,20,20]
  - [-1, 2, C2PSA, [1024]] # 10  [1,1024,20,20]

# YOLO11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # [1,1024,40,40]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4  [1,1024+512,40,40]
  - [-1, 2, C3k2, [512, False]] # 13  [1,512,40,40]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # [1,512,80,80]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3  [1,512+512,80,80]
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)  [1,256,80,80]

  - [-1, 1, GSConvE, [256, 3, 2]]  # [1,256,40,40]
  - [[-1, 13], 1, Concat, [1]] # cat head P4  [1,256+512,40,40]
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)  [1,512,40,40]

  - [-1, 1, GSConvE, [512, 3, 2]]  # [1,512,20,20]
  - [[-1, 10], 1, Concat, [1]] # cat head P5  [1,512+1024,20,20]
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)  [1,1024,20,20]

  - [[16, 19, 22], 1, Detect, [nc]]
# d.shape: torch.Size([1, 144, 80, 80])
# d.shape: torch.Size([1, 144, 40, 40])
# d.shape: torch.Size([1, 144, 20, 20])
```
- 将`ultralytics-8.3.152/ultralytics/nn/tasks.py`,`ultralytics-8.3.152/ultralytics/nn/modules/__init__.py`.'/data/liujie/data/ultralytics-8.3.152/ultralytics/nn/modules/conv.py'文件中均添加新卷积类，具体代码可以查看我的github仓库
- 新建测试脚本`test_yaml.py`测试模型架构是否配置正确
```
# 查看cfg文件对应的参数量/计算量信息
from ultralytics.nn.tasks import DetectionModel,RTDETRDetectionModel,WorldModel,YOLOESegModel,YOLOEModel
import torch
import numpy as np

def test_yaml(cfg_file_path):
    """output yaml file model info"""
    print('\n' + cfg_file_path)
    if 'world' in cfg_file_path:
        model=WorldModel(cfg=cfg_file_path)
    elif 'rtdetr' in cfg_file_path:  
        model=RTDETRDetectionModel(cfg=cfg_file_path)
    elif 'yoloe' in cfg_file_path and 'seg' in cfg_file_path:  
        model=YOLOESegModel(cfg=cfg_file_path)
    elif 'yoloe' in cfg_file_path:  
        model=YOLOEModel(cfg=cfg_file_path)
    else:
        model=DetectionModel(cfg=cfg_file_path)
    model.info()
    x=torch.ones(1,3,640,640)
    y=model(x)
    print(f"x.shape: {x.shape}")
    output_str=""
    if isinstance(y, dict):
        for k,v in y.items():
            if isinstance(v, list):
                output_str += k + ":" + "; ".join([f"list[{d_i}].shape: {d.shape}" for d_i,d in enumerate(v)]) + "\n"
    elif isinstance(y, list): # detect       
        output_str += "; ".join([f"list[{d_i}].shape: {d.shape}" for d_i,d in enumerate(y)])
    elif isinstance(y, torch.Tensor) or isinstance(y, np.ndarray):  # cls
        output_str += f"y.shape: {y.shape}\n"
    elif isinstance(y, tuple): # spp + seg
        tuple_str=""
        for d in y:
            if isinstance(d, (list, tuple)):
                tuple_str += "; ".join([f"list[{d_i}].shape: {d.shape}" for d_i,d in enumerate(d)]) + "\n"
            elif d is None:
                pass
            else:
                tuple_str += f"d.shape: {d.shape}" + "\n"
        output_str += tuple_str
    print(output_str)

# 单个配置文件参数量输出
cfg_file_path="/data/liujie/data/ultralytics-8.3.152/ultralytics/cfg/models/11/yolo11_gsconve.yaml"
test_yaml(cfg_file_path)

# 文件夹下配置文件参数量输出
# cfg_dir_path="/data/liujie/data/docker_data/ultralytics-8.3.152/ultralytics/cfg/models"
# import os
# for root, dirs, files in os.walk(cfg_dir_path):
#     for file in files:
#         if file.endswith(('yml', 'yaml')):
#             cfg_file_path = os.path.join(root, file)
#             test_yaml(cfg_file_path)
```
- 运行测试脚本`python ultralytics-8.3.152/tutorial/debug/test_yaml.py`,没有报错则为正确，增加block/head同理，若报错则需要调试`ultralytics-8.3.152/ultralytics/nn/tasks.py`中parse_model函数进行特殊配置
```

/data/liujie/data/ultralytics-8.3.152/ultralytics/cfg/models/11/yolo11_gsconve.yaml
WARNING ⚠️ no model scale passed. Assuming scale='n'.

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      
  3                  -1  1     28000  ultralytics.nn.modules.conv.GSConvE          [64, 64, 3, 2]                
  4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  5                  -1  1    111296  ultralytics.nn.modules.conv.GSConvE          [128, 128, 3, 2]              
  6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]           
  7                  -1  1    296320  ultralytics.nn.modules.conv.GSConvE          [128, 256, 3, 2]              
  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1     32096  ultralytics.nn.modules.block.C3k2            [256, 64, 1, False]           
 17                  -1  1     28000  ultralytics.nn.modules.conv.GSConvE          [64, 64, 3, 2]                
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1     86720  ultralytics.nn.modules.block.C3k2            [192, 128, 1, False]          
 20                  -1  1    111296  ultralytics.nn.modules.conv.GSConvE          [128, 128, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]           
 23        [16, 19, 22]  1    464912  ultralytics.nn.modules.head.Detect           [80, [64, 128, 256]]          
YOLO11_gsconve summary: 196 layers, 2,534,160 parameters, 2,534,144 gradients, 6.3 GFLOPs

YOLO11_gsconve summary: 196 layers, 2,534,160 parameters, 2,534,144 gradients, 6.3 GFLOPs
x.shape: torch.Size([1, 3, 640, 640])
list[0].shape: torch.Size([1, 144, 80, 80]); list[1].shape: torch.Size([1, 144, 40, 40]); list[2].shape: torch.Size([1, 144, 20, 20])
```
- 完成了上述步骤就可以愉快的参照官方文档进行训练/验证/预测/导出/基准测试/跟踪了
