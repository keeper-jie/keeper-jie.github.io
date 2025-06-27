# 使用ultralytics导出YOLOV8 onnx指标对比
##  实验环境：
```
硬件：
NVIDIA RTX A6000
Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz
软件
Ubuntu 22.04.3 LTS
NVIDIA-SMI 535.161.08
Python 3.10.15
ultralytics-8.3.13
onnx                          1.16.1
onnxruntime-gpu               1.18.1
onnxslim                      0.1.34
```
## 实验
命令
```
from ultralytics.utils.benchmarks import benchmark

benchmark(model="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_hand_smoke_phone.pt", data="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_v2_hand_smoke_phone/dataset.yaml", imgsz=800, half=False, int8=False, device=3)
print('benchmark(model="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_hand_smoke_phone.pt", data="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_v2_hand_smoke_phone/dataset.yaml", imgsz=800, half=False, int8=False, device=3)')
```
结果
```
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 36795/36795 [07:06<00:00, 86.30it/s]
all      36795      59253      0.965      0.941      0.981      0.824
Speed: 0.2ms preprocess, 9.4ms inference, 0.0ms loss, 0.7ms postprocess per image
Setup complete ✅ (128 CPUs, 755.3 GB RAM, 356.7/409.8 GB disk)

Benchmarks complete for od_hold_hand_smoke_phone.pt on /data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_v2_hand_smoke_phone/dataset.yaml at imgsz=800 (435.54s)
  Format Status❔  Size (MB)  metrics/mAP50-95(B)  Inference time (ms/im)     FPS
0   ONNX       ✅       39.3               0.8241                     9.4  106.34
```

命令
```
benchmark(model="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_hand_smoke_phone.pt", data="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_v2_hand_smoke_phone/dataset.yaml", imgsz=800, half=True, int8=False, device=1)
print('benchmark(model="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_hand_smoke_phone.pt", data="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_v2_hand_smoke_phone/dataset.yaml", imgsz=800, half=True, int8=False, device=1)')
```
结果
```
benchmark(model="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_hand_smoke_phone.pt", data="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_v2_hand_smoke_phone/dataset.yaml", imgsz=800, half=False, int8=False, device=3)

                Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 36795/36795 [05:37<00:00, 109.16it/s]
                   all      36795      59253      0.965       0.94      0.981      0.823
Speed: 0.2ms preprocess, 7.0ms inference, 0.0ms loss, 0.7ms postprocess per image
Setup complete ✅ (128 CPUs, 755.3 GB RAM, 356.7/409.8 GB disk)

Benchmarks complete for od_hold_hand_smoke_phone.pt on /data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_v2_hand_smoke_phone/dataset.yaml at imgsz=800 (346.69s)
  Format Status❔  Size (MB)  metrics/mAP50-95(B)  Inference time (ms/im)     FPS
0   ONNX       ✅       39.3               0.8231                    7.05  141.92
```

命令
```
benchmark(model="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_hand_smoke_phone.pt", data="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_v2_hand_smoke_phone/dataset.yaml", imgsz=800, half=False, int8=True, device=2)
print('benchmark(model="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_hand_smoke_phone.pt", data="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_v2_hand_smoke_phone/dataset.yaml", imgsz=800, half=False, int8=True, device=2)')
```
结果
```
benchmark(model="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_hand_smoke_phone.pt", data="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_v2_hand_smoke_phone/dataset.yaml", imgsz=800, half=True, int8=False, device=1)

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 36795/36795 [07:19<00:00, 83.73it/s]
                   all      36795      59253      0.965      0.941      0.981      0.824
Speed: 0.2ms preprocess, 9.7ms inference, 0.0ms loss, 0.7ms postprocess per image
Setup complete ✅ (128 CPUs, 755.3 GB RAM, 356.7/409.8 GB disk)

Benchmarks complete for od_hold_hand_smoke_phone.pt on /data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_v2_hand_smoke_phone/dataset.yaml at imgsz=800 (448.57s)
  Format Status❔  Size (MB)  metrics/mAP50-95(B)  Inference time (ms/im)     FPS
0   ONNX       ✅       39.3               0.8241                    9.71  102.96
```

## 结果汇总
| 模型 | metrics/mAP50-95(B) | FPS | 
| :---: | :---: | :---: |
| FP32(baseline) | 0.8241 | 106.34 |
| FP16 | 0.8231 | **141.92** |
| INT8 | 0.8241 | 102.96 |

# 结论
FP16稍微掉一点精度但是FPS更高，后续导出pytorch模型为onnx可以直接使用`benchmark(model="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_hand_smoke_phone.pt", data="/data2/liujie/data/docker_data/yolo_project/od_hold/od_hold_v2_hand_smoke_phone/dataset.yaml", imgsz=800, half=True, int8=False, device=1)`
