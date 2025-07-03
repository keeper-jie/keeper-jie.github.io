# 刘捷--高级计算机视觉算法工程师简历

## 基本信息
- 姓名：刘捷
- 民族：汉
- 邮箱：liujie0068@foxmail.com
- Github：[https://github.com/keeper-jie](https://github.com/keeper-jie)
- 住址：浙江省杭州市
- 出生年月：1998.02
- 身高：175cm
- 政治面貌：共青团员
- 毕业院校：南华大学
- 学历：硕士
- 电子简历地址：[https://keeper-jie.github.io/](https://keeper-jie.github.io/)

## 核心优势
- 3年计算机视觉实战经验：专注目标检测/分割/分类等任务，覆盖图片、视频及端侧部署
- 开源贡献者：2次Ultralytics框架PR被官方合并([GitHub链接](https://github.com/ultralytics/ultralytics/issues?q=state%3Aclosed%20is%3Apr%20author%3Akeeper-jie))
- 全栈落地能力：从算法研发（PyTorch）→ 模型部署（RK3588）→ 项目管理（系统集成证书）
- 多模态技术探索者：实践LLM/VLM前沿技术，具备技术迁移能力

## 技术栈
### 算法框架
- 主框架：Ultralytics YOLO系列（YOLOv5/v6/v8/V10/V11/V12）、MMDetection、PaddleDetection
- 多模态：DeepSeek-R1-Distill-Qwen-1.5B、Qwen2-VL-2B等VLM/LLM端侧部署经验
- 部署：RKNN-Toolkit2、RKLLM、TensorRT、ONNX

### 算法方向
- 强项：目标检测（Detect）、目标分割（Segment）、分类（Classify）、关键点检测（Pose）、旋转框检测（OBB）
- 熟练：视频目标跟踪（BoT-SORT/ByteTrack）、多分类（PP-LCNet）

### 工程能力
- 边缘计算：RK3588模型量化（INT8）、性能优化（10%↑FPS/同等精度）
- 工具链：Docker、Linux Shell、Git

### 项目经历
1. 交通违法智能检测系统（浙江力嘉电子科技有限公司 2022-07 - 2023-11）

- 技术栈：目标分类（ResNet18） + 目标检测（YOLOv8）/旋转框检测（YOLOv8-OBB）、目标跟踪（Bot-SORT）

- 关键创新：

    - 使用角度敏感损失 + 数据增强（重采样/copy-paste）提高大货车强光灯检测效果，mAP@0.5提升12.3%

    - 增加轨迹持续时间 + 注意力机制（CBAM模块聚焦非机动车关键区域）+ 基于运动特征的阈值过滤（过滤异常减速），降低非机动车闯红灯误报率至<1.5%

- 大货车非法加装强光灯检测子系统：[大货车非法加装强光灯检测子系统](./项目结果展示.pdf)

    - 目标：实时获取海康平台Kafka流中大货车过车图片URL，检测此图片是否有非法加装的强光灯

    - 解决方案：使用Python获取Kafka流数据，实时发送给YOLOv8 Flask集群进行检测（将Pytorch model转为OpenVINO IR部署在`Intel Xeon Gold 5218R CPU@2.1GHz`上），将最后结果发送给MQTT服务器并存储MySQL

    - 成果：程序19：00 - 5：00开启，部署于诸暨市交警系统，每日图片吞吐量为10万+，精确率95%+，召回率85%+，平均识别时间0.3秒/每张图片（原先方案使用Pytorch model为0.6秒/每张图）

- 非机动车闯红灯检测跟踪项目：[非机动车闯红灯检测跟踪子系统](./项目结果展示.pdf)

    - 目标：根据视频流检测非机动车并跟踪，结合当前红灯情况判断是否闯红灯

    - 解决方案：OpenCV实时获取视频RTSP流，使用YOLOv8间隔帧检测并跟踪非机动车，根据配置的左转、直行、右转红灯区域使用自定义ResNet18进行检测，保存闯红灯照片（包含轨迹），发送JSON数据给MQTT服务器并存储MySQL

    - 成果：在3台服务器共14张NVIDIA T4上部署项目监控50路视频，红灯检测0.03秒/每帧，YOLOv8检测跟踪0.05秒/每帧，内存占用3G/每路，显存占用1.5G/每路，带宽4M/每路，准确率94%+

2. 高危行为识别平台（杭州杰峰软件有限公司 2023-12 - 至今）
   
- 任务：跌倒检测（图像/视频级） + 吸烟/玩手机检测（图像级）
  
- 技术栈：关键点检测（YOLOv8-Pose） + 图卷积（STGCN）、目标检测（YOLOv8）

- 关键创新：
  
    - 跌倒检测视频方案：YOLOv8-Pose + 时空特征模型（STGCN），关键点轨迹分析

    - 跌倒检测图片方案：YOLOv8检测躺着的人，与床/沙发/椅子没有交集则认为跌倒

    - 小目标优化：引入GSConv替换标准卷积，吸烟检测AP@0.5提升8.7%

- 部署：云端TensorRT FP16加速，端侧rk3588单张图片推理时间为28ms

- 成果：部署于云端面向C端客户，每月巡检精确率95.2%

3. Ultralytics开源贡献（2024 - 2025）
   
- PR#1：修复框/置信度/类别显示问题([Github链接](https://github.com/ultralytics/ultralytics/pull/17384))

- PR#2：修复yoloe-11-seg训练报错问题([Github链接](https://github.com/ultralytics/ultralytics/pull/21004))

- 影响：被采纳并入`ultralytics:main`官方版本

### 落地经验
- 端侧部署：
  
    - RK3588平台完成YOLOv5/v10/v8-p2/v8/RetinaFace等模型部署（零拷贝），CPU利用率降低40%
  
    - 实现INT8量化校准工具链，精度损失<0.5%

- 大模型探索：
  
    - 完成RK3588 Qwen-VL 2B图文问答POC，探索VLM在工业质检的应用（onnx -> rknn(量化) -> Gradio/Flask部署）

## 证书与教育
- 全日制专业学位硕士 | 计算机技术 | 南华大学（统招 2019-2022）
- 计算机科学与技术学士 | 计算机科学与技术 | 衡阳师范学院（统招 2015-2019）
- 证书：软件设计师（中级） | 系统集成项目管理工程师（中级）| CET-6

## 个人特点
- 优秀的问题解决能力、团队合作和沟通能力
- 持续学习和跟踪最新的计算机视觉技术研究成果
- 2025趋势是端侧大模型（大模型做小），语音大模型，具身智能
- 喜欢尝试新的解决方案，并详细文档记录过程和结果使其沉淀为问题解决方案

## 文章
[YOLOV8导出onnx操作及指标评测](./yolov8_onnx_benchmark.md)  

## 项目成果
[站坐躺完整跌倒视频](./站坐躺完整跌倒视频.gif)

[向后跌倒测试](./向后跌倒测试.gif)

[吸烟检测演示视频](./吸烟检测演示视频.gif)

[往前跌倒撑在地上](./往前跌倒撑在地上.gif)

[往前跌倒](./往前跌倒.gif)

[玩手机演示视频](./玩手机演示视频.gif)

[玩手机检测演示视频](./玩手机检测演示视频.gif)

[三个小孩玩手机检测](./三个小孩玩手机检测.gif)

[面向摄像头向前跌倒](./面向摄像头向前跌倒.gif)

[监控视角下大人玩手机检测](./监控视角下大人玩手机检测.gif)
