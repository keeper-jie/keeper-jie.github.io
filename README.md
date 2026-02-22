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
- 电子简历地址：[https://keeper-jie.github.io](https://keeper-jie.github.io)
## 核心优势

* **3年计算机视觉实战经验**：覆盖目标检测/分割/分类/跟踪/关键点检测/旋转框检测，应用于图片、视频文件、视频流及端侧部署
* **开源贡献者**：2个Ultralytics框架PR被官方采纳 ([链接](https://github.com/ultralytics/ultralytics/issues?q=state%3Aclosed%20is%3Apr%20author%3Akeeper-jie))
* **全栈落地能力**：数据管理（数据爬虫采集 → 清洗 → CVAT/X-AnyLabeling标注 → 质检 → 训练 → 部署 → 漏检误检迭代优化）→ 算法研发（PyTorch）→ 模型优化部署（RKNN/TensorRT/OpenVINO/Triton）→ 项目交付与管理（系统集成项目管理工程师证书）
* **前沿探索**：具备VLM/LLM/ASR/TTS等前沿模型实践与迁移能力

## 技术栈

* **算法框架**：YOLOv5/v8/v10/v11/v12/26、MMDetection、PaddleDetection、Ultralytics
* **大模型**：DeepSeek-R1、Qwen3-VL Unsloth/LLaMA-Factory微调及端侧部署
* **模型部署**：RKNN-Toolkit2、RKLLM、TensorRT、ONNX、OpenVINO
* **算法方向**：目标检测/分割/分类、关键点检测、旋转框检测、视频跟踪
* **工程工具**：Docker、Linux、Git，具备模型量化（INT8）、蒸馏、剪枝经验

## 项目经历

### 1. 交通违法智能检测系统｜浙江力嘉电子科技 (2022.07 - 2023.12)

* **技术栈**：YOLOv8、ResNet、OBB、BoT-SORT
* **成果**：
  * 大货车非法加装强光灯检测：**创新角度敏感损失 + 数据增强**，mAP\@0.5 提升 **5.3%**，夜间部署日处理图片 **10万+**，平均推理 **0.01s/图**
  * 非机动车闯红灯检测：**CBAM+轨迹过滤**，**误报率 <1.5%**，多服务器部署 **50路视频流**，精确率 **98.5%+**

### 2. 高危行为识别平台｜杭州杰峰软件 (2023.12 - 2025.08)

* **任务**：跌倒检测、吸烟/玩手机检测
* **技术栈**：YOLOv8-Pose、STGCN、GSConv优化
* **成果**：
  * 视频跌倒检测：**YOLOv8-Pose+STGCN**实现时空轨迹分析
  * 小目标优化：**引入GSConv替换标准卷积**，AP\@0.5 提升 **8.7%**
  * 部署：云端T4推理 **14ms/图**，月调用量 **50万+**，RK3588端侧 **28ms/图**

### 3. Ultralytics开源贡献 (2024 - 2025)

* 修复检测框显示 & YOLOE-Seg训练报错问题 → **合入官方main分支**

### 4. 浙江赛目科技 (2025.08 - 2025.11)

* 部署 **Qwen2.5-VL-72B**，探索多模态在无人机领域的零样本检测应用
* 配合仁和街道需求进行特定事件AI检测，完成17个算法预研（确定需求-》开发MVP-》安排标注任务按质按量提供算法API）

### 5. 杭州研趣信息技术有限公司 (2025.11 - 至今)

* 基于**8000+**SEM（电子显微镜）图片生成**320K**图文问答对资料微调Qwen3 VL 8B多模态大模型，完成数据生成-》训练-》部署-》评测工作，基于测试集相较于baseline提升了**16.4%**（Qwen3 VL embedding 2B计算回答的语义相似度）
* 基于半导体图片的**模板匹配方法调研**：尝试YOLOE，轮廓匹配，直方图匹配，SIFT，ORB，opencv模板匹配( 平方差和(SSD)，标准平方差(SQDIFF_NORMED)，相关(CORR)，标准相关(CORR_NORMED)，相关系数(CCOEFF)，标准相关系数(CCOEFF_NORMED)，直方图匹配（RGB颜色空间， HSV颜色空间，GRAY颜色空间） )，canny，resnet50特征提取+滑框，特征点匹配LightGlue，T-rex2，X-AnyLabeling模板匹配。在半导体测试图片上效果最好的为模板匹配cv2.TM_CCOEFF_NORMED
* 半导体量测项目图片标注-》核验-》训练-》部署，迭代优化算法效果，针对fin layer大目标修改模型架构适配
* UniEM、UniAIMS、YOLOE、SAM-EM、Qwen3-VL、Qwen3-VL-Embedding and Qwen3-VL-Reranker论文精读并云文档记录理解和反思

## 落地经验

* **端侧部署**：RK3588完成YOLO/RetinaFace部署，量化+零拷贝，CPU利用率下降 **40%**
* **大模型探索**：Qwen3-VL-8B图文问答POC，实践提示工程 + Unsloth/LLaMA-Factory进行多模态大模型问答任务微调
* **语音模型**：Wav2Vec2/Whisper/Zipformer (ASR)，MMS\_TTS/MeloTTS (TTS) 量化部署

## 教育与证书

* 硕士 | 计算机技术 | 南华大学 (2019-2022)
* 学士 | 计算机科学与技术 | 衡阳师范学院 (2015-2019)
* 软件设计师（中级）、系统集成项目管理工程师（中级）、CET-6

## 个人特点

* 善于快速定位和解决问题，具备团队协作与沟通能力
* 持续学习，跟踪前沿趋势（端侧大模型部署、语音大模型、具身智能、AI+硬件）
* 注重文档沉淀与知识复用

## 文章

[YOLOV8导出onnx操作及指标评测](https://github.com/keeper-jie/keeper-jie.github.io/blob/main/yolov8_onnx_benchmark.md)  

[ultralytics框架修改模型架构-新卷积](https://github.com/keeper-jie/keeper-jie.github.io/blob/main/ultralytics_add_conv.md) 

## 项目成果

[大货车非法加装强光灯检测 + 非机动车闯红灯检测](./项目结果展示.pdf)

[站坐躺完整跌倒视频](./站坐躺完整跌倒视频.gif)

[向后跌倒测试](./向后跌倒测试.gif)

[吸烟检测演示视频](./吸烟检测演示视频.gif)

[往前跌倒撑在地上](./往前跌倒撑在地上.gif)

[往前跌倒](./往前跌倒.gif)

[玩手机检测演示视频](./玩手机检测演示视频.gif)

[三个小孩玩手机检测](./三个小孩玩手机检测.gif)

[面向摄像头向前跌倒](./面向摄像头向前跌倒.gif)

[监控视角下大人玩手机检测](./监控视角下大人玩手机检测.gif)

## RK3588

[RK3588实时目标检测-yolov10](./rk3588_yolov10_实时监测.mp4)

[RK3588部署QwenVL-2.5 3B微调后效果](./qwenvl25_finetune.png)

[RK3588部署QwenLLM-2.5 3B微调后效果](./qwenllm25.png)

[RK3588转换Qwen3VL模型和官方转换模型效果比较](./RK3588转换模型效果和官方转换模型比较.pdf)

## 大模型

[多模态大模型无人机场景机动车人行道违停检测](./大模型检测人行道上违停.png)

[多模态大模型无人机场景屋顶违建检测](./多模态大模型检测屋顶违建.png)

## 无人机场景算法

[共享单车乱停放](./共享单车乱停放.png)

[堆放生活垃圾](./堆放生活垃圾.png)

[无人机图像目标检测](./无人机图像目标检测.png)

[烟雾检测](./无人机场景烟雾检测.png)

[暴露垃圾](./暴露垃圾.png)

[游动商贩](./游动商贩.png)

[火焰检测](./火焰检测.png)

[绿地脏乱](./绿地脏乱.png)

[航空图像检测](./航空图像检测.png)

[道路不洁](./道路不洁.png)

[通用目标检测](./ai无人机通用目标检测.gif)

[非机动车道违停](./非机动车道违停.png)

[人行道违停](./人行道违停.png)

[水面漂浮物](./水面漂浮物.png)

[消防通道违停](./消防通道违停检测.png)
