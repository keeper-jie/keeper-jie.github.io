# 刘捷 ｜ 视觉算法工程师

**目标检测 · 实例分割 · 端侧部署 · 多模态大模型** —— 4 年+ 计算机视觉算法工程落地经验

📧 【请填写求职邮箱】 ｜ 🔗 [github.com/keeper-jie](https://github.com/keeper-jie) ｜ 📍 浙江杭州

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![TensorRT](https://img.shields.io/badge/TensorRT-76B900?style=flat-square&logo=nvidia&logoColor=white) ![ONNX](https://img.shields.io/badge/ONNX-D50032?style=flat-square) ![RKNN](https://img.shields.io/badge/RKNN--Toolkit2-0B6DA8?style=flat-square) ![OpenVINO](https://img.shields.io/badge/OpenVINO-2B7BB9?style=flat-square) ![Qwen-VL](https://img.shields.io/badge/Qwen--VL-6F42C1?style=flat-square) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) ![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black)

> 这是我的个人技术展示站：**我能做什么 → 我做了什么（真实项目效果演示）**。完整简历见文末 [完整简历](#完整简历)。

## 目录

- [我能做什么](#我能做什么)
- [项目案例](#项目案例)
- [技术栈](#技术栈)
- [开源与技术沉淀](#开源与技术沉淀)
- [教育与证书](#教育与证书)
- [完整简历](#完整简历)

---

## 我能做什么

| 能力方向 | 具体内容与代表成果 |
| --- | --- |
| **检测 / 分割 / 全任务栈** | 目标检测、实例分割、语义分割、分类、跟踪（BoT-SORT）、关键点（YOLOv8-Pose）、旋转框、深度估计；CNN（YOLOv5-v12/26、ResNet）与 Transformer（Qwen-VL、ViT、STGCN）均有生产级落地 |
| **模型调优** | 自定义损失函数设计（角度敏感损失，mAP\@0.5 **+5.3%**）、小目标优化（GSConv，AP\@0.5 **+8.7%**）、长尾 / 类别不均衡（误报率 **<1.5%**）、过拟合诊断；数据增强、重采样、EMA、学习率调度 |
| **云端到端侧部署** | TensorRT、ONNX、RKNN-Toolkit2、OpenVINO、Triton、RKLLM；INT8 量化、蒸馏、剪枝、零拷贝；云端 T4 **14ms/图、月调用 50 万+**；RK3588 端侧 **28ms/图、CPU 利用率降 40%** |
| **多模态大模型** | Qwen3-VL、Qwen2.5-VL 微调（Unsloth / LLaMA-Factory）与端侧部署；**320K 图文对**数据引擎；本地大模型 Agent（请求成功率 **93.7%**、缓存命中率 **70.5%**） |
| **语音模型** | ASR（Wav2Vec2 / Whisper / Zipformer）、TTS（MMS / MeloTTS）微调与量化部署；TTS 成功率 **96.2%**、首包延迟 **1.13s** |
| **数据工程闭环** | 爬虫采集 → 清洗 → CVAT / X-AnyLabeling / **AI 智能标注** → 质检 → 训练 → 部署 → 漏检误检迭代闭环；半年交付 **502 个模型**、应用 **15189 次** |

---

## 项目案例

### 半导体 SEM 智能分析（在职项目）

从事**扫描电子显微镜（SEM）图像智能分析**：缺陷检测 / 量测算法、多模态领域大模型、智能电镜 AI 化。

- **SEM 领域大模型微调**：基于 **10000+ SEM 图像**构建数据引擎，自动生成 **320K 图文问答对**，微调 Qwen3-VL-8B，完成"数据生成 → 训练 → 部署 → 评测"闭环；语义相似度评测较 baseline 提升 **16.4%**
- **半导体量测训练平台**：图片上传 → **AI 智能标注** → 人工核验 → 训练 → 部署上线闭环，支持漏检误检样本回流迭代；半年累计训练交付 **502 个模型**，模型应用 **15189 次**
- **智能电镜 AI 化**：语音助手（TTS 成功率 **96.2%** / 首包 1.13s；ASR 成功率 >80%）；本地大模型 Agent（成功率 **93.7%**、TTFT 2874ms、缓存命中率 **70.5%**）
- **前沿跟踪**：精读 UniEM、UniAIMS、SAM-EM、YOLOE、Qwen3-VL 系列等 SEM 基础模型论文并沉淀技术文档

> *注：半导体项目图像涉密，本页面不展示效果图。*

### 端侧部署：RK3588

YOLO 系列检测 / RetinaFace / 多模态大模型在 RK3588 上的量化部署与效果验证，量化 + 零拷贝优化使 **CPU 利用率下降 40%**。

<table>
  <tr>
    <td><img src="./qwenvl25_finetune.png" width="420"></td>
    <td><img src="./qwenllm25.png" width="420"></td>
  </tr>
  <tr>
    <td><b>RK3588 部署 Qwen2.5-VL-3B 微调后效果</b></td>
    <td><b>RK3588 部署 Qwen2.5-LLM-3B 微调后效果</b></td>
  </tr>
</table>

https://github.com/user-attachments/assets/6661d068-a9ec-4818-a516-f86e604d58cf

*RK3588 实时目标检测（YOLOv10）演示*

- 📄 [RK3588 转换 Qwen3-VL 模型与官方转换效果对比](./RK3588转换模型效果和官方转换模型比较.pdf)

### 多模态大模型应用

基于 **Qwen2.5-VL-72B** 的无人机场景零样本检测探索——无需训练即可响应新检测需求。

<table>
  <tr>
    <td><img src="./大模型检测人行道上违停.png" width="420"></td>
    <td><img src="./多模态大模型检测屋顶违建.png" width="420"></td>
  </tr>
  <tr>
    <td><b>人行道违停检测</b></td>
    <td><b>屋顶违建检测</b></td>
  </tr>
</table>

### 无人机场景算法

面向城市治理的无人机巡查算法，覆盖 **15+ 场景**；曾单周期完成 **17 个算法预研与 MVP 交付**（需求确认 → MVP 开发 → 标注管理 → 算法 API）。

<table>
  <tr>
    <td><img src="./共享单车乱停放.png" width="270"></td>
    <td><img src="./非机动车道违停.png" width="270"></td>
    <td><img src="./人行道违停.png" width="270"></td>
  </tr>
  <tr>
    <td><sub>共享单车乱停放</sub></td>
    <td><sub>非机动车道违停</sub></td>
    <td><sub>人行道违停</sub></td>
  </tr>
  <tr>
    <td><img src="./消防通道违停检测.png" width="270"></td>
    <td><img src="./堆放生活垃圾.png" width="270"></td>
    <td><img src="./暴露垃圾.png" width="270"></td>
  </tr>
  <tr>
    <td><sub>消防通道违停</sub></td>
    <td><sub>堆放生活垃圾</sub></td>
    <td><sub>暴露垃圾</sub></td>
  </tr>
  <tr>
    <td><img src="./道路不洁.png" width="270"></td>
    <td><img src="./绿地脏乱.png" width="270"></td>
    <td><img src="./水面漂浮物.png" width="270"></td>
  </tr>
  <tr>
    <td><sub>道路不洁</sub></td>
    <td><sub>绿地脏乱</sub></td>
    <td><sub>水面漂浮物</sub></td>
  </tr>
  <tr>
    <td><img src="./无人机场景烟雾检测.png" width="270"></td>
    <td><img src="./火焰检测.png" width="270"></td>
    <td><img src="./游动商贩.png" width="270"></td>
  </tr>
  <tr>
    <td><sub>烟雾检测</sub></td>
    <td><sub>火焰检测</sub></td>
    <td><sub>游动商贩</sub></td>
  </tr>
  <tr>
    <td><img src="./无人机图像目标检测.png" width="270"></td>
    <td><img src="./航空图像检测.png" width="270"></td>
    <td>🎬 <a href="./ai无人机通用目标检测.gif">通用目标检测动图</a>（23MB）</td>
  </tr>
  <tr>
    <td><sub>无人机图像通用检测</sub></td>
    <td><sub>航空图像检测</sub></td>
    <td><sub>点击查看动图</sub></td>
  </tr>
</table>

### 高危行为识别：跌倒 / 吸烟 / 玩手机

**YOLOv8-Pose + STGCN** 时空轨迹建模的跌倒检测（覆盖站 / 坐 / 躺 / 撑地等形态）与行为检测；云端 T4 推理 **14ms/图**、月调用 **50 万+**。

<table>
  <tr>
    <td><img src="./站坐躺完整跌倒视频.gif" width="300"></td>
    <td><img src="./监控视角下大人玩手机检测.gif" width="300"></td>
    <td><img src="./吸烟检测演示视频.gif" width="300"></td>
  </tr>
  <tr>
    <td><b>跌倒检测（站 / 坐 / 躺全流程）</b></td>
    <td><b>玩手机检测（监控视角）</b></td>
    <td><b>吸烟检测</b></td>
  </tr>
</table>

<details>
<summary><b>更多行为识别演示（点击展开）</b></summary>
<br>

<table>
  <tr>
    <td><img src="./向后跌倒测试.gif" width="300"></td>
    <td><img src="./面向摄像头向前跌倒.gif" width="300"></td>
  </tr>
  <tr>
    <td><sub>向后跌倒</sub></td>
    <td><sub>面向摄像头向前跌倒</sub></td>
  </tr>
  <tr>
    <td><img src="./往前跌倒.gif" width="300"></td>
    <td><img src="./往前跌倒撑在地上.gif" width="300"></td>
  </tr>
  <tr>
    <td><sub>向前跌倒</sub></td>
    <td><sub>向前跌倒撑地</sub></td>
  </tr>
  <tr>
    <td><img src="./玩手机检测演示视频.gif" width="300"></td>
    <td><img src="./三个小孩玩手机检测.gif" width="300"></td>
  </tr>
  <tr>
    <td><sub>玩手机检测</sub></td>
    <td><sub>多人玩手机检测</sub></td>
  </tr>
</table>

</details>

### 交通违法检测系统

智能交通场景的违法行为检测，针对夜间低照度与难样本问题做专项优化。

- **大货车非法加装强光灯检测**：创新设计**角度敏感损失 + 数据增强**，mAP\@0.5 提升 **5.3%**；夜间部署日处理图片 **10 万+**，平均推理 **0.01s/图**
- **非机动车闯红灯检测**：**CBAM 注意力 + 轨迹过滤**抑制误报，误报率 **<1.5%**、精确率 **98.5%+**；多服务器部署 **50 路视频流**
- 📄 [完整检测结果展示（PDF）](./项目结果展示.pdf)

---

## 技术栈

| 分类 | 明细 |
| --- | --- |
| **深度学习框架** | PyTorch（精通）、Ultralytics YOLOv5-v12/26、MMDetection、PaddleDetection、HuggingFace Transformers |
| **任务方向** | 目标检测、实例分割、语义分割、分类、跟踪、关键点检测、旋转框检测、深度估计、缺陷检测 |
| **训练调优** | 损失函数设计（自定义角度敏感损失）、数据增强、重采样 / 长尾优化、EMA、学习率调度、过拟合诊断 |
| **部署加速** | TensorRT、ONNX、RKNN-Toolkit2、OpenVINO、Triton、RKLLM、INT8 量化、蒸馏、剪枝、零拷贝 |
| **大模型** | Qwen3-VL / Qwen2.5-VL 微调与端侧部署（Unsloth / LLaMA-Factory）、ASR（Wav2Vec2 / Whisper / Zipformer）、TTS（MMS / MeloTTS） |
| **语言与环境** | Python（熟练）、C++（读写推理部署代码）、Linux、Docker、Git |
| **AI 编程** | Claude Code / Codex 等 CLI Agent、LLM 对话式编程辅助、前沿论文快速复现 |
| **数据工程** | 爬虫采集、CVAT / X-AnyLabeling、AI 智能标注、质检与漏检误检迭代 |

## 开源与技术沉淀

- **Ultralytics 开源贡献**：修复检测框显示与 YOLOE-Seg 训练报错问题，**2 个 PR 合入官方 main 分支**（[PR 记录](https://github.com/ultralytics/ultralytics/issues?q=state%3Aclosed%20is%3Apr%20author%3Akeeper-jie)）
- 📝 [YOLOV8 导出 ONNX 操作及指标评测](https://github.com/keeper-jie/keeper-jie.github.io/blob/main/yolov8_onnx_benchmark.md) —— A6000 实测 PyTorch / ONNX / FP16 的 mAP 与推理速度对比，含完整 benchmark 命令
- 📝 [ultralytics 框架修改模型架构 - 新卷积](https://github.com/keeper-jie/keeper-jie.github.io/blob/main/ultralytics_add_conv.md) —— 以 GSConvE 为例，演示如何在 ultralytics 中新增自定义卷积模块

## 教育与证书

- **硕士** ｜ 计算机技术 ｜ 南华大学（2019 - 2022）｜ **学士** ｜ 计算机科学与技术 ｜ 衡阳师范学院（2015 - 2019）
- 软件设计师（中级）｜ 系统集成项目管理工程师（中级）｜ CET-6

## 完整简历

- 📃 [在线完整简历（resume.md）](./resume.md)
- 📥 [Word 版简历下载（resume.docx）](./resume.docx)
- 简历可按目标岗位定制输出 Word / PDF，欢迎通过页面顶部邮箱或 [GitHub](https://github.com/keeper-jie) 联系。

---

*© 2026 刘捷 · 由 GitHub Pages 渲染 · 最后更新：2026.11*
