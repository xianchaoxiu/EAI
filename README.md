# Embodied Navigation

I currently focus on vision-and-language navigation including
- [Surveys](#Surveys)
- [Perception](#Perception)
- [Planning](#Planning)
- [Control](#Control)
- [Vision-and-Language Navigation（VLN）](#VLN)
<strong> Last Update: 2025/06/07 </strong>




<a name="Surveys" />

## Surveys
- [2025] A Survey of Robotic Navigation and Manipulation with Physics Simulators in the Era of Embodied AI, arXiv [[Paper](https://arxiv.org/abs/2505.01458)]
- [2025] Multimodal Perception for Goal-oriented Navigation: A Survey, arXiv [[Paper](https://arxiv.org/abs/2504.15643)]
- [2025] A Comprehensive Survey on Continual Learning in Generative Models, arXiv [[Paper](https://www.arxiv.org/abs/2506.13045)] [[Code](https://github.com/Ghy0501/Awesome-Continual-Learning-in-Generative-Models)]
- [2025] Toward Embodied AGI: A Review of Embodied AI and the Road Ahead, arXiv [[Paper](https://arxiv.org/pdf/2505.14235)]
- [2025] Vision-Language-Action Models: Concepts, Progress, Applications and Challenges, arXiv [[Paper](http://export.arxiv.org/abs/2505.04769)]
- [2025] Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models, arXiv [[Paper](https://arxiv.org/abs/2505.04921)] [[Code](https://github.com/HITsz-TMG/Awesome-Large-Multimodal-Reasoning-Models)]
- [2025] AI 大模型驱动的具身智能人形机器人技术与展望, 中国科学 [[Paper](https://www.sciengine.com/SSI/doi/10.1360/SSI-2024-0350)]
- [2025] Generative Models in Decision Making: A Survey, arXiv [[Paper](https://arxiv.org/abs/2502.17100)]
- [2025] A Survey of Efficient Reasoning for Large Reasoning Models: Language, Multimodality, and Beyond, arXiv [[Paper](https://arxiv.org/abs/2503.21614)]
- [2025] Are VLMs Ready for Autonomous Driving? An Empirical Study from the Reliability, Data, and Metric Perspectives, arXiv [[Paper](https://arxiv.org/abs/2501.04003)] [[Code](https://drive-bench.github.io/)]
- [2025] The Role of World Models in Shaping Autonomous Driving: A Comprehensive Survey, arXiv [[Paper](https://arxiv.org/abs/2502.10498)] [[Code](https://github.com/LMD0311/Awesome-World-Model)]
- [2025] A Survey on Mechanistic Interpretability for Multi-Modal Foundation Models, arXiv [[Paper](https://arxiv.org/abs/2502.17516)]
- [2025] Embodied Intelligence: A Synergy of Morphology, Action, Perception and Learning, ACM Computing Surveys  [[Paper](https://dl.acm.org/doi/abs/10.1145/3717059)] 
- [2025] Large Language Models for Multi-Robot Systems: A Survey, arXiv [[Paper](https://arxiv.org/abs/2502.03814)] [[Code](https://github.com/Zhourobotics/LLM-MRS-survey)]
- [2025] Survey on Large Language Model Enhanced Reinforcement Learning: Conceptaxonomy, and Methods, IEEE TNNLS  [[Paper](https://ieeexplore.ieee.org/abstract/document/10766898/)] 
- [2025] Humanoid Locomotion and Manipulation: Current Progress and Challenges in Control, Planning, and Learning, arXiv [[Paper](https://arxiv.org/pdf/2501.02116)] 
- [2025] UAVs Meet LLMs: Overviews and Perspectives Toward Agentic Low-Altitude Mobility, arXiv [[Paper](https://arxiv.org/abs/2501.02341)] [[Code](https://github.com/Hub-Tian/UAVs_Meet_LLMs)]
- [2025] A Survey of World Models for Autonomous Driving, arXiv [[Paper](https://arxiv.org/abs/2501.11260)] 
- [2025] Towards Large Reasoning Models: A Survey on Scaling LLM Reasoning Capabilities, arXiv [[Paper](https://arxiv.org/abs/2501.09686)] 
- [2025] A Survey on Large Language Models with some Insights on their Capabilities and Limitations, arXiv [[Paper](https://arxiv.org/abs/2501.04040)]
- [2025] 基于大模型的具身智能系统综述, 自动化学报 [[Paper](http://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c240542)]
- [2025] 具身智能研究的关键问题: 自主感知、行动与进化, 自动化学报 [[Paper](http://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c240364)] [[Code](https://github.com/BUCT-IUSRC/Survey__EmbodiedAI)]
- [2024] 大模型驱动的具身智能: 发展与挑战, 中国科学 [[Paper](https://doi.org/10.1360/SSI-2024-0076)]
- [2024] From Specific-MLLMs to Omni-MLLMs: A Survey on MLLMs Aligned with Multi-modalities, arXiv [[Paper](https://arxiv.org/abs/2412.11694)]  [[Code](https://github.com/threegold116/Awesome-Omni-MLLMs)]
- [2024] Vision-and-Language Navigation Today and Tomorrow: A Survey in the Era of Foundation Models, TMLR [[Paper](https://arxiv.org/abs/2407.07035)] [[Code](https://github.com/zhangyuejoslin/VLN-Survey-with-Foundation-Models)]
- [2024] Efficient Large Language Models: A Survey, TMLR [[Paper](https://arxiv.org/abs/2312.03863)] [[Code](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)]
- [2024] A Survey on Multimodal Large Language Models for Autonomous Driving, WACV [[Paper](https://openaccess.thecvf.com/content/WACV2024W/LLVM-AD/html/Cui_A_Survey_on_Multimodal_Large_Language_Models_for_Autonomous_Driving_WACVW_2024_paper.html)] 
- [2024] Personalization of Large Language Models: A Survey, arXiv [[Paper](https://arxiv.org/pdf/2411.00027)] 
- [2024] A Survey on LLM Inference-Time Self-Improvement, arXiv [[Paper](https://arxiv.org/abs/2412.14352)] 
- [2024] Embodied Navigation with Multi-modal Information: A Survey from Tasks to Methodology, Information Fusion [[Paper](https://www.sciencedirect.com/science/article/pii/S1566253524003105)] 
- [2024] Recent Advances in Robot Navigation via Large Language Models: A Review, arXiv [[Paper](https://www.researchgate.net/profile/Xian-Wei-3/publication/384537380)] 
- [2024] Large Language Models for Robotics: Opportunities, Challenges, and Perspectives, arXiv [[Paper](https://arxiv.org/abs/2401.04334)]
- [2024] Advances in Embodied Navigation Using Large Language Models: A Survey, arXiv [[Paper](https://arxiv.org/pdf/2311.00530)]  
- [2024] Foundation Models in Robotics: Applications, Challenges, and the Future, IJRR [[Paper](https://doi.org/10.1177/02783649241281508)] [[Code](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models)]
- [2024] A Survey of Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2303.18223)] [[Code](https://github.com/RUCAIBox/LLMSurvey)]
- [2024] ChatGPT for Robotics: Design Principles and Model Abilities, IEEE Access [[Paper](https://ieeexplore.ieee.org/abstract/document/10500490)] 
- [2023] Large Language Models for Robotics: A Survey, arXiv [[Paper](https://arxiv.org/abs/2311.07226)]
- [2023] LLM4Drive: A Survey of Large Language Models for Autonomous Driving, arXiv [[Paper](https://arxiv.org/abs/2311.01043)] [[Code](https://github.com/Thinklab-SJTU/Awesome-LLM4AD)]
- 



<a name="Perception" />

## Perception
- [2025] PanoGen++: Domain-Adapted Text-Guided Panoramic Environment Generation for Vision-and-Language Navigation, NN [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608025001996)]
- [2025] NavigateDiff: Visual Predictors are Zero-Shot Navigation Assistants, arXiv [[Paper](https://arxiv.org/pdf/2502.13894)] [[Code](https://21styouth.github.io/NavigateDiff/)]
- [2025] Mem2Ego: Empowering Vision-Language Models with Global-to-Ego Memory for Long-Horizon Embodied Navigation, arXiv [[Paper](https://arxiv.org/abs/2502.14254)]
- [2025] OpenDriveVLA: Towards End-to-end Autonomous Driving with Large Vision Language Action Model, arXiv [[Paper](https://arxiv.org/abs/2503.23463)]
- [2025] Visual-RFT: Visual Reinforcement Fine-Tuning, arXiv [[Paper](https://arxiv.org/abs/2503.01785)] [[Code](https://github.com/Liuziyu77/Visual-RFT)]
- [2025] ObjectVLA: End-to-End Open-World Object Manipulation Without Demonstration, arXiv [[Paper](https://arxiv.org/abs/2502.19250)] [[Code](https://objectvla.github.io/)]
- [2025] LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token, arXiv [[Paper](https://arxiv.org/abs/2501.03895)] [[Code](https://huggingface.co/ICTNLP/llava-mini-llama-3.1-8b)]
- [2025] Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling, arXiv [[Paper](https://arxiv.org/abs/2501.17811)] [[Code](https://github.com/deepseek-ai/Janus)]
- [2024] OVAL-Prompt: Open-Vocabulary Affordance Localization for Robot Manipulation through LLM Affordance-Grounding, arXiv [[Paper](https://arxiv.org/abs/2404.11000)] 
- [2024] NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models, AAAI [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28597)]
- [2024] LLaVA-MoD: Making LLaVA Tiny via MoE Knowledge Distillation, arXiv [[Paper](https://arxiv.org/abs/2408.15881)]  [[Code](https://github.com/shufangxun/LLaVA-MoD)]
- [2024] OpenGraph: Open-Vocabulary Hierarchical 3D Graph Representation in Large-Scale Outdoor Environments, arXiv [[Paper](https://arxiv.org/abs/2403.09412)]  
- [2023] Chat with the Environment: Interactive Multimodal Perception using Large Language Models, IROS [[Paper](https://ieeexplore.ieee.org/abstract/document/10342363)]
- [2023] VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models, arXiv [[Paper](https://arxiv.org/abs/2307.05973)]  
- [2023] Steve-Eye: Equipping LLM-Based Embodied Agents with Visual Perception in Open Worlds, ICLR [[Paper](https://arxiv.org/pdf/2310.13255)] 
- [2023] LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action, CORL [[Paper](https://doi.org/10.1177/02783649241281508)] 
- [2022] Flamingo: a Visual Language Model for Few-Shot Learning, NeurIPS [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html)] 
- [2021] Open-Vocabulary Object Detection via Vision and Language Knowledge Distillation, arXiv [[Paper](https://arxiv.org/pdf/2104.13921)]  



<a name="Planning" />

## Planning
- [2025] DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning, arXiv [[Paper](https://arxiv.org/abs/2506.06659)] 
- [2025] Ground-level Viewpoint Vision-and-Language Navigation in Continuous Environments, ICRA [[Paper](https://arxiv.org/abs/2502.19024)] 
- [2025] COSMO:Combination of Selective Memorization for Low-cost Vision-and-Language Navigation, arXiv [[Paper](https://arxiv.org/pdf/2503.24065)] 
- [2025] NavCoT: Boosting LLM-Based Vision-and-Language Navigation via Learning Disentangled Reasoning, IEEE TPAMI [[Paper](https://ieeexplore.ieee.org/abstract/document/10938647)] 
- [2025] FlexVLN: Flexible Adaptation for Diverse Vision-and-Language Navigation Tasks, arXiv [[Paper](https://arxiv.org/abs/2503.13966)] 
- [2025] MapNav: A Novel Memory Representation via Annotated Semantic Maps for VLM-based Vision-and-Language Navigation, arXiv [[Paper](https://arxiv.org/abs/2502.13451)] 
- [2025] NavRAG: Generating User Demand Instructions for Embodied Navigation through Retrieval-Augmented LLM, arXiv [[Paper](https://arxiv.org/abs/2502.11142)] 
- [2025] LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs, arXiv [[Paper](https://arxiv.org/abs/2501.06186)]  [[Video](https://github.com/mbzuai-oryx/LlamaV-o1)]
- [2025] Are VLMs Ready for Autonomous Driving? An Empirical Study from the Reliability, Data, and Metric Perspectives, arXiv [[Paper](https://arxiv.org/abs/2501.04003)]  [[Video](https://drive-bench.github.io/)]
- [2025] SD++: Enhancing Standard Defnition Mapsby Incorporating Road Knowledge using LLMs, arXiv [[Paper](https://arxiv.org/abs/2502.02773)]  
- [2025] FAST: Efficient Action Tokenization for Vision-Language-Action Models, arXiv [[Paper](https://arxiv.org/abs/2501.09747)]   [[Video](https://www.pi.website/research/fast)]
- [2025] AdaWM: Adaptive World Model based Planning for Autonomous Driving, arXiv [[Paper](https://arxiv.org/abs/2501.13072)] 
- [2025] Generative Planning with 3D-vision Language Pre-training for End-to-End Autonomous Driving, AAAI [[Paper](https://arxiv.org/abs/2501.08861)] 
- [2025] LLM-attacker: Enhancing Closed-loop Adversarial Scenario Generation for Autonomous Driving with Large Language Models, arXiv [[Paper](https://arxiv.org/pdf/2501.15850)]   [[Video](https://drive.google.com/file/d/1Zv4V3iG7825oyiKbUwS2Y-rR0DQIE1ZA/view)]
- [2024] Learning to Plan for Retrieval-Augmented Large Language Models from Knowledge Graphs, EMNLP [[Paper](https://arxiv.org/abs/2406.14282)] 
- [2024] Mastering Board Games by External and Internal Planning with Language Models, arXiv [[Paper](https://arxiv.org/abs/2412.12119)] 
- [2024] TopV-Nav: Unlocking the TopView Spatial Reasoning Potential of MLLM for Zero-shot Obiect Navigation, arXiv [[Paper](https://arxiv.org/abs/2411.16425)] 
- [2024] The One RING: a Robotic Indoor Navigation Generalist, arXiv [[Paper](https://arxiv.org/pdf/2412.14401)]  [[Video](https://one-ring-policy.allen.ai/)]
- [2024] World-Consistent Data Generation for Vision-and-Language Navigation, arXiv [[Paper](https://arxiv.org/abs/2412.06413)]
- [2024] Asynchronous Large Language Model Enhanced Planner for Autonomous Driving, ECCV [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-72764-1_2)]
- [2024] Large Language Model guided Deep Reinforcement Learning for Decision Making in Autonomous Driving, arXiv [[Paper](https://arxiv.org/pdf/2412.18511)]
- [2024] LLM-A*: Large Language Model Enhanced Incremental Heuristic Search on Path Planning, arXiv [[Paper](https://arxiv.org/abs/2407.02511)]  
- [2024] SayNav: Grounding Large Language Models for Dynamic Planning to Navigation in New Environments, ICAPS [[Paper](https://ojs.aaai.org/index.php/ICAPS/article/view/31506)] 
- [2024] AutoTAMP: Autoregressive Task and Motion Planning with LLMs as Translators and Checkers, ICRA [[Paper](https://ieeexplore.ieee.org/abstract/document/10611163)] 
- [2023] ProgPrompt: Generating Situated Robot Task Plans using Large Language Models, ICRA [[Paper](https://ieeexplore.ieee.org/abstract/document/10161317)] 
- [2023] Text2Motion: from Natural Language Instructions to Feasible Plans, Autonomous Robots [[Paper](https://link.springer.com/article/10.1007/s10514-023-10131-7)] 
- [2023] LLM as A Robotic Brain: Unifying Egocentric Memory and Control, arXiv [[Paper](https://arxiv.org/abs/2304.09349)]  
- [2023] PaLM-E: An Embodied Multimodal Language Model, arXiv [[Paper](https://arxiv.org/abs/2303.03378)]  
- [2022] Do As I Can, Not As I Say: Grounding Language in Robotic Affordances, arXiv [[Paper](https://arxiv.org/abs/2204.01691)]  
- [2022] Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents, ICML [[Paper](https://proceedings.mlr.press/v162/huang22a.html)]  
- [2021] Learning a Decision Module by Imitating Driver’s Control Behaviors, CORL [[Paper](https://proceedings.mlr.press/v155/huang21a.html)]  
- [2021] Neuro-Symbolic Program Search for Autonomous Driving Decision Module Design, CORL [[Paper](https://proceedings.mlr.press/v155/sun21a.html)]  
- [2021] A Lifelong Learning Approach to Mobile Robot Navigation, IEEE RAL [[Paper](https://ieeexplore.ieee.org/abstract/document/9345478)]  



<a name="Control" />

## Control
- [2025] TokenFLEX: Unified VLM Training for Flexible Visual Tokens Inference, arXiv [[Paper](https://arxiv.org/pdf/2504.03154)]  
- [2025] ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model, arXiv [[Paper](https://arxiv.org/abs/2502.14420)]  
- [2024] π0: A Vision-Language-Action Flow Model for General Robot Control, arXiv [[Paper](https://arxiv.org/abs/2410.24164)]  [[Video](https://www.physicalintelligence.company/blog/pi0)] 
- [2024] NaVILA: Legged Robot Vision-Language-Action Model for Navigation, arXiv [[Paper](https://arxiv.org/abs/2412.04453)]  [[Video](https://navila-bot.github.io/)]
- [2024] An LLM-based vision and language cobot navigation approach for Human-centric Smart Manufacturing, JMS [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0278612524000864)] 
- [2024] Hierarchical Auto-Organizing System for Open-Ended Multi-Agent Navigation, arXiv [[Paper](https://arxiv.org/abs/2403.08282)]  
- [2024] GOMA: Proactive Embodied Cooperative Communication via Goal-Oriented Mental Alignment, arXiv [[Paper](https://arxiv.org/abs/2403.11075)]  
- [2024] Probabilistically Correct Language-based Multi-Robot Planning using Conformal Prediction, arXiv [[Paper](https://arxiv.org/abs/2402.15368)]  
- [2024] Enhancing the LLM-Based Robot Manipulation Through Human-Robot Collaboration, arXiv [[Paper](https://arxiv.org/abs/2406.14097)]
- [2024] Scalable Multi-Robot Collaboration with Large Language Models: Centralized or Decentralized Systems?, ICRA [[Paper](https://ieeexplore.ieee.org/abstract/document/10610676)]  
- [2024] LLM-Powered Hierarchical Language Agent for Real-time Human-AI Coordination, AAMAS [[Paper](https://arxiv.org/abs/2312.15224)]  
- [2024] VELMA: Verbalization Embodiment of LLM Agents for Vision and Language Navigation in Street View, AAAI [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/29858)]  
- [2024] SRLM: Human-in-Loop Interactive Social Robot Navigation with Large Language Model and Deep Reinforcement Learning, arXiv [[Paper](https://arxiv.org/abs/2403.15648)]
- [2024] RoCo: Dialectic Multi-Robot Collaboration with Large Language Models, ICRA [[Paper](https://ieeexplore.ieee.org/abstract/document/10610855)]  
- [2024] Building Cooperative Embodied Agents Modularly with Large Language Models, ICLR [[Paper](https://arxiv.org/abs/2307.02485)]  
- [2024] Lifelong Robot Learning with Human Assisted Language Planners, ICRA [[Paper](https://ieeexplore.ieee.org/abstract/document/10610225)]  
- [2024] MoRAL: MoE Augmented LoRA for LLMs' Lifelong Learning, arXiv [[Paper](https://arxiv.org/abs/2402.11260)]
- [2024] LANCAR: Leveraging Language for Context-Aware Robot Locomotion in Unstructured Environments, IROS [[Paper](https://ieeexplore.ieee.org/abstract/document/10802075)]
- [2023] Instruct2Act: Mapping Multi-modality Instructions to Robotic Actions with Large Language Model, arXiv [[Paper](https://arxiv.org/abs/2305.11176)]  
- [2023] NaviSTAR: Socially Aware Robot Navigation with Hybrid Spatio-Temporal Graph Transformer and Preference Learning, IROS [[Paper](https://ieeexplore.ieee.org/abstract/document/10341395)]
- [2023] Asynchronous Multi-Agent Reinforcement Learning for Efficient Real-Time Multi-Robot Cooperative Exploration, arXiv [[Paper](https://arxiv.org/abs/2301.03398)]  
- [2023] Co-NavGPT: Multi-Robot Cooperative Visual Semantic Navigation using Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2310.07937)]  
- [2023] Controlling Large Language Model-based Agents for Large-Scale Decision-Making: An Actor-Critic Approach, arXiv [[Paper](https://arxiv.org/abs/2311.13884)]  
- [2023] LLM-Coordination: Evaluating and Analyzing Multi-agent Coordination Abilities in Large Language Models, arXiv [[Paper](https://arxiv.org/abs/2310.03903)]  
- [2023] ESC: Exploration with Soft Commonsense Constraints for Zero-shot Object Navigation, ICML [[Paper](https://proceedings.mlr.press/v202/zhou23r.html)]  
- [2023] Code as Policies: Language Model Programs for Embodied Control, ICRA [[Paper](https://ieeexplore.ieee.org/abstract/document/10160591)]  
- [2022] Multi-Agent Embodied Visual Semantic Navigation With Scene Prior Knowledge, IEEE RAL [[Paper](https://ieeexplore.ieee.org/abstract/document/9691871)]  
- [2022] Multi-Robot Active Mapping via Neural Bipartite Graph Matching, CVPR [[Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Ye_Multi-Robot_Active_Mapping_via_Neural_Bipartite_Graph_Matching_CVPR_2022_paper.html)]
- [2022] Learning Efficient Multi-agent Cooperative Visual Exploration, ECCV [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-19842-7_29)]






<a name="VLN" />

## VLN
- [2025] GENERAL SCENE ADAPTATION FOR VISION-AND-LANGUAGE NAVIGATION, arXiv [[Paper](https://arxiv.org/abs/2501.17403)] [[Code](https://github.com/honghd16/GSA-VLN)]
- [2025] UniGoal: Towards Universal Zero-shot Goal-oriented Navigation, arXiv [[Paper](https://arxiv.org/abs/2503.10630)] [[Code](https://github.com/bagh2178/UniGoal)]
- [2025] TRAVEL: Training-Free Retrieval and Alignment for Vision-and-Language Navigation, arXiv [[Paper](https://arxiv.org/pdf/2502.07306)]
- [2025] NavigateDiff: Visual Predictors are Zero-Shot Navigation Assistants, arXiv [[Paper](https://arxiv.org/pdf/2502.13894)] [[Code](https://21styouth.github.io/NavigateDiff/)]
- [2024] Zero-Shot Object Navigation with Vision-Language Models Reasoning, ICPR [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-78456-9_25)] [[Code](https://vlt-lzson.github.io/)] 
- [2024] Perceive, Reflect, and Plan: Designing LLM Agent for Goal-Directed City Navigation without Instructions, arXiv [[Paper](https://arxiv.org/pdf/2408.04168)] [[Code](https://anonymous.4open.science/r/PReP-13B5)] 
- [2024] Recent Advances in Robot Navigation via Large Language Models: A Review, researchgate [[Paper](https://www.researchgate.net/publication/384537380_Recent_Advances_in_Robot_Navigation_via_Large_Language_Models_A_Review)] 
- [2024] InstructNav: Zero-shot System for Generic Instruction Navigation in Unexplored Environment, arXiv [[Paper](https://arxiv.org/abs/2406.04882)] [[Code](https://github.com/LYX0501/InstructNav)] [[Project](https://sites.google.com/view/instructnav)]
- [2023] ESC: Exploration with Soft Commonsense Constraints for Zero-shot Object Navigation, arXiv [[Paper](https://arxiv.org/pdf/2301.13166)] [[Project](https://sites.google.com/ucsc.edu/escnav/home)]
- [2023] Find What You Want: Learning Demand-conditioned Object Attribute Space for Demand-driven Navigation, NeurIPS [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/34e278fbbd7d6d7d788c98065988e1a9-Paper-Conference.pdf)] [[Code](https://github.com/whcpumpkin/Demand-driven-navigation/tree/main)] [[Project](https://sites.google.com/view/demand-driven-navigation)]
- [2023] VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation , arXiv [[Paper](https://arxiv.org/abs/2312.03275)] [[Code](https://github.com/bdaiinstitute/vlfm)] [[Project](https://naoki.io/portfolio/vlfm)]
- [2022] CLIP on Wheels: Zero-Shot Object Navigation as Object Localization and Exploration, arXiv [[Paper](https://export.arxiv.org/abs/2203.10421v1)] 
- [2020] REVERIE: Remote Embodied Visual Referring Expression in Real Indoor Environments, CVPR [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Qi_REVERIE_Remote_Embodied_Visual_Referring_Expression_in_Real_Indoor_Environments_CVPR_2020_paper.html)] [[Code](https://github.com/YuankaiQi/REVERIE)]
- [2018] Vision-and-Language Navigation: Interpreting visually-groundednavigation instructions in real environments, CVPR [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.html)] [[Code](https://github.com/peteanderson80/Matterport3DSimulator)] [[Project](https://bringmeaspoon.org/)]
- [2011] A Recurrent Vision-and-Language BERT for Navigation, arXiv [[Paper](https://arxiv.org/abs/2011.13922)] [[Code](https://github.com/YicongHong/Recurrent-VLN-BERT)] 
- [2010] Room-Across-Room: Multilingual Vision-and-Language Navigation with Dense Spatiotemporal Grounding, arXiv [[Paper](https://arxiv.org/pdf/2010.07954)] [[Code](https://github.com/google-research-datasets/RxR)]





## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=xianchaoxiu/Embodied-Navigation&type=Date)](https://star-history.com/#xianchaoxiu/Embodied-Navigation)

