from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
from lib.utils.heapmap_utils import generate_heatmap
from lib.utils.ce_utils import generate_mask_cond, adjust_keep_rate,generate_bbox_mask
from lib.train.admin import multigpu
import torch.nn as nn
from lib.utils.misc import NestedTensor
from lib.models.layers.lora import collect_lora_residual_energies, collect_lora_router_weights


class SALTTrackActor(BaseActor):
    """Actor for training SALTTrack."""
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

        self.task_cls_loss_fn = nn.CrossEntropyLoss()
        # reg loss
        self.confidence_reg_loss = nn.MSELoss()
        self.lora_cfg = cfg.MODEL.get('LORA', None) if cfg else None
        self.use_semantic_guided_lora = bool(self.lora_cfg and self.lora_cfg.get('ENABLED', False) and
                                             self.lora_cfg.get('SEMANTIC_GUIDED', False))
        self.lora_semantic_guide_weight = self.lora_cfg.get('SEMANTIC_GUIDE_WEIGHT', 0.0) \
            if self.lora_cfg else 0.0
        self.lora_semantic_guide_type = self.lora_cfg.get('SEMANTIC_GUIDE_TYPE', 'suppress_unreliable') \
            if self.lora_cfg else 'suppress_unreliable'

        # LoRA 结构类型：
        #   standard: 原始单分支 LoRA，只使用 residual energy 正则。
        #   routed:   双专家 SRR-LoRA，额外用语义可靠性监督 semantic expert route。
        self.lora_type = self.lora_cfg.get('TYPE', 'standard') if self.lora_cfg else 'standard'

        # ROUTER_SUPERVISE_WEIGHT 只对 routed LoRA 生效。它控制 L_router 的强度：
        #   L_router = || r_s - semantic_gate ||^2
        # 其中 r_s 是 semantic expert 的平均路由概率，semantic_gate 来自
        # GT/pred/text 的语义可靠性估计。
        self.lora_router_supervise_weight = self.lora_cfg.get('ROUTER_SUPERVISE_WEIGHT', 0.0) \
            if self.lora_cfg else 0.0

        # 语义对齐相关配置
        self.use_semantic_align = cfg.MODEL.get('USE_SEMANTIC_ALIGN', False) if cfg else False
        if self.use_semantic_align:
            # WYP: 最小版语义监督权重
            self.semantic_weight_stage1 = cfg.TRAIN.get('SEMANTIC_WEIGHT_STAGE1', 0.2)
            self.semantic_weight_stage2 = cfg.TRAIN.get('SEMANTIC_WEIGHT_STAGE2', 0.05)
            self.semantic_visual_weight = cfg.TRAIN.get('SEMANTIC_VISUAL_WEIGHT', 1.0)
            self.semantic_text_weight = cfg.TRAIN.get('SEMANTIC_TEXT_WEIGHT', 1.0)
            self.semantic_gate_type = cfg.TRAIN.get('SEMANTIC_GATE_TYPE', 'clamp')
            self.semantic_gate_tau = cfg.TRAIN.get('SEMANTIC_GATE_TAU', 0.05)
            self.semantic_gate_floor = cfg.TRAIN.get('SEMANTIC_GATE_FLOOR', 0.0)
            # WYP: 文本对齐损失类型：'distance' (距离监督) 或 'direction' (方向监督) 或 'both' (两者结合)
            self.semantic_text_loss_type = cfg.TRAIN.get('SEMANTIC_TEXT_LOSS_TYPE', 'distance')
            # WYP: 当使用 'both' 时，两种损失的权重
            self.semantic_distance_weight = cfg.TRAIN.get('SEMANTIC_DISTANCE_WEIGHT', 0.5)
            self.semantic_direction_weight = cfg.TRAIN.get('SEMANTIC_DIRECTION_WEIGHT', 0.5)
            # WYP: 语义权重阶段切换不再复用 LoRA 配置，单独放在 TRAIN 里管理。
            self.stage1_ratio = cfg.TRAIN.get('SEMANTIC_STAGE1_RATIO', 0.4)
            self.total_epochs = cfg.TRAIN.EPOCH
    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.box_head.apply(self.fix_bn)

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def get_semantic_weight(self, epoch):
        """根据训练阶段返回不同的语义对齐权重"""
        if not self.use_semantic_align:
            return 0.0

        if self.semantic_weight_stage1 == self.semantic_weight_stage2:
            return self.semantic_weight_stage1

        if self.stage1_ratio <= 0:
            return self.semantic_weight_stage2

        stage1_epochs = int(self.total_epochs * self.stage1_ratio)
        if epoch < stage1_epochs:
            return self.semantic_weight_stage1  # Stage 1: 语义探索
        else:
            return self.semantic_weight_stage2  # Stage 2: 精细定位

    def _align_lora_energy_to_target(self, energy: torch.Tensor, target_size: int) -> torch.Tensor:
        """Align per-layer LoRA energy vectors to the semantic guide batch size.

        Some LoRA layers run on temporally expanded tensors (e.g. B * num_search),
        while others run on per-sample tensors (B). We reduce expanded energies back
        to the semantic guide batch size before aggregation.
        """
        if energy is None:
            return None
        if energy.dim() != 1:
            energy = energy.reshape(-1)

        current_size = energy.numel()
        if current_size == target_size:
            return energy

        if current_size % target_size == 0:
            return energy.reshape(-1, target_size).mean(dim=0)

        if target_size % current_size == 0:
            repeat_factor = target_size // current_size
            return energy.repeat_interleave(repeat_factor)

        # Fallback: keep training robust even if some layer uses an unexpected layout.
        return energy.mean().expand(target_size)

    def forward_pass(self, data):
        # assert len(data['template_images']) == 1
        template_list, search_list = [], []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 6, 128, 128)
            template_list.append(template_img_i)

        # search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 6, 320, 320)
        for i in range(self.settings.num_search):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])
            search_list.append(search_img_i)

        # soft token type infor
        bbox_mask_list = []
        for template_item in data["template_anno"]:
            template_bbox = template_item * template_list[0].shape[2]
            bbox_mask = torch.zeros((template_list[0].shape[0], template_list[0].shape[2], template_list[0].shape[3] )).to(template_list[0].device)
            bbox_mask = generate_bbox_mask(bbox_mask, template_bbox )

            bbox_mask = bbox_mask.unfold(1, 16, 16).unfold(2, 16, 16)
            bbox_mask = bbox_mask.mean(dim=(-1, -2)).view(bbox_mask.shape[0],-1).unsqueeze(-1)
            bbox_mask_list.append(bbox_mask)

        ## nlp + subject mask
        exp_str_subject_mask_infor = data["nlp"]
        exp_str_list = []
        subject_mask_list = []
        for item in exp_str_subject_mask_infor:
            item_list = item.split("+")
            exp_str_list.append(item_list[0])
            index_list = list(map(int, item_list[-1].split(",")))
            subject_mask_list.append(index_list)

        # WYP: 训练时不再把数据标签里的 subject mask 喂给文本前向。
        # 这样可以让训练和推理的文本侧前向保持一致：
        # 1. 都只依赖原始文本输入
        # 2. 都由模型自己预测 subject-related token importance
        # subject_mask_list 这里先保留解析，便于后续若要恢复 token-level 监督时继续使用。
        out_dict = self.net(template=template_list,
                            search=search_list,
                            soft_token_template_mask = bbox_mask_list,
                            exp_str=exp_str_list,
                            exp_subject_mask = None,
                            search_anno=data['search_anno']
                            )

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        # gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_bbox = gt_dict['search_anno'].view(-1, 4)
        gts = gt_bbox.unsqueeze(0)
        gt_gaussian_maps = generate_heatmap(gts, self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # (B,1,H,W)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)


        ## involve confidence_pred_score
        confidence_pred = pred_dict["confidence_pred"].squeeze(1)
        confidence_loss = self.confidence_reg_loss(confidence_pred.float(), iou.float())

        # WYP: 最小版语义监督
        # visual loss: Pred 和 GT 的实例特征应该接近
        # text loss:   Pred 和文本特征应该接近
        # gate:        如果 GT 比 Pred 更接近文本，则增强文本辅助损失，否则削弱
        semantic_visual_loss = torch.tensor(0.0, device=l1_loss.device)
        semantic_text_loss = torch.tensor(0.0, device=l1_loss.device)
        semantic_loss = torch.tensor(0.0, device=l1_loss.device)
        semantic_gate = torch.tensor(0.0, device=l1_loss.device)
        lora_semantic_reg = torch.tensor(0.0, device=l1_loss.device)
        lora_router_loss = torch.tensor(0.0, device=l1_loss.device)
        lora_semantic_route = torch.tensor(0.0, device=l1_loss.device)
        gt_text_similarity = torch.tensor(0.0, device=l1_loss.device)
        pred_text_similarity = torch.tensor(0.0, device=l1_loss.device)

        if self.use_semantic_align and 'pred_visual_features' in pred_dict and 'gt_visual_features' in pred_dict and 'target_text_features' in pred_dict:
            pred_feat = pred_dict['pred_visual_features']
            gt_feat = pred_dict['gt_visual_features']
            text_feat = pred_dict['target_text_features']

            if gt_feat is not None and text_feat is not None:
                # 视觉一致性损失：pred 和 gt 的实例特征应该接近
                semantic_visual_loss = 1 - torch.nn.functional.cosine_similarity(pred_feat, gt_feat, dim=-1).mean()

                # 计算相似度（用于日志监控）
                pred_text_sim = torch.nn.functional.cosine_similarity(pred_feat, text_feat, dim=-1)
                gt_text_sim = torch.nn.functional.cosine_similarity(gt_feat, text_feat, dim=-1)
                pred_text_similarity = pred_text_sim.mean()
                gt_text_similarity = gt_text_sim.mean()

                # WYP: 文本对齐损失 - 支持三种模式
                if self.semantic_text_loss_type == 'direction':
                    # 方向监督 - 防止特征坍塌
                    pred_to_text = torch.nn.functional.normalize(text_feat - pred_feat, dim=-1)
                    gt_to_text = torch.nn.functional.normalize(text_feat - gt_feat, dim=-1)
                    direction_consistency = torch.nn.functional.cosine_similarity(pred_to_text, gt_to_text, dim=-1)
                    semantic_text_loss = (1 - direction_consistency).mean()
                elif self.semantic_text_loss_type == 'both':
                    # 组合监督 - 距离 + 方向
                    # 距离损失：拉近 pred 和 text
                    distance_loss = (1 - pred_text_sim).mean()
                    # 方向损失：对齐 pred 和 gt 的改进方向
                    pred_to_text = torch.nn.functional.normalize(text_feat - pred_feat, dim=-1)
                    gt_to_text = torch.nn.functional.normalize(text_feat - gt_feat, dim=-1)
                    direction_consistency = torch.nn.functional.cosine_similarity(pred_to_text, gt_to_text, dim=-1)
                    direction_loss = (1 - direction_consistency).mean()
                    # 加权组合
                    semantic_text_loss = self.semantic_distance_weight * distance_loss + \
                                       self.semantic_direction_weight * direction_loss
                else:
                    # 距离监督（原始方法）- 直接拉近 pred 和 text
                    semantic_text_loss = (1 - pred_text_sim).mean()

                # 可靠性门控：如果 GT 比 pred 更接近文本，则增强监督
                delta_text_sim = (gt_text_sim - pred_text_sim).detach()
                if self.semantic_gate_type == 'none':
                    # 完全不使用gate
                    semantic_gate_per_sample = torch.ones_like(delta_text_sim)
                elif self.semantic_gate_type == 'sigmoid':
                    semantic_gate_per_sample = torch.sigmoid(delta_text_sim / self.semantic_gate_tau)
                    if self.semantic_gate_floor > 0:
                        semantic_gate_per_sample = self.semantic_gate_floor + \
                                                   (1.0 - self.semantic_gate_floor) * semantic_gate_per_sample
                else:
                    # clamp
                    semantic_gate_per_sample = torch.clamp(delta_text_sim, min=0.0)
                semantic_gate = semantic_gate_per_sample.mean()

                if self.use_semantic_guided_lora and self.lora_semantic_guide_weight > 0:
                    net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
                    lora_energies = collect_lora_residual_energies(net)
                    if len(lora_energies) > 0:
                        if self.lora_semantic_guide_type == 'enhance_reliable':
                            guide_weight = semantic_gate_per_sample
                        else:
                            guide_weight = 1.0 - semantic_gate_per_sample
                        target_size = guide_weight.numel()

                        # 不同 LoRA 层的输入形状不完全一致：
                        # 有的层按样本 B 前向，有的层按 B * num_search 或 token
                        # 维展开前向。这里统一压回 semantic_gate_per_sample 的
                        # 样本维度，才能逐样本加权 residual energy。
                        aligned_energies = [
                            self._align_lora_energy_to_target(energy, target_size)
                            for energy in lora_energies
                        ]
                        lora_energy = torch.stack(aligned_energies, dim=0).mean(dim=0)

                        # suppress_unreliable 模式下 guide_weight = 1 - gate：
                        # 语义不可靠时更强地压制 LoRA residual，避免低秩适配被
                        # 噪声文本/错误语义信号带偏。
                        lora_semantic_reg = (guide_weight * lora_energy).mean()
                    else:
                        lora_semantic_reg = torch.tensor(0.0, device=l1_loss.device)
                else:
                    lora_semantic_reg = torch.tensor(0.0, device=l1_loss.device)

                if self.lora_type == 'routed' and self.lora_router_supervise_weight > 0:
                    net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
                    router_weights = collect_lora_router_weights(net)
                    if len(router_weights) > 0:
                        target_size = semantic_gate_per_sample.numel()

                        # 收集所有 RoutedLoRALinear 的 semantic expert 概率 r_s。
                        # 每层可能有不同 token 数或 batch 展开方式，因此同样对齐到
                        # semantic_gate_per_sample 的样本维度后再跨层平均。
                        aligned_routes = [
                            self._align_lora_energy_to_target(route_weight, target_size)
                            for route_weight in router_weights
                        ]
                        semantic_route = torch.stack(aligned_routes, dim=0).mean(dim=0)
                        lora_semantic_route = semantic_route.mean()

                        # 用语义可靠性 gate 监督 router，而不是直接把文本喂进
                        # LoRA 参数生成器。这样 routed LoRA 在推理时只依赖输入
                        # feature 做路由，但训练阶段学到“何时更该走 semantic expert”。
                        lora_router_loss = torch.nn.functional.mse_loss(
                            semantic_route,
                            semantic_gate_per_sample.detach(),
                        )
                    else:
                        lora_router_loss = torch.tensor(0.0, device=l1_loss.device)

                current_epoch = gt_dict.get('epoch', 0)
                lambda_semantic = self.get_semantic_weight(current_epoch)
                semantic_loss = self.semantic_visual_weight * semantic_visual_loss + \
                                self.semantic_text_weight * semantic_gate * semantic_text_loss
            else:
                lambda_semantic = 0.0
        else:
            lambda_semantic = 0.0

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + \
               self.loss_weight['focal'] * location_loss + confidence_loss + \
               lambda_semantic * semantic_loss
        if self.use_semantic_guided_lora and self.lora_semantic_guide_weight > 0:
            loss = loss + lambda_semantic * self.lora_semantic_guide_weight * lora_semantic_reg

        # Routed LoRA 的结构监督项：把 semantic expert 的路由概率 r_s 拉向
        # semantic_gate。lambda_semantic 复用语义监督的阶段调度，避免训练早晚期
        # router 监督强度与主语义分支脱节。
        if self.lora_type == 'routed' and self.lora_router_supervise_weight > 0:
            loss = loss + lambda_semantic * self.lora_router_supervise_weight * lora_router_loss

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/confidence_loss": confidence_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU_main": mean_iou.item()
                      }

            # 添加语义对齐相关的日志
            if self.use_semantic_align:
                status["Loss/semantic"] = semantic_loss.item()
                status["Loss/semantic_visual"] = semantic_visual_loss.item()
                status["Loss/semantic_text"] = semantic_text_loss.item()
                status["semantic_weight"] = lambda_semantic
                status["semantic_gate"] = semantic_gate.item()
                status["gt_text_similarity"] = gt_text_similarity.item()
                status["pred_text_similarity"] = pred_text_similarity.item()
            if self.use_semantic_guided_lora:
                status["Loss/lora_semantic_reg"] = lora_semantic_reg.item()
            if self.lora_type == 'routed':
                status["Loss/lora_router"] = lora_router_loss.item()
                status["lora_semantic_route"] = lora_semantic_route.item()

            return loss, status
        else:
            return loss
