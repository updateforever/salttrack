from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
from lib.utils.heapmap_utils import generate_heatmap
from lib.utils.ce_utils import generate_mask_cond, adjust_keep_rate,generate_bbox_mask
from lib.train.admin import multigpu
import torch.nn as nn
import torch.distributed as dist
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
            self.semantic_contrast_weight = cfg.TRAIN.get('SEMANTIC_CONTRAST_WEIGHT', 0.0)
            self.semantic_contrast_tau = cfg.TRAIN.get('SEMANTIC_CONTRAST_TAU', 0.07)
            self.semantic_contrast_mode = cfg.TRAIN.get('SEMANTIC_CONTRAST_MODE', 'v2t')
            self.semantic_hard_neg_weight = cfg.TRAIN.get('SEMANTIC_HARD_NEG_WEIGHT', 0.0)
            self.semantic_hard_neg_topk = cfg.TRAIN.get('SEMANTIC_HARD_NEG_TOPK', 8)
            self.semantic_hard_neg_margin = cfg.TRAIN.get('SEMANTIC_HARD_NEG_MARGIN', 0.1)
            self.semantic_hard_neg_suppress_radius = cfg.TRAIN.get('SEMANTIC_HARD_NEG_SUPPRESS_RADIUS', 0.15)
            self.semantic_hard_neg_suppress_power = cfg.TRAIN.get('SEMANTIC_HARD_NEG_SUPPRESS_POWER', 2.0)
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

    def _build_temporal_sample_ids(self, total_size: int, local_batch_size: int, device,
                                   group_ids=None):
        """Return ids and whether they are globally comparable across GPUs.

        Training data is laid out as [search_step_0 batch, search_step_1 batch, ...].
        Frames from the same original sample share the same language description and
        must not be treated as negatives in the contrastive objective.

        If dataset-level group ids are provided, they identify the original
        dataset sequence. This additionally prevents samples drawn from the same
        sequence, possibly on different GPUs, from being used as negatives.
        """
        if group_ids is not None:
            group_ids = torch.as_tensor(group_ids, device=device, dtype=torch.long).reshape(-1)
            if group_ids.numel() == local_batch_size and total_size % local_batch_size == 0:
                return group_ids.repeat(total_size // local_batch_size), True

        if local_batch_size <= 0 or total_size % local_batch_size != 0:
            return torch.arange(total_size, device=device), False
        return torch.arange(local_batch_size, device=device).repeat(total_size // local_batch_size), False

    def _masked_contrastive_loss(self, visual_feat: torch.Tensor, text_feat: torch.Tensor,
                                 sample_ids: torch.Tensor,
                                 sample_ids_are_global: bool = False) -> torch.Tensor:
        """Global contrastive loss with configurable v2t or bidirectional mode.

        sample_ids mark entries that should not be negatives against each other.
        When ids come from dataset sequence ids, they are already globally unique
        and must stay unchanged after DDP all_gather. When ids are local fallback
        indices, we offset them by rank to avoid accidental cross-GPU matches.
        """
        if visual_feat.size(0) <= 1:
            return visual_feat.new_tensor(0.0)

        visual_feat = torch.nn.functional.normalize(visual_feat, dim=-1)
        text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
        local_visual_feat = visual_feat
        local_text_feat = text_feat
        local_sample_ids = sample_ids

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            visual_list = [torch.zeros_like(visual_feat) for _ in range(world_size)]
            text_list = [torch.zeros_like(text_feat) for _ in range(world_size)]
            sample_id_list = [torch.zeros_like(sample_ids) for _ in range(world_size)]
            dist.all_gather(visual_list, visual_feat.detach())
            dist.all_gather(text_list, text_feat.detach())
            dist.all_gather(sample_id_list, sample_ids)
            visual_list[rank] = visual_feat
            text_list[rank] = text_feat
            if sample_ids_are_global:
                local_sample_ids = sample_ids
            else:
                global_batch = sample_ids.max().detach() + 1
                sample_id_list = [
                    sid + global_batch * device_rank
                    for device_rank, sid in enumerate(sample_id_list)
                ]
                local_sample_ids = sample_ids + global_batch * rank
            visual_feat = torch.cat(visual_list, dim=0)
            text_feat = torch.cat(text_list, dim=0)
            sample_ids = torch.cat(sample_id_list, dim=0)

        logits_v2t = torch.matmul(local_visual_feat, text_feat.t()) / max(float(self.semantic_contrast_tau), 1e-6)
        pos_mask_v2t = local_sample_ids[:, None].eq(sample_ids[None, :]).float()
        pos_count_v2t = pos_mask_v2t.sum(dim=1).clamp(min=1.0)

        log_prob_v2t = logits_v2t - torch.logsumexp(logits_v2t, dim=1, keepdim=True)
        loss_v2t = -((pos_mask_v2t * log_prob_v2t).sum(dim=1) / pos_count_v2t).mean()
        if self.semantic_contrast_mode != 'bidirectional':
            return loss_v2t

        logits_t2v = torch.matmul(local_text_feat, visual_feat.t()) / max(float(self.semantic_contrast_tau), 1e-6)
        pos_mask_t2v = local_sample_ids[:, None].eq(sample_ids[None, :]).float()
        pos_count_t2v = pos_mask_t2v.sum(dim=1).clamp(min=1.0)
        log_prob_t2v = logits_t2v - torch.logsumexp(logits_t2v, dim=1, keepdim=True)
        loss_t2v = -((pos_mask_t2v * log_prob_t2v).sum(dim=1) / pos_count_t2v).mean()
        return 0.5 * (loss_v2t + loss_t2v)

    def _hard_visual_negative_loss(self, feature_map: torch.Tensor, score_map: torch.Tensor,
                                   size_map: torch.Tensor, offset_map: torch.Tensor,
                                   gt_bbox_xywh: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """Use high-confidence non-target proposals as text-conditioned hard visual negatives.

        score_map is the tracker's center confidence map. High responses away from
        the GT center are exactly the distractor locations that the tracker is most
        likely to confuse with the target. For each top-k response, we combine its
        center, size_map, and offset_map to form a predicted proposal box, then use
        the same RoIAlign feature extractor as the positive GT box. We enforce:

            sim(text, GT visual) > sim(text, hard negative visual) + margin

        Locations close to the GT center are softly suppressed, since nearby peaks
        can still describe the target reasonably well and should not be punished as
        strongly as far-away distractors.
        """
        if feature_map is None or score_map is None or size_map is None or offset_map is None \
                or gt_bbox_xywh is None or text_feat is None:
            ref_tensor = feature_map if feature_map is not None else score_map
            ref_tensor = ref_tensor if ref_tensor is not None else size_map
            ref_tensor = ref_tensor if ref_tensor is not None else offset_map
            ref_tensor = ref_tensor if ref_tensor is not None else gt_bbox_xywh
            ref_tensor = ref_tensor if ref_tensor is not None else text_feat
            return ref_tensor.new_tensor(0.0)
        if feature_map.dim() != 4 or score_map.dim() != 4 or size_map.dim() != 4 or offset_map.dim() != 4:
            return text_feat.new_tensor(0.0)

        n, c, h, w = feature_map.shape
        if n == 0 or score_map.shape[0] != n or size_map.shape[0] != n or offset_map.shape[0] != n:
            return feature_map.new_tensor(0.0)

        # 1. Pick top-k high-confidence center locations from the score map.
        #    We detach the score map for selection so the loss does not optimize by
        #    merely changing which locations are selected as hard negatives.
        score_flat = score_map[:, 0].reshape(n, -1)
        topk = min(int(self.semantic_hard_neg_topk), score_flat.size(1))
        if topk <= 0:
            return feature_map.new_tensor(0.0)

        _, topk_idx = torch.topk(score_flat.detach(), k=topk, dim=1)
        y = torch.div(topk_idx, w, rounding_mode='floor')
        x = topk_idx % w

        # 2. Convert each selected peak into a predicted proposal box using the
        #    same center-head decoding rule as the normal prediction path:
        #      cx, cy = (grid location + offset) / feature_size
        #      w, h   = size_map at that location
        size_flat = size_map.flatten(2).transpose(1, 2)
        offset_flat = offset_map.flatten(2).transpose(1, 2)
        box_gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, 2)
        neg_size = torch.gather(size_flat, dim=1, index=box_gather_idx)
        neg_offset = torch.gather(offset_flat, dim=1, index=box_gather_idx)
        neg_cx = (x.float() + neg_offset[..., 0]) / float(w)
        neg_cy = (y.float() + neg_offset[..., 1]) / float(h)
        neg_boxes = torch.stack([neg_cx, neg_cy, neg_size[..., 0], neg_size[..., 1]], dim=-1)
        neg_boxes = neg_boxes.reshape(n * topk, 4).clamp(min=0.0, max=1.0)

        # 3. Extract negative proposal RoI features. Positive and negative visual
        #    samples now use the same RoIAlign-based feature extractor.
        feature_map_expanded = feature_map[:, None].expand(n, topk, c, h, w).reshape(n * topk, c, h, w)
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        neg_feat = net.extract_instance_features(feature_map_expanded, neg_boxes).reshape(n, topk, c)

        # 4. Compute text-negative similarities for each hard proposal.
        text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
        neg_feat = torch.nn.functional.normalize(neg_feat, dim=-1)
        neg_sim = (neg_feat * text_feat[:, None, :]).sum(dim=-1)

        # 5. Down-weight negatives close to the GT center. Close responses may
        #    overlap the target or be equivalent localizations, so their penalty
        #    should be small. Far-away high responses keep full weight.
        gt_cx = gt_bbox_xywh[:, 0].clamp(0.0, 1.0)
        gt_cy = gt_bbox_xywh[:, 1].clamp(0.0, 1.0)
        grid_x = (x.float() + 0.5) / float(w)
        grid_y = (y.float() + 0.5) / float(h)
        dist = torch.sqrt((grid_x - gt_cx[:, None]) ** 2 + (grid_y - gt_cy[:, None]) ** 2)

        radius = max(float(self.semantic_hard_neg_suppress_radius), 1e-6)
        power = max(float(self.semantic_hard_neg_suppress_power), 0.0)
        neg_weight = torch.clamp(dist / radius, min=0.0, max=1.0) ** power

        # 6. Positive visual feature is the RoI feature of the GT box. The margin
        #    loss only fires when a hard negative is too similar to the text.
        pos_visual = net.extract_instance_features(feature_map, gt_bbox_xywh)
        pos_visual = torch.nn.functional.normalize(pos_visual, dim=-1)
        pos_sim = (pos_visual * text_feat).sum(dim=-1, keepdim=True)

        margin = float(self.semantic_hard_neg_margin)
        hard_loss = torch.nn.functional.relu(margin - pos_sim + neg_sim) * neg_weight
        return hard_loss.sum() / neg_weight.sum().clamp(min=1.0)

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
        semantic_distance_loss = torch.tensor(0.0, device=l1_loss.device)
        semantic_direction_loss = torch.tensor(0.0, device=l1_loss.device)
        semantic_direction_consistency = torch.tensor(0.0, device=l1_loss.device)
        semantic_contrast_loss = torch.tensor(0.0, device=l1_loss.device)
        semantic_hard_neg_loss = torch.tensor(0.0, device=l1_loss.device)
        delta_text_similarity = torch.tensor(0.0, device=l1_loss.device)

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

                pred_to_text = torch.nn.functional.normalize(text_feat - pred_feat, dim=-1)
                gt_to_text = torch.nn.functional.normalize(text_feat - gt_feat, dim=-1)
                direction_consistency = torch.nn.functional.cosine_similarity(pred_to_text, gt_to_text, dim=-1)
                semantic_direction_loss = (1 - direction_consistency).mean()
                semantic_direction_consistency = direction_consistency.mean()
                semantic_distance_loss = (1 - pred_text_sim).mean()

                if self.semantic_text_loss_type in ('contrast', 'direction_contrast', 'direction_contrast_hard'):
                    sample_ids, sample_ids_are_global = self._build_temporal_sample_ids(
                        pred_feat.size(0),
                        gt_dict['search_anno'].shape[1],
                        pred_feat.device,
                        gt_dict.get('contrast_group_id', None),
                    )
                    semantic_contrast_loss = self._masked_contrastive_loss(
                        pred_feat,
                        text_feat,
                        sample_ids,
                        sample_ids_are_global,
                    )
                    semantic_text_loss = self.semantic_contrast_weight * semantic_contrast_loss
                    if self.semantic_text_loss_type in ('direction_contrast', 'direction_contrast_hard'):
                        semantic_text_loss = semantic_text_loss + self.semantic_direction_weight * semantic_direction_loss
                    if self.semantic_text_loss_type == 'direction_contrast_hard' and self.semantic_hard_neg_weight > 0:
                        semantic_hard_neg_loss = self._hard_visual_negative_loss(
                            pred_dict.get('semantic_feature_map', None),
                            pred_dict.get('score_map', None),
                            pred_dict.get('size_map', None),
                            pred_dict.get('offset_map', None),
                            box_xywh_to_cxcywh(gt_bbox),
                            text_feat,
                        )
                        semantic_text_loss = semantic_text_loss + self.semantic_hard_neg_weight * semantic_hard_neg_loss
                elif self.semantic_text_loss_type == 'direction':
                    # 方向监督 - 防止特征坍塌
                    semantic_text_loss = semantic_direction_loss
                elif self.semantic_text_loss_type == 'both':
                    # 组合监督 - 距离 + 方向
                    semantic_text_loss = self.semantic_distance_weight * semantic_distance_loss + \
                                       self.semantic_direction_weight * semantic_direction_loss
                else:
                    # 距离监督（原始方法）- 直接拉近 pred 和 text
                    semantic_text_loss = semantic_distance_loss

                # 可靠性门控：如果 GT 比 pred 更接近文本，则增强监督
                delta_text_sim = (gt_text_sim - pred_text_sim).detach()
                delta_text_similarity = delta_text_sim.mean()
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
                status["semantic_distance_loss"] = semantic_distance_loss.item()
                status["semantic_direction_loss"] = semantic_direction_loss.item()
                status["semantic_direction_consistency"] = semantic_direction_consistency.item()
                status["semantic_contrast_loss"] = semantic_contrast_loss.item()
                status["semantic_hard_neg_loss"] = semantic_hard_neg_loss.item()
                status["delta_text_similarity"] = delta_text_similarity.item()
            if self.use_semantic_guided_lora:
                status["Loss/lora_semantic_reg"] = lora_semantic_reg.item()
            if self.lora_type == 'routed':
                status["Loss/lora_router"] = lora_router_loss.item()
                status["lora_semantic_route"] = lora_semantic_route.item()

            return loss, status
        else:
            return loss
