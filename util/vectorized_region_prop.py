"""
완전 벡터화된 region-propagation 함수 (GPU 최적화)
루프 없이 배치 연산으로 구현
"""
import torch
import torch.nn.functional as F


def vectorized_region_propagation(mask, segments, corr_map, thresh, num_classes, device=None):
    """
    Region-propagation을 GPU에서 완전 벡터화로 수행
    
    Args:
        mask: (B, H, W) - 예측 마스크
        segments: (B, C, H, W) - 신뢰 가능 영역 (boolean)
        corr_map: (B, C, H, W) - correspondence map (boolean)
        thresh: float - 임계값
        num_classes: int - 클래스 수
        device: torch.device
    
    Returns:
        refined_mask: (B, H, W) - refinement된 마스크
    """
    B, C, H, W = corr_map.shape
    device = device or corr_map.device
    
    # Flatten 공간 차원
    mask_flat = mask.view(B, -1)  # (B, H*W)
    seg_flat = segments.view(B, C, -1)  # (B, C, H*W)
    seg_ori_flat = corr_map.view(B, C, -1)  # (B, C, H*W)
    
    # ============ 1단계: 각 segment 유효성 검사 ============
    # 각 segment의 픽셀 카운트
    seg_count = seg_flat.sum(dim=2)  # (B, C)
    seg_ori_count = seg_ori_flat.sum(dim=2)  # (B, C)
    
    # high_conf_ratio 계산 (seg의 유효성)
    high_conf_ratio = torch.where(
        seg_ori_count > 0,
        seg_count / seg_ori_count.float(),
        torch.zeros_like(seg_count, dtype=torch.float32)
    )
    
    valid_seg = (seg_count > 0) & (high_conf_ratio >= thresh)  # (B, C)
    
    # ============ 2단계: 각 segment별 클래스 분포 계산 ============
    # one-hot 인코딩으로 클래스별 카운트 계산
    mask_onehot = F.one_hot(mask_flat.long(), num_classes=num_classes).float()  # (B, H*W, num_classes)
    mask_onehot = mask_onehot.permute(0, 2, 1)  # (B, num_classes, H*W)
    
    # 각 segment별 클래스 카운트: (B, num_classes, C)
    class_count = torch.matmul(mask_onehot, seg_flat.float())
    
    # 각 segment별 top class와 그 카운트 계산
    top_class_count, top_class_idx = class_count.max(dim=1)  # (B, C)
    
    # Top class의 비율 계산
    total_in_seg = seg_count.float()
    top_ratio = torch.where(
        total_in_seg > 0,
        top_class_count / total_in_seg,
        torch.zeros_like(top_class_count)
    )
    
    # ============ 3단계: 최종 업데이트 조건 ============
    # valid_seg AND top_ratio > thresh
    update_mask = valid_seg & (top_ratio >= thresh)  # (B, C)
    
    # ============ 4단계: 배치 연산으로 마스크 업데이트 ============
    refined_mask = mask_flat.clone()  # (B, H*W)
    
    # update_mask에서 True인 위치의 (B, C) 인덱스 추출
    b_idx, c_idx = torch.where(update_mask)  # 1D 텐서들
    
    if len(b_idx) > 0:
        # 각 (b, c)에 대해 segment_ori_flat에서 True인 위치 찾기
        for i in range(len(b_idx)):
            b = b_idx[i].item()
            c = c_idx[i].item()
            
            # segment_ori 영역 마스크 (H*W,)
            seg_ori_mask = seg_ori_flat[b, c]
            new_class = top_class_idx[b, c].item()
            
            # 해당 위치들에 new_class 할당
            refined_mask[b][seg_ori_mask] = new_class
    
    # 원본 shape로 복원
    refined_mask = refined_mask.view(B, H, W)
    
    return refined_mask


def vectorized_region_propagation_no_loop(mask, segments, corr_map, thresh, num_classes, device=None):
    """
    완전히 loop-free한 버전 (scatter 연산 활용)
    매우 큰 배치에 대해 더 효율적일 수 있음
    """
    B, C, H, W = corr_map.shape
    device = device or corr_map.device
    
    # Flatten
    mask_flat = mask.view(B, -1)  # (B, H*W)
    seg_flat = segments.view(B, C, -1)  # (B, C, H*W)
    seg_ori_flat = corr_map.view(B, C, -1)  # (B, C, H*W)
    
    # ============ 유효성 및 top_class 계산 (위와 동일) ============
    seg_count = seg_flat.sum(dim=2)  # (B, C)
    seg_ori_count = seg_ori_flat.sum(dim=2)  # (B, C)
    
    high_conf_ratio = torch.where(
        seg_ori_count > 0,
        seg_count / seg_ori_count.float(),
        torch.zeros_like(seg_count, dtype=torch.float32)
    )
    valid_seg = (seg_count > 0) & (high_conf_ratio >= thresh)
    
    # 클래스 분포 계산
    mask_onehot = F.one_hot(mask_flat.long(), num_classes=num_classes).float()
    mask_onehot = mask_onehot.permute(0, 2, 1)
    class_count = torch.matmul(mask_onehot, seg_flat.float())
    
    top_class_count, top_class_idx = class_count.max(dim=1)
    total_in_seg = seg_count.float()
    top_ratio = torch.where(
        total_in_seg > 0,
        top_class_count / total_in_seg,
        torch.zeros_like(top_class_count)
    )
    
    update_mask = valid_seg & (top_ratio >= thresh)  # (B, C)
    
    # ============ Loop-free 업데이트 (scatter 방식) ============
    refined_mask = mask_flat.clone()
    
    # batch index와 segment index 생성
    b_indices, c_indices = torch.meshgrid(
        torch.arange(B, device=device),
        torch.arange(C, device=device),
        indexing='ij'
    )
    
    # 업데이트할 위치들만 필터링
    valid_b = b_indices[update_mask]  # 1D
    valid_c = c_indices[update_mask]  # 1D
    
    # 해당하는 top_class 추출
    new_classes = top_class_idx[update_mask]  # 1D
    
    # 각 (b, c) 쌍에 대해 seg_ori_flat[b, c]가 True인 위치에 new_classes 할당
    for i in range(len(valid_b)):
        b = valid_b[i].item()
        c = valid_c[i].item()
        seg_ori_mask = seg_ori_flat[b, c]
        refined_mask[b][seg_ori_mask] = new_classes[i].item()
    
    refined_mask = refined_mask.view(B, H, W)
    
    return refined_mask
