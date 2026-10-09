"""
Tests for the TuSimple label pipeline, encoder wiring, metrics and inference
engine. Everything runs on CPU with small synthetic inputs; no dataset or
pretrained-weight download is needed.
"""

import json

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from torchvision import models

from src.dataset import LaneDataset, rasterize_tusimple_lanes
from src.inference import LaneInferenceEngine
from src.metrics import LaneMetrics
from src.model import create_lanenet


# --- TuSimple rasterization -------------------------------------------------

H_SAMPLES = list(range(240, 720, 10))


def _straight_lane(x_top, x_bottom, start_row=0):
    """x positions for a straight lane, absent (-2) above start_row."""
    xs = np.linspace(x_top, x_bottom, len(H_SAMPLES))
    return [-2 if i < start_row else float(x) for i, x in enumerate(xs)]


def test_rasterize_assigns_instance_ids():
    lanes = [_straight_lane(600, 300), _straight_lane(680, 1000)]
    mask = rasterize_tusimple_lanes(lanes, H_SAMPLES, (720, 1280), thickness=5)
    assert mask.shape == (720, 1280)
    assert set(np.unique(mask)) == {0, 1, 2}
    # Lane 1 passes through x=300 at the bottom row, lane 2 through x=1000
    assert mask[710, 295:306].max() == 1
    assert mask[710, 995:1006].max() == 2
    # Nothing is drawn above the first h_sample
    assert mask[:235].sum() == 0


def test_rasterize_skips_missing_points():
    lane = _straight_lane(600, 300, start_row=20)  # first 20 samples absent
    mask = rasterize_tusimple_lanes([lane], H_SAMPLES, (720, 1280), thickness=3)
    first_row = H_SAMPLES[20]
    assert mask[: first_row - 3].sum() == 0
    assert mask[first_row:].sum() > 0


def test_rasterize_ignores_lanes_with_fewer_than_two_points():
    lane = [-2] * len(H_SAMPLES)
    lane[5] = 400
    mask = rasterize_tusimple_lanes([lane], H_SAMPLES, (720, 1280))
    assert mask.sum() == 0


@pytest.fixture
def tiny_tusimple(tmp_path):
    """Two fake 1280x720 frames in TuSimple layout with one label file."""
    records = []
    for i in range(2):
        rel = f"clips/0313-1/{i}/20.jpg"
        img_path = tmp_path / rel
        img_path.parent.mkdir(parents=True)
        cv2.imwrite(str(img_path), np.full((720, 1280, 3), 80, dtype=np.uint8))
        records.append({
            "lanes": [_straight_lane(600, 300), _straight_lane(680, 1000)],
            "h_samples": H_SAMPLES,
            "raw_file": rel,
        })
    with open(tmp_path / "label_data_0313.json", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return tmp_path


def test_dataset_reads_tusimple_labels(tiny_tusimple):
    ds = LaneDataset(str(tiny_tusimple), split='val', image_size=(288, 512), augment=False)
    assert len(ds) == 2
    sample = ds[0]
    assert sample['image'].shape == (3, 288, 512)
    assert sample['mask'].shape == (1, 288, 512)
    assert sample['instance'].shape == (288, 512)
    # Labels are real now: lane pixels exist and keep their instance ids
    assert sample['mask'].sum() > 0
    assert set(torch.unique(sample['instance']).tolist()) == {0, 1, 2}


def test_dataset_without_labels_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        LaneDataset(str(tmp_path), split='train')


def test_dataset_train_augmentation_keeps_mask_binary(tiny_tusimple):
    ds = LaneDataset(str(tiny_tusimple), split='train', image_size=(288, 512), augment=True)
    sample = ds[1]
    assert set(torch.unique(sample['mask']).tolist()) <= {0.0, 1.0}


# --- Encoder wiring / pretrained weights ------------------------------------

@pytest.mark.parametrize('backbone', ['efficientnet', 'mobilenet'])
def test_encoder_strides_and_channels(backbone):
    model = create_lanenet(backbone=backbone, pretrained=False)
    assert model.backbone_strides == [2, 4, 8, 16, 32]
    assert len(model.backbone_channels) == 5
    assert model.backbone_channels[-1] == 320  # head conv (1280 ch) is excluded


def test_init_weights_does_not_touch_backbone(monkeypatch):
    """Pretrained encoder weights must survive LaneNet's own initialisation."""
    real_ctor = models.efficientnet_b0

    def fake_pretrained(weights=None):
        net = real_ctor(weights=None)
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0.0123)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.5)
        return net

    monkeypatch.setattr(models, 'efficientnet_b0', fake_pretrained)
    model = create_lanenet(backbone='efficientnet', pretrained=True)

    for stage in model.encoder_stages():
        for m in stage.modules():
            if isinstance(m, nn.Conv2d):
                assert torch.all(m.weight == 0.0123)
            elif isinstance(m, nn.BatchNorm2d):
                assert torch.all(m.weight == 0.5)
    # ...while the decoder was freshly initialised
    assert not torch.all(model.decoder1.conv1.conv.weight == 0.0123)


# --- Metrics ----------------------------------------------------------------

def test_f1_and_precision_recall_are_correct():
    gt = np.zeros((10, 10))
    gt[:, 2:4] = 1          # 20 lane pixels
    pred = np.zeros((10, 10))
    pred[:, 3:5] = 0.9      # 20 predicted, 10 overlap
    precision, recall = LaneMetrics.precision_recall(pred, gt)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert LaneMetrics.f1_score(pred, gt) == pytest.approx(0.5)
    assert LaneMetrics.accuracy(pred, gt) == pytest.approx(0.8)


# --- Inference engine ---------------------------------------------------------

def test_inference_engine_round_trip(tmp_path):
    model = create_lanenet(backbone='mobilenet', pretrained=False)
    ckpt = tmp_path / 'ckpt.pth'
    torch.save({'model_state_dict': model.state_dict()}, ckpt)

    engine = LaneInferenceEngine(model_path=str(ckpt), device='cpu', backbone='mobilenet')
    image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    out = engine.infer_single(image)
    assert out['seg_logits'].shape == (1, 384, 640)
    assert out['embeddings'].shape == (4, 384, 640)
    assert out['latency_ms'] > 0
    assert isinstance(engine.detect_lanes(image), list)
