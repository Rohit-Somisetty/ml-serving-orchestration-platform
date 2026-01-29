from __future__ import annotations

from ml_platform.serving.predictor import ModelPredictor


def test_predictor_outputs_label(predictor: ModelPredictor) -> None:
    record = {
        "title": "Luxury sofa",
        "description": "Sectional couch with storage",
        "price": 1200,
        "brand": "FurniCo",
    }
    result = predictor.predict_one(record)
    assert "category" in result
    assert result["confidence"] > 0
