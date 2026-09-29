from src.validate_data import validate_data

def test_dataset_validation():
    df = validate_data()
    assert len(df) > 0
    assert "price_lakh" in df.columns
