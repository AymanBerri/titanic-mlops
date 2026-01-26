from src.data_preprocessing import load_data, preprocess


def test_load_data():
    df = load_data("data/raw/train.csv")
    assert not df.empty
    assert "survived" in df.columns


def test_preprocess():
    df = load_data("data/raw/train.csv")
    df_processed = preprocess(df)
    # Check no missing values
    assert df_processed.isnull().sum().sum() == 0
    # Check columns are correctly encoded
    assert "sex_male" in df_processed.columns
