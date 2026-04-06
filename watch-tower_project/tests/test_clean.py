from src.data.clean import extract_day, extract_time, extract_class_type


def test_extract_day_returns_standardized_day_name():
    assert extract_day("ASC Monday 6pm") == "Monday"



def test_extract_time_extracts_ampm_time():
    assert extract_time("ACD Wed 7pm") == "7 pm"



def test_extract_class_type_identifies_asc():
    assert extract_class_type("ASC Sunday 9am") == "ASC"
