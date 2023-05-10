from os.path import abspath, dirname, split, join, basename
import glob
import json
import re
from typing import Dict, List
import unicodedata


ROOT_PATH = join(*split(abspath(dirname("__file__"))))
DATA_PATH = join(ROOT_PATH, "data")
RAW_DATA_PATH = f"{DATA_PATH}/raw_data"
PROCESSED_DATA_PATH = f"{DATA_PATH}/processed_data"

FILE_TO_CONTENT_COLUMN = {
    'agricultural_news': 'description',
    'bamid': 'Content',
    'cultural_news': 'Content',
    'customer_news': '內容',
    'customs_news': 'description',
    'edu_radio': 'description',
    'environment_news': 'newscontent',
    'exam_news': '內容',
    'exam_reports': '內容',
    'ey_news': '內容',
    'fda_news': '內容',
    'fish_news': '內文',
    'forest_news': 'NewsContext',
    'gov_news': "content",
    'international_news': "newscontent",
    'justice_news': '內文',
    'labor_news': 'description（描述）',
    'nmtl_news': 'Content',
    'part_southern_news': '內容',
    'pingtung_news': '公告內容',
    'president_news': 'Description',
    'southern_news': '內容',
    'taichung_news': 'content(內容)',
    'taipei_news': '內容',
    'taoyuan_news': 'detailcontent',
    'twmuseum_news': 'Content',
    'chiayicounty_news': '內容',
}
FILE_MAIN_PART = {
    "customs_news": "item",
    "edu_radio": "rows",
    "environment_news": "records",
    "international_news": "records",
}


def load_and_process_all_data(data_path_list):
    for i in range(len(data_path_list)):
        data_path = data_path_list[i]
        with open(data_path, "r", encoding='utf-8') as f:
            corpus = json.load(f)
        filename = data_path.split("/")[-1].split(".")[0]

        if isinstance(corpus, list):
            preprocessed_corpus = preprocess_raw_data(filename, corpus, FILE_TO_CONTENT_COLUMN[filename])
        elif isinstance(corpus, dict):
            corpus = corpus[FILE_MAIN_PART[filename]]
            preprocessed_corpus = preprocess_raw_data(filename, corpus, FILE_TO_CONTENT_COLUMN[filename])

        with open(f"{PROCESSED_DATA_PATH}/{filename}.json", 'w', encoding="utf8") as fp:
            json.dump(preprocessed_corpus, fp, indent=4, ensure_ascii=False)


def preprocess_raw_data(filename, corpus, content_col):
    uid_prefix = filename
    rows_list = []
    count = 0
    for i in range(len(corpus)):
        row = corpus[i]
        if not row[content_col]:
            continue
        else:
            row_dict = {}
            count += 1
            row_dict['uid'] = f"{uid_prefix}-{count}"
            text = remove_html_tags(row[content_col])
            text = unicodedata.normalize('NFKC', text)
            text = remove_phone_number(text)
            text = remove_urls(text)
            text = remove_date_and_time(text)
            clean_text = remove_redundant_char(text)
            if len(clean_text) < 20:
                continue
            else:
                row_dict['class'] = filename
                row_dict['text'] = clean_text.strip()
                rows_list.append(row_dict)
    return rows_list


def remove_html_tags(text):
    html_tags_pattern = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    clean_text = re.sub(html_tags_pattern, '', text)
    return clean_text


def remove_urls(text):
    url_pattern = re.compile(r'(http[s]?:\/\/|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(\w|[a-zA-Z0-9_-]|\.|\/|\?|\=|\&|\%|\@)*')
    email_pattern = re.compile(r'\S*@\S*\s?')
    clean_text = re.sub(url_pattern, '', text)
    clean_text = re.sub(email_pattern, '', clean_text)
    return clean_text


def remove_date_and_time(text):
    date_time_pattern = re.compile(r'(\d{1,4}[-./年月日:時])?(\d{1,2}[-./年月日:時分])?(\d{1,4}[年月日分秒\s])?(?:\([^)]*\))?')
    clean_text = re.sub(date_time_pattern, '', text)
    return clean_text


def remove_phone_number(text):
    phone_num_pattern = re.compile(r'(\+\d{2,4})?[-,\s]?\(?\d{2,4}\)?[-,\s]?\d{3,4}[-,\s]?\d{3,4}(#\d{1,4})?')
    clean_text = re.sub(phone_num_pattern, '', text)
    return clean_text

def remove_redundant_char(text):
    char_pattern = re.compile(r"(?<!\()\ufeff|\(\s*\)")
    clean_text = re.sub(char_pattern, '', text)
    
    return clean_text


def main():
    raw_data_path_list = [d for d in sorted(glob.glob(join(RAW_DATA_PATH, "*")))]

    load_and_process_all_data(raw_data_path_list)
    processed_data_path_list = [d for d in sorted(glob.glob(join(PROCESSED_DATA_PATH, "*")))]
    corpus_list = []
    for i in range(len(processed_data_path_list)):
        with open(processed_data_path_list[i], 'r', encoding='utf8') as f:
            corpus = json.load(f)
        corpus_list.extend(corpus)
    
    with open(f"{DATA_PATH}/corpus.json", 'w', encoding='utf8') as fp:
        json.dump(corpus_list, fp, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()