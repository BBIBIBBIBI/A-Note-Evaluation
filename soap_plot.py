import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from tabulate import tabulate
import diff_match_patch as dmp_module
import re

# diff_match_patch 객체 초기화
dmp = dmp_module.diff_match_patch()

# 텍스트에서 S, O, A, P 섹션을 추출하는 함수
def extract_sections(text):
    sections = {}
    section_titles = ['S)', 'O)', 'A)', 'P)']
    pattern = r"(" + "|".join([re.escape(title) for title in section_titles]) + r")"
    split_text = re.split(pattern, text)

    for i in range(1, len(split_text), 2):
        section_title = split_text[i].strip()
        section_content = split_text[i + 1].strip()
        sections[section_title] = section_content
    return sections

# 개행 문자 제거 후 텍스트를 비교하고 추가된 비율, 삭제된 비율, 유사도를 계산하는 함수
def generate_stats_and_similarity(text1, text2):
    # 개행 문자 제거
    text1 = text1.replace('\n', ' ').replace('\r', ' ')
    text2 = text2.replace('\n', ' ').replace('\r', ' ')
    
    diffs = dmp.diff_main(text1, text2)
    dmp.diff_cleanupSemantic(diffs)

    add_count = sum(len(text) for op, text in diffs if op == dmp.DIFF_INSERT)
    delete_count = sum(len(text) for op, text in diffs if op == dmp.DIFF_DELETE)
    equal_count = sum(len(text) for op, text in diffs if op == dmp.DIFF_EQUAL)
    total_chars = add_count + delete_count + equal_count

    add_ratio = (add_count / total_chars) * 100 if total_chars else 0
    delete_ratio = (delete_count / total_chars) * 100 if total_chars else 0
    overall_similarity = (equal_count / total_chars) * 100 if total_chars else 0

    return add_ratio, delete_ratio, overall_similarity

# 특정 섹션(S, O, A, P)을 비교하는 함수
def compare_specific_section(file1, file2, section):
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        text1 = f1.read()
        text2 = f2.read()
    
    # S, O, A, P 섹션을 추출
    sections1 = extract_sections(text1)
    sections2 = extract_sections(text2)
    
    # 선택된 섹션을 비교
    text1_section = sections1.get(section, '')
    text2_section = sections2.get(section, '')
    add_ratio, delete_ratio, overall_similarity = generate_stats_and_similarity(text1_section, text2_section)
    
    return add_ratio, delete_ratio, overall_similarity

# 디렉토리 내의 파일을 찾아 지정된 섹션을 비교하는 함수
def find_and_compare_files(directory, section):
    files = glob.glob(os.path.join(directory, '*-note1.txt'))
    results = []
    filtered_similarities = []  # 유사도 100이 아닌 데이터를 위한 리스트
    filtered_timestamps = []  # 유사도 100이 아닌 데이터의 시간 정보를 위한 리스트
    count_100_similarity = 0
    total_pairs = 0

    # 파일 확인 로그
    print(f"Found {len(files)} note files.")

    for file1 in files:
        file2 = file1.replace('note1.txt', 'note2.txt')
        if os.path.exists(file2):  # note2 파일이 있을 때만 처리
            total_pairs += 1  # 쌍으로 묶였을 때만 total_pairs 증가
            add_ratio, delete_ratio, overall_similarity = compare_specific_section(file1, file2, section)

            # 파일명에서 날짜 정보를 추출
            file_index = os.path.basename(file1).split('-')[0]
            date_obj = datetime.datetime.strptime(file_index, "%Y%m%d%H%M%S")
            
            # 결과 리스트에 추가 (모든 파일 포함)
            formatted_stats = [date_obj, file_index, section, round(add_ratio, 2), round(delete_ratio, 2), round(overall_similarity, 2)]
            results.append(formatted_stats)

            if overall_similarity == 100.0:
                count_100_similarity += 1  # 100% 유사도 파일 수 증가

            # 유사도 100%인 파일을 제외하고 오전 9시부터 오후 6시 사이의 데이터만 그래프에서 필터링
            if overall_similarity != 100.0 and 9 <= date_obj.hour <= 18:
                filtered_similarities.append(overall_similarity)
                filtered_timestamps.append(date_obj)

    # 로그 출력: 실제로 results에 들어간 데이터 개수 확인
    print(f"Actual number of results: {len(results)}")
    print(f"Actual number of filtered results (for plotting): {len(filtered_similarities)}")

    # pandas DataFrame으로 변환 (유사도 100% 제외된 데이터, 그래프용)
    df = pd.DataFrame({
        'Timestamp': filtered_timestamps,
        'Similarity': filtered_similarities
    })

    # 날짜 순서대로 정렬
    df = df.sort_values(by='Timestamp').reset_index(drop=True)

    # 전체 파일에 대한 출력
    headers = ["Date", "File", "Section", "Added Ratio", "Deleted Ratio", "Similarity"]

    # 날짜 순서대로 results 정렬
    results = sorted(results, key=lambda x: x[0])

    print(tabulate(results, headers=headers, tablefmt="pretty"))
    print(f"Total number of file pairs: {total_pairs}")  # 쌍의 개수로 출력
    print(f"Number of files with 100% similarity: {count_100_similarity}")

    # 결과를 파일에 기록 (모든 파일 포함)
    output_path = "/Users/hjy/Desktop/section_o.txt"
    with open(output_path, "w") as file:
        file.write(f"Total number of file pairs: {total_pairs}\n")
        file.write(f"Number of files with 100% similarity: {count_100_similarity}\n")
        file.write(tabulate(results, headers=headers, tablefmt="pretty"))

    return df

# 유사도 그래프 그리기
def plot_similarities(df):
    plt.figure(figsize=(20, 5))

    # 바 그래프 그리기
    plt.bar(df.index, df['Similarity'])

    plt.title('Section Similarity')
    plt.xlabel('Date')
    plt.ylabel('Similarity (%)')
    plt.ylim(0, 100)
    plt.grid(True)

    # 일정한 간격으로 x축에 레이블 추가 (최대 10개의 레이블만 표시)
    tick_indices = df.index[::max(1, len(df)//10)]  # 최대 10개의 레이블로 제한
    tick_labels = df['Timestamp'].dt.strftime('%m-%d').iloc[::max(1, len(df)//10)]

    plt.xticks(tick_indices, tick_labels, rotation=45)

    plt.show()

# 경로 설정 및 비교할 섹션 지정
directory_path = '/Users/hjy/Desktop/output/P2'
section = input("비교할 섹션을 입력하세요 (S, O, A, P): ").strip()
if section and not section.endswith(')'):
    section += ')'

df = find_and_compare_files(directory_path, section)
plot_similarities(df)
