import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
from tabulate import tabulate
import diff_match_patch as dmp_module

# 개행 문자 제거 후 텍스트를 비교하고 추가된 비율, 삭제된 비율, 유사도를 계산하는 함수
def generate_stats_and_similarity(text1, text2):
    dmp = dmp_module.diff_match_patch()
    
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

    # 사람 수정 비율은 Add Ratio와 Delete Ratio의 합
    human_edit_ratio = add_ratio + delete_ratio

    return add_ratio, delete_ratio, overall_similarity, human_edit_ratio

# 디렉토리 내의 파일을 찾아 비교하는 함수 (유사도 100%는 선택적으로 제외)
def find_and_compare_files(directory, exclude_100_similarity=True):
    files = glob.glob(os.path.join(directory, '*-note1.txt'))
    results = []
    
    # 평균을 계산하기 위한 리스트들
    added_ratios = []
    deleted_ratios = []
    similarities = []
    human_edit_ratios = []
    
    filtered_similarities = []  # 유사도 100이 아닌 데이터를 위한 리스트
    filtered_timestamps = []  # 유사도 100이 아닌 데이터의 시간 정보를 위한 리스트
    all_similarities = []  # 유사도 100% 포함한 데이터를 위한 리스트
    all_timestamps = []  # 유사도 100% 포함한 데이터의 시간 정보를 위한 리스트
    count_100_similarity = 0
    total_pairs = 0

    # 파일 확인 로그
    print(f"Found {len(files)} note1 files.")

    for file1 in files:
        file2 = file1.replace('note1.txt', 'note2.txt')
        if os.path.exists(file2):  # note2 파일이 있을 때만 처리
            total_pairs += 1  # 쌍으로 묶였을 때만 total_pairs 증가
            with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
                text1 = f1.read()
                text2 = f2.read()
            stats = generate_stats_and_similarity(text1, text2)

            # 파일명에서 날짜 정보를 추출
            file_index = os.path.basename(file1).split('-')[0]
            date_obj = datetime.datetime.strptime(file_index, "%Y%m%d%H%M%S")
            
            # 결과 리스트에 추가 (모든 파일 포함)
            formatted_stats = [date_obj, file_index] + [round(stat, 2) for stat in stats]
            results.append(formatted_stats)

            # 각 Ratio 값들을 리스트에 저장
            added_ratios.append(stats[0])
            deleted_ratios.append(stats[1])
            similarities.append(stats[2])
            human_edit_ratios.append(stats[3])

            # 유사도 100%도 포함한 리스트에 추가
            all_similarities.append(stats[2])
            all_timestamps.append(date_obj)

            if stats[2] == 100.0:
                count_100_similarity += 1  # 100% 유사도 파일 수 증가
            else:
                filtered_similarities.append(stats[2])
                filtered_timestamps.append(date_obj)

    # 로그 출력: 실제로 results에 들어간 데이터 개수 확인
    print(f"Actual number of results: {len(results)}")

    # 결과를 날짜 순으로 정렬
    results = sorted(results, key=lambda x: x[0])  # 첫 번째 열인 date_obj 기준으로 정렬

    # pandas DataFrame으로 변환 (유사도 100% 포함/제외 선택 가능)
    if exclude_100_similarity:
        df_filtered = pd.DataFrame({
            'Timestamp': filtered_timestamps,
            'Similarity': filtered_similarities
        })
    else:
        df_filtered = pd.DataFrame({
            'Timestamp': all_timestamps,
            'Similarity': all_similarities
        })

    # pandas DataFrame으로 변환 (유사도 100% 포함한 데이터, 구간별 그래프용)
    df_all = pd.DataFrame({
        'Timestamp': all_timestamps,
        'Similarity': all_similarities
    })

    # 날짜 순서대로 정렬
    df_filtered = df_filtered.sort_values(by='Timestamp').reset_index(drop=True)
    df_all = df_all.sort_values(by='Timestamp').reset_index(drop=True)

    # 전체 파일에 대한 출력
    headers = ["Date", "File", "Added Ratio", "Deleted Ratio", "Similarity", "Human Edited Ratio"]
    print(tabulate(results, headers=headers, tablefmt="pretty"))
    print(f"Total number of file pairs: {total_pairs}")  # 쌍의 개수로 출력
    print(f"Number of files with 100% similarity: {count_100_similarity}")

    # 평균 계산
    avg_added_ratio = np.mean(added_ratios) if added_ratios else 0
    avg_deleted_ratio = np.mean(deleted_ratios) if deleted_ratios else 0
    avg_similarity = np.mean(similarities) if similarities else 0
    avg_human_edit_ratio = np.mean(human_edit_ratios) if human_edit_ratios else 0

    # 평균 출력
    print("\nAverage Values:")
    print(f"Avg Added Ratio: {avg_added_ratio:.2f}%")
    print(f"Avg Deleted Ratio: {avg_deleted_ratio:.2f}%")
    print(f"Avg Similarity: {avg_similarity:.2f}%")
    print(f"Avg Human Edited Ratio: {avg_human_edit_ratio:.2f}%")

    # 결과를 파일에 기록 (모든 파일 포함)
    output_path = "/Users/hjy/Desktop/results.txt"
    with open(output_path, "w") as file:
        file.write(f"Total number of file pairs: {total_pairs}\n")
        file.write(f"Number of files with 100% similarity: {count_100_similarity}\n")
        file.write(tabulate(results, headers=headers, tablefmt="pretty"))
        file.write("\nAverage Values:\n")
        file.write(f"Avg Added Ratio: {avg_added_ratio:.2f}%\n")
        file.write(f"Avg Deleted Ratio: {avg_deleted_ratio:.2f}%\n")
        file.write(f"Avg Similarity: {avg_similarity:.2f}%\n")
        file.write(f"Avg Human Edited Ratio: {avg_human_edit_ratio:.2f}%\n")

    return df_filtered, df_all

# 유사도 그래프 그리기 (유사도 100% 포함/제외 선택 가능)
def plot_similarities(df, title='Overall Similarity'):
    plt.figure(figsize=(20, 5))

    # 바 그래프 그리기
    plt.bar(df.index, df['Similarity'])

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Similarity (%)')
    plt.ylim(0, 100)
    plt.grid(True)

    # 일정한 간격으로 x축에 레이블 추가 (최대 10개의 레이블만 표시)
    tick_indices = df.index[::max(1, len(df)//10)]  # 최대 10개의 레이블로 제한
    tick_labels = df['Timestamp'].dt.strftime('%m-%d').iloc[::max(1, len(df)//10)]

    plt.xticks(tick_indices, tick_labels, rotation=45)

    plt.show()

# 유사도를 구간별로 나누어 개수를 세고 바 그래프 생성 (유사도 100% 포함)
def plot_similarity_distribution_by_date(df):
    bins = [0, 20, 40, 60, 80, 100]  # 유사도 구간 설정
    bin_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
    df['Similarity_Bin'] = pd.cut(df['Similarity'], bins=bins, labels=bin_labels, include_lowest=True)

    # 날짜별로 데이터를 그룹화하고, 각 날짜에 따라 다른 밝은 색상을 할당
    unique_dates = df['Timestamp'].dt.strftime('%Y-%m-%d').unique()
    colors = sns.color_palette('husl', len(unique_dates))

    # 그래프 그리기
    plt.figure(figsize=(20, 5))

    bar_width = 0.07  # 각 막대의 너비
    r = np.arange(len(bin_labels))  # x축 위치 (유사도 구간의 개수만큼 배열 생성)

    for i, date in enumerate(unique_dates):
        # 각 날짜에 해당하는 데이터만 필터링
        date_df = df[df['Timestamp'].dt.strftime('%Y-%m-%d') == date]
        
        # 각 날짜별로 유사도 구간에서 개수를 계산
        bin_counts = date_df['Similarity_Bin'].value_counts().sort_index()

        # 날짜별로 막대 그리기 (r + i * bar_width로 옆으로 나란히 배치)
        plt.bar(r + i * bar_width, bin_counts, width=bar_width, label=date, color=colors[i])

    plt.title('Similarity Distribution by Range and Date')
    plt.xlabel('Similarity Range')
    plt.ylabel('Number of Files')
    
    # x축 레이블을 가로로 출력
    plt.xticks(r + bar_width * (len(unique_dates) / 2), bin_labels, rotation=0)

    # 범례 추가
    plt.legend(title="Date", bbox_to_anchor=(1.05, 1), loc='upper left')

    # y축에만 그리드 적용
    plt.grid(axis='y')

    plt.tight_layout()  # 그래프 레이아웃을 자동으로 조정해 범례와 막대가 겹치지 않도록 함
    plt.show()

# 경로 설정
directory_path = '/Users/hjy/Desktop/output/P2'
exclude_100 = input("100% 유사도 데이터를 제외할까요? (y/n): ").strip().lower() == 'y'
df_filtered, df_all = find_and_compare_files(directory_path, exclude_100_similarity=exclude_100)

plot_similarities(df_filtered, title='Overall Similarity (Excluding 100%)' if exclude_100 else 'Overall Similarity (Including 100%)')
plot_similarity_distribution_by_date(df_all)
