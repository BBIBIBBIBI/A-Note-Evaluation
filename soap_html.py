import diff_match_patch as dmp_module
from pathlib import Path
import re

# diff_match_patch 객체 초기화
dmp = dmp_module.diff_match_patch()

# 두 개의 노트 파일 경로 설정
base_path = '/Users/hjy/Desktop/output/'
note_code = '20240904143356'  # 이 부분을 필요에 따라 변경

note_path1 = f"{base_path}{note_code}-note1.txt"
note_path2 = f"{base_path}{note_code}-note2.txt"

note1_text = Path(note_path1).read_text()
note2_text = Path(note_path2).read_text()

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

# 두 노트에서 각 섹션(S, O, A, P)을 추출
note1_sections = extract_sections(note1_text)
note2_sections = extract_sections(note2_text)

# 두 섹션을 비교하고 HTML 형식으로 diff를 생성하는 함수
def generate_dmp_diff(text1, text2):
    # 개행 문자 제거
    text1 = text1.replace('\n', ' ').replace('\r', ' ')
    text2 = text2.replace('\n', ' ').replace('\r', ' ')
    
    diffs = dmp.diff_main(text1, text2)
    dmp.diff_cleanupSemantic(diffs)  # 읽기 쉽게 diff 정리
    html_diff = dmp.diff_prettyHtml(diffs)
    return html_diff

# 텍스트를 비교하고 추가된 비율, 삭제된 비율, 유사도를 계산하는 함수
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

# 특정 섹션에 대해 HTML 파일을 생성하고 비율을 계산하는 함수
def create_dmp_html_output_for_section(section, note_code):
    html_output = "<html><head><style>"
    html_output += "body {font-family: Arial, sans-serif;} h2 {color: #333;}"
    html_output += "</style></head><body>"
    
    # 선택된 섹션에 대한 비교 결과 가져오기
    text1 = note1_sections.get(section, '')
    text2 = note2_sections.get(section, '')
    
    # HTML diff 생성
    html_diff = generate_dmp_diff(text1, text2)
    html_output += f"<h2>{section}</h2>"
    html_output += html_diff  # 선택된 섹션의 HTML diff 추가

    # 통계 및 유사도 계산
    add_ratio, delete_ratio, overall_similarity = generate_stats_and_similarity(text1, text2)
    html_output += f"<p><strong>추가된 비율:</strong> {add_ratio:.2f}%</p>"
    html_output += f"<p><strong>삭제된 비율:</strong> {delete_ratio:.2f}%</p>"
    html_output += f"<p><strong>전체 유사도:</strong> {overall_similarity:.2f}%</p>"
    
    # 결과를 cmd에도 출력
    print(f"=== {section} 섹션 비교 결과 ===")
    print(f"추가된 비율: {add_ratio:.2f}%")
    print(f"삭제된 비율: {delete_ratio:.2f}%")
    print(f"전체 유사도: {overall_similarity:.2f}%")
    print("=========================\n")

    html_output += "</body></html>"
    return html_output

# 사용자로부터 비교할 섹션(S, O, A, P)을 입력받기
section = input("비교할 섹션을 입력하세요 (S, O, A, P): ").strip()

# 만약 입력에 괄호가 없으면 자동으로 추가
if section and not section.endswith(')'):
    section += ')'

# 선택된 섹션에 대한 HTML 출력 생성
dmp_html_output = create_dmp_html_output_for_section(section, note_code)

# 결과 HTML 파일 저장 (note_code와 섹션명을 포함한 파일명)
html_diff_path = f"/Users/hjy/Desktop/{note_code}-comparison-{section}.html"
with open(html_diff_path, 'w', encoding='utf-8') as f:
    f.write(dmp_html_output)

print(f"HTML diff saved to: {html_diff_path}")


