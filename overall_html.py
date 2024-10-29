import diff_match_patch as dmp_module

def generate_html_diff_and_stats(text1, text2):
    dmp = dmp_module.diff_match_patch()
    diffs = dmp.diff_main(text1, text2)
    dmp.diff_cleanupSemantic(diffs)  # 의미론적으로 diff 정리

    # 각 유형의 변경 사항 카운트
    add_count = sum(len(text) for op, text in diffs if op == dmp.DIFF_INSERT)
    delete_count = sum(len(text) for op, text in diffs if op == dmp.DIFF_DELETE)
    equal_count = sum(len(text) for op, text in diffs if op == dmp.DIFF_EQUAL)
    total_chars = add_count + delete_count + equal_count

    # 비율 계산
    add_ratio = add_count / total_chars if total_chars else 0
    delete_ratio = delete_count / total_chars if total_chars else 0
    modify_ratio = equal_count / total_chars if total_chars else 0

    # 전체 유사도 계산
    overall_similarity = (equal_count / total_chars) * 100 if total_chars else 0

    # HTML 시각화 생성
    html_diff = dmp.diff_prettyHtml(diffs)

    return html_diff, add_ratio, delete_ratio, modify_ratio, overall_similarity

base_path = '/Users/hjy/Desktop/output/P1/'
note_code = '20240924142746'  # 이 부분을 필요에 따라 변경

note_path1 = f"{base_path}{note_code}-note1.txt"
note_path2 = f"{base_path}{note_code}-note2.txt"

with open(note_path1, 'r', encoding='utf-8') as file1, \
     open(note_path2, 'r', encoding='utf-8') as file2:
    text1 = file1.read()
    text2 = file2.read()

html_diff, add_ratio, delete_ratio, modify_ratio, overall_similarity = generate_html_diff_and_stats(text1, text2)

print(f"Added content ratio: {add_ratio:.2%}")
print(f"Deleted content ratio: {delete_ratio:.2%}")
print(f"Modified content ratio: {modify_ratio:.2%}")
print(f"Overall similarity: {overall_similarity:.2f}%")

# HTML 파일로 저장
html_diff_path = f"/Users/hjy/Desktop/{note_code}-comparison.html"
with open(html_diff_path, 'w', encoding='utf-8') as f:
    f.write(html_diff)

print(f"HTML diff saved to: {html_diff_path}")
