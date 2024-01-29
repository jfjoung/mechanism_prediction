import os

dir = 'unseen/result'
# file_names = ['aaa.txt', 'bbb.txt']

final_output = {}  # 결과를 저장할 딕셔너리

file_names = os.listdir(dir)

for file_name in file_names:
    file_path = f'{dir}/{file_name}'

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            final_output[file_name[:-4]] = file_content  # 파일 이름에서 '.txt'를 제거하여 키로 사용
    except FileNotFoundError:
        final_output[file_name[:-4]] = 'File not found'  # 파일을 찾을 수 없을 때의 처리 (선택 사항)

# 결과를 파일에 저장
with open('output.txt', 'w', encoding='utf-8') as output_file:
    for key, value in final_output.items():
        output_file.write(f'{key}: {value}\n')
