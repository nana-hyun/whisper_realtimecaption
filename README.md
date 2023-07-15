# whisper_real time caption with chat_GPT
2023 데이터분석경진대회 

---
* install package

(pip or pip3)

pip install torch

pip install torchaudio

pip install openai-whisper

pip install numba

pip install halo

pip install easydict

pip install soundcard

pip install webrtcvad

pip install threading

pip install queue

pip install collections

...

이외 필요한 패키지 install 해주시고 오류가 나면 uninstall 후 다시 install 해주세요.

cmd창에서 실행 
>
> python test1.py

![image](https://github.com/nana-hyun/whisper_realtimecaption/assets/101063108/8f8571d5-a453-4f03-b185-494c6e689e0c)



## about test1
![image](https://github.com/nana-hyun/whisper_realtimecaption/assets/101063108/d0801e50-dc2e-4acc-b261-71d4fc7e3caa)

![image](https://github.com/nana-hyun/whisper_realtimecaption/assets/101063108/6a5d79b2-96ae-4b56-a02d-59d677463a80)

test1은 아래 페이지를 참고하여 제작하였습니다.

https://digging-on-bytes.com/whisper%EC%99%80-python%EC%9C%BC%EB%A1%9C-%EC%8B%A4%EC%8B%9C%EA%B0%84-%EB%B2%88%EC%97%AD%EA%B8%B0-%EB%A7%8C%EB%93%A4%EA%B8%B0-part1/



추가할 함수는 다음과 같습니다.

1. text를 저장
2. 저장한 text를 이용해 chat_gpt api를 활용, 더 나은 품질의 자막 제공
3. 전체 저장한 text를 이용해 chat_gpt를 이용해 요약 정리


*query.py

chat api를 활용하여 chat gpt한테 요약정리를 시키는 코드입니다.
get_chatgpt_response는 chat api한테 prompt를 넘겨주는 함수입니다.
split_text는 chat api 입력 토큰 길이 제한 때문에, 특정 길이로 잘라서 분할해주는 함수입니다.
process_text는 텍스트를 분할하여 get_chatgpt_response를 적용하는 함수입니다.
사용예시는 다음과 같습니다.
```
input_text = "텍스트 입력 예시"
processed_text = process_text(input_text)
```

텍스트가 아니라 데이터프레임 값을 입력하고 싶다면 다음 코드를 사용하세요
```
from tqdm.auto import tqdm

total_rows = df.shape[0]
for index in tqdm(range(total_rows), desc="Processing rows"):
    content = df.loc[index, 'content']

    # 텍스트 분할
    content_chunks = split_text(content)

    answers = []
    for chunk in content_chunks:
        answer = get_chatgpt_response(chunk)
        answers.append(answer)
        print(f"Received response for chunk: {answer}")  # 응답을 출력합니다.

    # 결과를 하나의 문자열로 결합
    combined_answer = ' '.join(answers)
    df.at[index, 'answer'] = combined_answer
    print(f"Processed row {index + 1}/{total_rows}: {combined_answer}\n")
```
