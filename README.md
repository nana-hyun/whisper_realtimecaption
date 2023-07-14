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

