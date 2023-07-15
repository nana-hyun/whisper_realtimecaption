openai.api_key = " "

def get_chatgpt_response(content):
    prompt = f"""I will provide the text, so please summarize the contents in Korean in detail.
\n\n```{content}```"""
    
    print(f"Sending prompt to GPT-3:\n{prompt}\n")
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a deep learning model for learning assistance."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        n=1,
        temperature=0,
    )

    answer = response.choices[0].message['content'].strip()
    return answer


def split_text(text, limit=1700):
    if len(text) <= limit:
        return [text]

    chunks = []
    while text:
        chunk = text[:limit]
        last_space_index = chunk.rfind(' ')

        # 추가: last_space_index가 -1이면 텍스트를 그대로 분할합니다.
        if last_space_index == -1:
            last_space_index = limit

        chunk = chunk[:last_space_index]
        chunks.append(chunk)
        text = text[last_space_index:].strip()

    return chunks


def process_text(text):
    # 텍스트 분할
    content_chunks = split_text(text)

    answers = []
    for chunk in content_chunks:
        answer = get_chatgpt_response(chunk)
        answers.append(answer)
        print(f"Received response for chunk: {answer}")  # 응답을 출력합니다.

    # 결과를 하나의 문자열로 결합
    combined_answer = ' '.join(answers)
    return combined_answer
