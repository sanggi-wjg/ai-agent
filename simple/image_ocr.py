import base64
import sys
from pathlib import Path

from langchain_core.globals import set_debug, set_verbose
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

set_debug(True)
set_verbose(True)


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ocr_image(image_path: str) -> str:
    llm = ChatOllama(model="qwen3-vl:235b-cloud", temperature=0)

    image_data = encode_image_to_base64(image_path)
    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime_type = "jpeg" if suffix in ("jpg", "jpeg") else suffix

    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{mime_type};base64,{image_data}"},
            },
            {
                "type": "text",
                "text": (
                    "이 이미지에서 텍스트를 모두 추출해주세요.\n"
                    "- 원본 텍스트의 구조와 줄바꿈을 최대한 유지하세요.\n"
                    "- 텍스트 외 설명은 포함하지 마세요.\n"
                    "- 텍스트가 없으면 '텍스트 없음' 이라고 출력하세요."
                ),
            },
        ]
    )

    response = llm.invoke([message])
    return response.content


if __name__ == "__main__":
    # image_path = "images/file_1763616627947_9470_qajqku.png"
    image_path = "images/file_1763616627958_9582_nllmgg.png"

    if not Path(image_path).exists():
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        sys.exit(1)

    print(f"OCR 처리 중: {image_path}\n")
    result = ocr_image(image_path)
    print(result)

"""
OCR 처리 중: images/file_1763616627958_9582_nllmgg.png

[llm/start] [llm:ChatOllama] Entering LLM run with input:
{
  "prompts": [
    "Human: 이 이미지에서 텍스트를 모두 추출해주세요.\n- 원본 텍스트의 구조와 줄바꿈을 최대한 유지하세요.\n- 텍스트 외 설명은 포함하지 마세요.\n- 텍스트가 없으면 '텍스트 없음' 이라고 출력하세요."
  ]
}
[llm/end] [llm:ChatOllama] [34.14s] Exiting LLM run with output:
{
  "generations": [
    [
      {
        "text": "청정 제주가 통째로 담긴\n잇츄 제주산\n동결건조 간식\n\n무엇이 특별할까요?\n\n한라산 흑돼지\n· 지방 함량이 적은 다리살 100%\n· 첨가물 없이 고단백 에너지 공급\n· 제주도 제철 원물, 수량 한정\n\n서귀포 은갈치\n· 강아지 간식 최초 제주산 은갈치\n· 잔가시 제거한 순살로 간편 급여\n· 무염처리 완료, 가을/겨울철 한정 별미\n\n구좌 당근\n· 겨울에 더 달콤한 제주 구좌 당근\n· 일반 당근보다 높은 식이섬유 함유\n· 제주도 제철 원물, 수량 한정",
        "generation_info": {
          "model": "qwen3-vl:235b",
          "created_at": "2026-03-20T09:28:02.282417158Z",
          "done": true,
          "done_reason": "stop",
          "total_duration": 32366079415,
          "load_duration": null,
          "prompt_eval_count": 1870,
          "prompt_eval_duration": null,
          "eval_count": 1023,
          "eval_duration": null,
          "logprobs": null,
          "model_name": "qwen3-vl:235b",
          "model_provider": "ollama"
        },
        "type": "ChatGeneration",
        "message": {
          "lc": 1,
          "type": "constructor",
          "id": [
            "langchain",
            "schema",
            "messages",
            "AIMessage"
          ],
          "kwargs": {
            "content": "청정 제주가 통째로 담긴\n잇츄 제주산\n동결건조 간식\n\n무엇이 특별할까요?\n\n한라산 흑돼지\n· 지방 함량이 적은 다리살 100%\n· 첨가물 없이 고단백 에너지 공급\n· 제주도 제철 원물, 수량 한정\n\n서귀포 은갈치\n· 강아지 간식 최초 제주산 은갈치\n· 잔가시 제거한 순살로 간편 급여\n· 무염처리 완료, 가을/겨울철 한정 별미\n\n구좌 당근\n· 겨울에 더 달콤한 제주 구좌 당근\n· 일반 당근보다 높은 식이섬유 함유\n· 제주도 제철 원물, 수량 한정",
            "response_metadata": {
              "model": "qwen3-vl:235b",
              "created_at": "2026-03-20T09:28:02.282417158Z",
              "done": true,
              "done_reason": "stop",
              "total_duration": 32366079415,
              "load_duration": null,
              "prompt_eval_count": 1870,
              "prompt_eval_duration": null,
              "eval_count": 1023,
              "eval_duration": null,
              "logprobs": null,
              "model_name": "qwen3-vl:235b",
              "model_provider": "ollama"
            },
            "type": "ai",
            "id": "lc_run--019d0a92-391c-77d1-96dc-7be679706e15-0",
            "usage_metadata": {
              "input_tokens": 1870,
              "output_tokens": 1023,
              "total_tokens": 2893
            },
            "tool_calls": [],
            "invalid_tool_calls": []
          }
        }
      }
    ]
  ],
  "llm_output": null,
  "run": null,
  "type": "LLMResult"
}
청정 제주가 통째로 담긴
잇츄 제주산
동결건조 간식

무엇이 특별할까요?

한라산 흑돼지
· 지방 함량이 적은 다리살 100%
· 첨가물 없이 고단백 에너지 공급
· 제주도 제철 원물, 수량 한정

서귀포 은갈치
· 강아지 간식 최초 제주산 은갈치
· 잔가시 제거한 순살로 간편 급여
· 무염처리 완료, 가을/겨울철 한정 별미

구좌 당근
· 겨울에 더 달콤한 제주 구좌 당근
· 일반 당근보다 높은 식이섬유 함유
· 제주도 제철 원물, 수량 한정
"""
