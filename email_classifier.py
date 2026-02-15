#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
email_classifier.py
코스맥스 연구원 이메일 분류기 (Gemini AI 기반)

수신 이메일을 분석하여 다음을 수행:
1. 카테고리 분류 (원료 문의, 처방 요청, 품질 이슈, 일정 조율 등)
2. 긴급도 판별 (긴급/높음/보통/낮음)
3. 담당 연구소/랩 자동 매칭
4. 담당자 추천 (researcher_db.json 기반)
5. 요약 및 추천 액션 생성

사용법:
  python3 email_classifier.py                    # 데모 이메일로 테스트
  python3 email_classifier.py --interactive      # 대화형 모드
  python3 email_classifier.py --file email.txt   # 파일에서 이메일 읽기
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# .env 파일 로드 (프로젝트 루트)
load_dotenv(Path(__file__).parent / ".env")

from google import genai
from google.genai import types

# === 설정 ===
RESEARCHER_DB_PATH = Path("data/researcher_db.json")
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",       # fallback: 별도 할당량
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",  # fallback: 경량 모델
]

# 코스맥스 이메일 분류 카테고리
CATEGORIES = {
    "원료_문의": "원료 관련 문의, 원료 스펙, 원료 추천, 원료 변경",
    "처방_요청": "신제품 처방 개발, 처방 변경, 처방 최적화 요청",
    "품질_이슈": "제품 품질 문제, 클레임, 불량, 안정성 이슈",
    "일정_조율": "개발 일정, 납기, 미팅 일정, 샘플 일정",
    "규제_인허가": "인허가, 규제, 성분 규제, 수출 규정, INCI",
    "샘플_요청": "샘플 제작, 샘플 발송, 시제품 요청",
    "기술_검토": "기술 검토, 특허, 기술 자문, 공정 문의",
    "견적_계약": "견적서, 단가, 계약, MOQ, 거래 조건",
    "기타": "위 카테고리에 해당하지 않는 일반 문의",
}

# 긴급도 레벨
URGENCY_LEVELS = {
    "긴급": "즉시 대응 필요 (품질 사고, 라인 중단, 클레임 등)",
    "높음": "당일 또는 익일 대응 필요 (납기 임박, 고객 긴급 요청)",
    "보통": "일반적인 업무 처리 (3-5일 내 대응)",
    "낮음": "참고/정보 공유 성격 (일주일 이내 대응)",
}


@dataclass
class EmailInput:
    """분류할 이메일 입력 데이터"""
    subject: str
    body: str
    sender: str = ""
    date: str = ""


@dataclass
class ClassificationResult:
    """이메일 분류 결과"""
    category: str = ""
    category_description: str = ""
    urgency: str = ""
    urgency_reason: str = ""
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    recommended_department: str = ""
    recommended_lab: str = ""
    recommended_team: str = ""
    recommended_researchers: list[dict] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)
    raw_response: str = ""


def load_researcher_db() -> dict:
    """연구원 데이터베이스 로드"""
    if not RESEARCHER_DB_PATH.exists():
        print(f"[경고] 연구원 DB 없음: {RESEARCHER_DB_PATH}")
        print("[경고] preprocess_researchers.py 를 먼저 실행하세요.")
        return {}

    with RESEARCHER_DB_PATH.open("r", encoding="utf-8") as f:
        db = json.load(f)

    print(f"[INFO] 연구원 DB 로드: {len(db)}명")
    return db


def get_department_summary(researcher_db: dict) -> str:
    """연구원 DB에서 부서/랩/팀 구조 요약 생성 (Gemini 프롬프트용)"""
    departments: dict[str, dict[str, set]] = {}

    for info in researcher_db.values():
        dept = info.get("department", "").strip()
        lab = info.get("lab", "").strip()
        team = info.get("team", "").strip()

        if not dept or dept == "nan" or dept == "-":
            continue

        if dept not in departments:
            departments[dept] = {"labs": set(), "teams": set()}
        if lab:
            departments[dept]["labs"].add(lab)
        if team:
            departments[dept]["teams"].add(team)

    lines = []
    for dept, info in sorted(departments.items()):
        labs_str = ", ".join(sorted(info["labs"]))
        lines.append(f"- {dept}: {labs_str}")

    return "\n".join(lines)


def configure_gemini() -> tuple[genai.Client, str]:
    """Gemini API 클라이언트 설정 및 사용 가능한 모델 탐색

    Returns:
        (client, model_name) 튜플. 모델 우선순위에 따라 첫 번째로 응답 가능한 모델 선택.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[에러] API 키가 설정되지 않았습니다.")
        print("  방법 1: export GEMINI_API_KEY='your-key'")
        print("  방법 2: .env 파일에 GEMINI_API_KEY=your-key 또는 GOOGLE_API_KEY=your-key 추가")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # 모델 우선순위대로 간단한 ping 테스트
    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents="ping",
                config=types.GenerateContentConfig(max_output_tokens=8),
            )
            print(f"[INFO] Gemini 모델 선택 완료: {model_name}")
            return client, model_name
        except Exception as e:
            reason = "quota" if "429" in str(e) else str(e)[:60]
            print(f"[INFO] {model_name} 사용 불가 ({reason}), 다음 모델 시도...")

    print(f"[에러] 사용 가능한 Gemini 모델이 없습니다. API 키 또는 할당량을 확인하세요.")
    sys.exit(1)


def build_classification_prompt(email: EmailInput, dept_summary: str) -> str:
    """이메일 분류를 위한 Gemini 프롬프트 생성"""

    categories_text = "\n".join(
        f"  - {k}: {v}" for k, v in CATEGORIES.items()
    )
    urgency_text = "\n".join(
        f"  - {k}: {v}" for k, v in URGENCY_LEVELS.items()
    )

    prompt = f"""당신은 코스맥스(Cosmax) 화장품 OEM 회사의 이메일 분류 전문가입니다.
코스맥스는 한국의 화장품 OEM/ODM 기업으로, 스킨케어, 메이크업, 선케어 등을 연구·개발·생산합니다.

아래 이메일을 분석하여 JSON 형식으로 분류 결과를 반환하세요.

=== 이메일 정보 ===
발신자: {email.sender or '(미상)'}
날짜: {email.date or '(미상)'}
제목: {email.subject}

본문:
{email.body}

=== 분류 카테고리 ===
{categories_text}

=== 긴급도 레벨 ===
{urgency_text}

=== 코스맥스 연구소 구조 ===
{dept_summary}

=== 응답 형식 (반드시 JSON만 반환) ===
{{
  "category": "카테고리명 (위 목록에서 선택)",
  "category_description": "해당 카테고리로 분류한 이유 (1문장)",
  "urgency": "긴급/높음/보통/낮음",
  "urgency_reason": "긴급도 판단 근거 (1문장)",
  "summary": "이메일 핵심 내용 요약 (2-3문장)",
  "key_points": ["핵심 포인트1", "핵심 포인트2"],
  "recommended_department": "추천 담당 연구소",
  "recommended_lab": "추천 담당 랩",
  "recommended_team": "추천 담당 팀 (알 수 없으면 빈 문자열)",
  "suggested_actions": ["추천 액션1", "추천 액션2", "추천 액션3"]
}}

중요:
- 반드시 유효한 JSON만 반환하세요. 설명이나 마크다운 없이 JSON만 출력하세요.
- 코스맥스 연구소 구조를 참고하여 가장 적합한 부서를 추천하세요.
- 모든 응답은 한국어로 작성하세요.
"""
    return prompt


def find_matching_researchers(
    researcher_db: dict, department: str, lab: str, team: str, limit: int = 5
) -> list[dict]:
    """분류 결과에 맞는 담당자 후보 검색"""
    candidates = []

    for code, info in researcher_db.items():
        score = 0
        r_dept = info.get("department", "")
        r_lab = info.get("lab", "")
        r_team = info.get("team", "")

        # 연구소 매칭
        if department and r_dept and department in r_dept:
            score += 3
        # 랩 매칭
        if lab and r_lab and lab in r_lab:
            score += 2
        # 팀 매칭
        if team and r_team and team in r_team:
            score += 1

        if score > 0:
            candidates.append({
                "code": code,
                "name": info["name"],
                "department": r_dept,
                "lab": r_lab,
                "team": r_team,
                "position": info.get("position", ""),
                "email": info.get("email", ""),
                "email_verified": info.get("email_verified", False),
                "match_score": score,
            })

    # 점수 내림차순 정렬, 상위 N명 반환
    candidates.sort(key=lambda x: x["match_score"], reverse=True)
    return candidates[:limit]


def parse_gemini_response(response_text: str) -> dict:
    """Gemini 응답에서 JSON 파싱 (마크다운 코드블록, 중첩 블록 처리)"""
    import re

    text = response_text.strip()

    # ```json ... ``` 코드블록 추출 (가장 마지막 JSON 블록 우선)
    code_blocks = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        # 마지막 코드블록이 JSON일 가능성이 가장 높음
        for block in reversed(code_blocks):
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue

    # 코드블록이 없으면 직접 파싱 시도
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # { ... } 중 가장 큰 범위의 JSON 추출
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return {}


def classify_email(
    client: genai.Client,
    model_name: str,
    email: EmailInput,
    researcher_db: dict,
    dept_summary: str,
) -> ClassificationResult:
    """이메일 분류 실행"""
    result = ClassificationResult()

    # 1. Gemini로 분류
    prompt = build_classification_prompt(email, dept_summary)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,  # 분류 정확도를 위해 낮은 temperature
                top_p=0.8,
                max_output_tokens=8192,  # 2.5-flash는 thinking 토큰 포함하여 충분히 확보
            ),
        )
        result.raw_response = response.text
    except Exception as e:
        print(f"[에러] Gemini API 호출 실패: {e}")
        result.summary = f"분류 실패: {e}"
        return result

    # 2. 응답 파싱
    parsed = parse_gemini_response(response.text)
    if not parsed:
        print(f"[경고] Gemini 응답 JSON 파싱 실패")
        print(f"[DEBUG] 원본 응답:\n{response.text[:500]}")
        result.summary = "JSON 파싱 실패 — 원본 응답을 확인하세요"
        return result

    result.category = parsed.get("category", "기타")
    result.category_description = parsed.get("category_description", "")
    result.urgency = parsed.get("urgency", "보통")
    result.urgency_reason = parsed.get("urgency_reason", "")
    result.summary = parsed.get("summary", "")
    result.key_points = parsed.get("key_points", [])
    result.recommended_department = parsed.get("recommended_department", "")
    result.recommended_lab = parsed.get("recommended_lab", "")
    result.recommended_team = parsed.get("recommended_team", "")
    result.suggested_actions = parsed.get("suggested_actions", [])

    # 3. 담당자 매칭
    if researcher_db:
        result.recommended_researchers = find_matching_researchers(
            researcher_db,
            result.recommended_department,
            result.recommended_lab,
            result.recommended_team,
        )

    return result


def print_result(email: EmailInput, result: ClassificationResult) -> None:
    """분류 결과 출력"""
    # 긴급도별 표시
    urgency_icons = {"긴급": "🔴", "높음": "🟠", "보통": "🟢", "낮음": "⚪"}
    icon = urgency_icons.get(result.urgency, "⚪")

    print("\n" + "=" * 60)
    print("  코스맥스 이메일 분류 결과")
    print("=" * 60)

    print(f"\n[이메일 정보]")
    print(f"  제목: {email.subject}")
    if email.sender:
        print(f"  발신자: {email.sender}")

    print(f"\n[분류 결과]")
    print(f"  카테고리: {result.category}")
    print(f"  분류 근거: {result.category_description}")
    print(f"  긴급도: {icon} {result.urgency}")
    print(f"  긴급도 근거: {result.urgency_reason}")

    print(f"\n[요약]")
    print(f"  {result.summary}")

    if result.key_points:
        print(f"\n[핵심 포인트]")
        for point in result.key_points:
            print(f"  • {point}")

    print(f"\n[추천 담당부서]")
    dept_parts = [
        p for p in [result.recommended_department, result.recommended_lab, result.recommended_team] if p
    ]
    print(f"  {' > '.join(dept_parts) if dept_parts else '(판별 불가)'}")

    if result.recommended_researchers:
        print(f"\n[추천 담당자 후보]")
        for r in result.recommended_researchers:
            verified = "✓" if r["email_verified"] else "✗"
            print(f"  [{verified}] {r['name']} ({r['code']}) — {r['department']} > {r['lab']} > {r['team']} | {r['position']}")
            if r["email_verified"]:
                print(f"       이메일: {r['email']}")
    else:
        print(f"\n[추천 담당자]")
        print(f"  매칭되는 담당자를 찾지 못했습니다.")

    if result.suggested_actions:
        print(f"\n[추천 액션]")
        for i, action in enumerate(result.suggested_actions, 1):
            print(f"  {i}. {action}")

    print("\n" + "=" * 60)


def get_demo_emails() -> list[EmailInput]:
    """테스트용 데모 이메일 목록"""
    return [
        EmailInput(
            subject="[긴급] 선크림 SPF 테스트 결과 이상 — 출하 보류 요청",
            body="""안녕하세요, OO브랜드 품질관리팀 김수현입니다.

금일 입고된 선크림 LOT#2025-0892 에 대해 자체 SPF 테스트를 진행한 결과,
표기 SPF 50+ 대비 실측값이 SPF 38로 확인되었습니다.

해당 LOT 출하를 즉시 보류해 주시고, 코스맥스 측 QC 데이터 및
원인 분석 결과를 금일 중 공유 부탁드립니다.

출하 예정일이 내일(2/16)이라 매우 긴급합니다.

감사합니다.
김수현 드림""",
            sender="soohyun.kim@oobrand.com",
            date="2026-02-15",
        ),
        EmailInput(
            subject="신규 비건 파운데이션 처방 개발 의뢰",
            body="""코스맥스 연구소 담당자님께,

저희 AB코스메틱에서 2026 F/W 시즌 신제품으로
비건 인증 가능한 리퀴드 파운데이션 개발을 의뢰드리고자 합니다.

주요 요구사항:
1. 비건 인증 (한국비건인증원 또는 EVE VEGAN)
2. 커버력 중~고 수준
3. 12시간 지속력
4. 색상 10호~25호 (6 shade)
5. 타겟 단가: 개당 3,500원 이내 (MOQ 10,000개 기준)

3월 초까지 초기 샘플 2-3안 검토 가능할까요?
가능한 일정과 기술 미팅 날짜를 잡아주시면 감사하겠습니다.

AB코스메틱 상품기획팀
박지연 과장 (jiyeon.park@abcosmetic.co.kr)""",
            sender="jiyeon.park@abcosmetic.co.kr",
            date="2026-02-14",
        ),
        EmailInput(
            subject="히알루론산 원료 수급 관련 문의",
            body="""안녕하세요, 코스맥스 원료 담당자님.

저희가 공급 중인 저분자 히알루론산(HA-LMW-500) 원료와 관련하여,
3월분 발주량 확인 요청드립니다.

현재 글로벌 수급 상황이 다소 타이트하여
2주 전 사전 발주가 필요한 상황입니다.

참고로, 신규 원료 고분자 히알루론산(HA-HMW-2000)도 출시되었으니
스펙시트 첨부합니다. 검토 후 테스트 희망 시 샘플 발송 가능합니다.

문의사항 있으시면 연락 부탁드립니다.

(주)바이오소재
영업팀 이정호""",
            sender="jungho.lee@biomaterials.co.kr",
            date="2026-02-13",
        ),
    ]


def run_interactive_mode(
    client: genai.Client,
    model_name: str,
    researcher_db: dict,
    dept_summary: str,
) -> None:
    """대화형 모드: 사용자가 이메일을 입력하여 분류"""
    print("\n[대화형 모드] 이메일 정보를 입력하세요. (종료: Ctrl+C 또는 빈 제목)")

    while True:
        try:
            print("\n" + "-" * 40)
            subject = input("이메일 제목: ").strip()
            if not subject:
                print("종료합니다.")
                break

            sender = input("발신자 (선택, Enter로 건너뛰기): ").strip()

            print("본문 (입력 후 빈 줄에서 'END' 입력):")
            body_lines = []
            while True:
                line = input()
                if line.strip().upper() == "END":
                    break
                body_lines.append(line)
            body = "\n".join(body_lines)

            if not body:
                print("[경고] 본문이 비어있습니다.")
                continue

            email = EmailInput(subject=subject, body=body, sender=sender)

            print("\n분류 중...")
            result = classify_email(client, model_name, email, researcher_db, dept_summary)
            print_result(email, result)

        except KeyboardInterrupt:
            print("\n\n종료합니다.")
            break


def run_file_mode(
    filepath: str,
    client: genai.Client,
    model_name: str,
    researcher_db: dict,
    dept_summary: str,
) -> None:
    """파일에서 이메일 읽어서 분류"""
    path = Path(filepath)
    if not path.exists():
        print(f"[에러] 파일을 찾을 수 없습니다: {path}")
        sys.exit(1)

    text = path.read_text(encoding="utf-8")

    # 간단한 파싱: 첫 줄을 제목, 나머지를 본문으로 처리
    lines = text.strip().split("\n")
    subject = lines[0].strip()

    # "Subject:" 접두사 제거
    if subject.lower().startswith("subject:"):
        subject = subject[len("subject:"):].strip()

    body = "\n".join(lines[1:]).strip()

    email = EmailInput(subject=subject, body=body)

    print(f"[INFO] 파일에서 이메일 로드: {path}")
    result = classify_email(client, model_name, email, researcher_db, dept_summary)
    print_result(email, result)


def main() -> None:
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="코스맥스 이메일 분류기 (Gemini AI 기반)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="대화형 모드로 실행",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="이메일 텍스트 파일 경로 (첫 줄: 제목, 나머지: 본문)",
    )
    parser.add_argument(
        "--demo-index",
        type=int,
        default=None,
        help="데모 이메일 인덱스 (0, 1, 2). 미지정 시 전체 실행",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  코스맥스 이메일 분류기 (Gemini AI)")
    print("=" * 60)

    # 1. 연구원 DB 로드
    researcher_db = load_researcher_db()
    dept_summary = get_department_summary(researcher_db) if researcher_db else "(연구원 DB 없음)"

    # 2. Gemini 클라이언트 설정 (모델 자동 fallback)
    client, model_name = configure_gemini()

    # 3. 실행 모드 분기
    if args.interactive:
        run_interactive_mode(client, model_name, researcher_db, dept_summary)
    elif args.file:
        run_file_mode(args.file, client, model_name, researcher_db, dept_summary)
    else:
        # 데모 모드
        demos = get_demo_emails()

        if args.demo_index is not None:
            if 0 <= args.demo_index < len(demos):
                demos = [demos[args.demo_index]]
            else:
                print(f"[에러] demo-index는 0~{len(demos)-1} 범위여야 합니다.")
                sys.exit(1)

        print(f"\n[데모 모드] {len(demos)}개 테스트 이메일 분류 시작...\n")

        for i, email in enumerate(demos):
            print(f"\n{'#' * 60}")
            print(f"  데모 이메일 {i + 1}/{len(demos)}")
            print(f"{'#' * 60}")

            result = classify_email(client, model_name, email, researcher_db, dept_summary)
            print_result(email, result)

        print(f"\n[완료] {len(demos)}개 이메일 분류 완료")
        print(f"[TIP] 대화형 모드: python3 email_classifier.py --interactive")
        print(f"[TIP] 파일 입력: python3 email_classifier.py --file email.txt")


if __name__ == "__main__":
    main()
