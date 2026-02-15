#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_researchers.py
코스맥스 연구원 엑셀 데이터를 JSON 데이터베이스로 변환하는 스크립트

엑셀 파일 구조:
- 파일: data/researchers.xlsx
- 시트 '이니셜 검색': 검색 인터페이스 (D4, D5 검색창), 실제 데이터 B8부터
- 시트 '이니셜 명단': 전체 연구원 명단 (헤더 + 데이터)
  - 컬럼: 이름, 이니셜, 연구소, 랩, 팀, 직책

이메일 정책:
- 코스맥스 이메일 형식: firstname.lastname@cosmax.com
- 단, 사람마다 형식이 다를 수 있으므로 자동 생성하지 않음
- 수동 확인 필요 → MANUAL_CHECK_REQUIRED@cosmax.com 플레이스홀더 사용
- data/email_overrides.json 에 확인된 이메일을 등록하면 자동 반영

email_overrides.json 형식 예시:
{
  "AHJ01": "hyunjung.ahn@cosmax.com",
  "SYW01": "yewon.shin@cosmax.com"
}
"""

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# === 설정 ===
EXCEL_PATH = Path("data/researchers.xlsx")
OUTPUT_PATH = Path("data/researcher_db.json")
EMAIL_OVERRIDES_PATH = Path("data/email_overrides.json")
COMPANY_DOMAIN = "cosmax.com"
PLACEHOLDER_EMAIL = f"MANUAL_CHECK_REQUIRED@{COMPANY_DOMAIN}"

# 엑셀 시트 설정
# 시트 '이니셜 검색': 검색 인터페이스 (D4, D5에 검색창), B8부터 데이터
# 시트 '이니셜 명단': 실제 연구원 전체 명단
PREFERRED_SHEET = "이니셜 명단"
SEARCH_SHEET = "이니셜 검색"
SEARCH_SHEET_SKIPROWS = 7  # B8부터 데이터 시작 → skiprows=7


def _normalize_column(val: Any) -> str:
    """컬럼명 정규화: 공백/개행 제거"""
    if val is None:
        return ""
    return str(val).replace("\n", "").replace("\r", "").replace("\t", "").replace(" ", "").strip()


def _find_header_row(path: Path, max_scan: int = 60) -> tuple[str, int]:
    """'이름'과 '이니셜' 컬럼이 함께 있는 헤더 행을 자동 탐색"""
    xls = pd.ExcelFile(path, engine="openpyxl")
    print(f"[INFO] 시트 목록: {xls.sheet_names}")

    # 우선순위: '이니셜 명단' 시트 먼저, 그 다음 나머지
    sheet_order = []
    if PREFERRED_SHEET in xls.sheet_names:
        sheet_order.append(PREFERRED_SHEET)
    sheet_order.extend(s for s in xls.sheet_names if s != PREFERRED_SHEET)

    for sheet in sheet_order:
        raw = pd.read_excel(path, engine="openpyxl", sheet_name=sheet, header=None)
        scan_n = min(max_scan, len(raw))
        for i in range(scan_n):
            row = raw.iloc[i].astype(str).map(_normalize_column)
            if (row == "이름").any() and (row == "이니셜").any():
                return sheet, i

    raise ValueError("헤더(이름/이니셜) 행을 찾을 수 없습니다. 엑셀 구조를 확인하세요.")


def load_excel_data() -> pd.DataFrame:
    """엑셀 파일에서 연구원 데이터 로드 (openpyxl 엔진 사용)"""
    if not EXCEL_PATH.exists():
        print(f"[에러] 엑셀 파일을 찾을 수 없습니다: {EXCEL_PATH.resolve()}")
        sys.exit(1)

    print(f"[INFO] 엑셀 파일 로드 중: {EXCEL_PATH}")

    # 헤더 행 자동 감지
    sheet, header_row = _find_header_row(EXCEL_PATH)
    print(f"[INFO] 시트 '{sheet}', 헤더 행(0-index): {header_row}")

    df = pd.read_excel(
        EXCEL_PATH,
        sheet_name=sheet,
        header=header_row,
        engine="openpyxl",
    )

    # 'Unnamed' 컬럼 및 빈 컬럼 제거
    keep_cols = [
        c for c in df.columns
        if str(c).strip() and str(c).strip().lower() != "nan" and not str(c).startswith("Unnamed")
    ]
    df = df[keep_cols]

    # 디버깅: 감지된 컬럼명 출력
    print(f"[DEBUG] 감지된 컬럼: {[str(c).strip() for c in df.columns]}")
    print(f"[DEBUG] 전체 데이터 행 수 (정리 전): {len(df)}")

    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """데이터프레임 정리: 컬럼 매핑, 빈 행 제거, 한글 인코딩 처리"""
    # 컬럼명 정규화 매핑 (유연한 매칭)
    col_map = {_normalize_column(c): c for c in df.columns}

    name_col = col_map.get("이름")
    code_col = col_map.get("이니셜")

    if name_col is None or code_col is None:
        print(f"[에러] 필수 컬럼(이름/이니셜)을 찾을 수 없습니다.")
        print(f"[에러] 사용 가능한 컬럼: {list(col_map.keys())}")
        sys.exit(1)

    # 부서 관련 컬럼 (다양한 이름 대응)
    dept_col = col_map.get("연구소") or col_map.get("부서") or col_map.get("소속")
    lab_col = col_map.get("랩")
    team_col = col_map.get("팀")
    pos_col = col_map.get("직책") or col_map.get("직급")

    # 표준 컬럼명으로 변환
    rename_map = {}
    if name_col:
        rename_map[name_col] = "이름"
    if code_col:
        rename_map[code_col] = "이니셜"
    if dept_col:
        rename_map[dept_col] = "연구소"
    if lab_col:
        rename_map[lab_col] = "랩"
    if team_col:
        rename_map[team_col] = "팀"
    if pos_col:
        rename_map[pos_col] = "직책"

    df = df.rename(columns=rename_map)

    # 빈 행 제거 (이름과 이니셜 모두 비어있는 행)
    df = df.dropna(subset=["이름", "이니셜"], how="all")

    # 문자열 컬럼 정리
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("nan", "")

    print(f"[DEBUG] 데이터 행 수 (정리 후): {len(df)}")

    return df


def load_email_overrides() -> dict[str, str]:
    """수동으로 확인된 이메일 오버라이드 파일 로드

    지원 형식:
      {"AHJ01": "name@cosmax.com"}
      또는
      {"AHJ01": {"email": "name@cosmax.com"}}
    """
    if not EMAIL_OVERRIDES_PATH.exists():
        print(f"[INFO] 이메일 오버라이드 파일 없음: {EMAIL_OVERRIDES_PATH}")
        return {}

    try:
        with EMAIL_OVERRIDES_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[경고] 이메일 오버라이드 파일 읽기 실패: {e}")
        return {}

    if not isinstance(data, dict):
        print("[경고] email_overrides.json 형식이 올바르지 않습니다 (JSON object 필요)")
        return {}

    overrides: dict[str, str] = {}
    for key, val in data.items():
        if not key:
            continue

        email = ""
        if isinstance(val, str):
            email = val.strip()
        elif isinstance(val, dict) and isinstance(val.get("email"), str):
            email = val["email"].strip()

        if email:
            if "@" not in email:
                print(f"[경고] 이메일 형식 이상 (code={key}): {email}")
                continue
            overrides[str(key).strip()] = email

    print(f"[INFO] 이메일 오버라이드 로드 완료: {len(overrides)}건")
    return overrides


def build_researcher_db(df: pd.DataFrame, email_overrides: dict[str, str]) -> dict[str, dict[str, Any]]:
    """연구원 데이터베이스 딕셔너리 생성"""
    db: dict[str, dict[str, Any]] = {}
    skipped = 0
    dup_codes: list[str] = []
    override_applied = 0

    for _, row in df.iterrows():
        name = str(row.get("이름", "")).strip()
        code = str(row.get("이니셜", "")).strip()

        # 빈 데이터 건너뛰기
        if not name or name.lower() == "nan" or not code or code.lower() == "nan":
            skipped += 1
            continue

        # 중복 코드 체크
        if code in db:
            dup_codes.append(code)
            continue

        # 이메일 결정: 오버라이드 > 대문자 오버라이드 > 플레이스홀더
        email = PLACEHOLDER_EMAIL
        email_verified = False
        note = "실제 이메일 주소 확인 후 수동 입력 필요"

        if code in email_overrides:
            email = email_overrides[code]
            email_verified = True
            note = "이메일 수동 확인 완료 (override 적용)"
            override_applied += 1
        elif code.upper() in email_overrides:
            email = email_overrides[code.upper()]
            email_verified = True
            note = "이메일 수동 확인 완료 (override 적용)"
            override_applied += 1

        # 연구원 정보 구성
        db[code] = {
            "code": code,
            "name": name,
            "email": email,
            "department": str(row.get("연구소", "")).strip() if "연구소" in df.columns else "",
            "lab": str(row.get("랩", "")).strip() if "랩" in df.columns else "",
            "team": str(row.get("팀", "")).strip() if "팀" in df.columns else "",
            "position": str(row.get("직책", "")).strip() if "직책" in df.columns else "",
            "email_verified": email_verified,
            "note": note,
        }

    # 처리 로그
    if skipped > 0:
        print(f"[INFO] 빈 행 {skipped}건 건너뜀")
    if dup_codes:
        print(f"[INFO] 중복 코드 {len(dup_codes)}건 건너뜀: {dup_codes[:5]}")
    print(f"[INFO] 이메일 override 적용: {override_applied}건")

    return db


def save_database(db: dict[str, dict[str, Any]]) -> None:
    """JSON 데이터베이스 파일 저장 (한글 지원: ensure_ascii=False)"""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 데이터베이스 저장 완료: {OUTPUT_PATH.resolve()}")


def print_summary(db: dict[str, dict[str, Any]], df: pd.DataFrame) -> None:
    """처리 결과 요약 출력"""
    total = len(db)
    verified = sum(1 for r in db.values() if r["email_verified"])
    unverified = total - verified

    print("\n" + "=" * 60)
    print("  코스맥스 연구원 데이터베이스 생성 결과")
    print("=" * 60)

    # 기본 통계
    print(f"\n[통계]")
    print(f"  총 연구원 수: {total}명")
    print(f"  이메일 확인 완료: {verified}명")
    print(f"  이메일 미확인 (수동 확인 필요): {unverified}명")
    print(f"  감지된 컬럼: {df.columns.tolist()}")

    # 부서별 통계
    if "연구소" in df.columns:
        dept_counts = df["연구소"].value_counts()
        print(f"\n[부서별 인원 (상위 10개)]")
        for dept, count in dept_counts.head(10).items():
            if dept and dept != "nan" and dept.strip():
                print(f"  {dept}: {count}명")

    # 샘플 엔트리 (처음 5개)
    print(f"\n[샘플 엔트리 (처음 5개)]")
    for i, (code, info) in enumerate(db.items()):
        if i >= 5:
            break
        status = "✓" if info["email_verified"] else "✗"
        print(f"  [{status}] {code}: {info['name']}")
        print(f"       부서: {info['department']} > {info['lab']} > {info['team']}")
        print(f"       직책: {info['position']}")
        print(f"       이메일: {info['email']}")

    # 이메일 확인 안내
    print(f"\n[이메일 수동 확인 안내]")
    print(f"  1. 코스맥스 이메일 형식: firstname.lastname@{COMPANY_DOMAIN}")
    print(f"     (예: junyoung.ha@{COMPANY_DOMAIN})")
    print(f"  2. 단, 사람마다 형식이 다를 수 있어 자동 생성하지 않음")
    print(f"  3. 확인된 이메일은 '{EMAIL_OVERRIDES_PATH}'에 추가:")
    print(f'     예시: {{"JHK01": "junhyeok.kim@{COMPANY_DOMAIN}"}}')
    print(f"  4. 이 스크립트를 다시 실행하면 오버라이드가 자동 반영됨")

    # 미확인 override 경고
    if EMAIL_OVERRIDES_PATH.exists():
        try:
            with EMAIL_OVERRIDES_PATH.open("r", encoding="utf-8") as f:
                overrides = json.load(f)
            unknown = [c for c in overrides if c not in db]
            if unknown:
                print(f"\n[경고] 엑셀에 없는 코드가 override에 존재: {unknown}")
        except Exception:
            pass

    print(f"\n  출력 파일: {OUTPUT_PATH.resolve()}")
    print("=" * 60)


def main() -> None:
    """메인 실행 함수"""
    print("코스맥스 연구원 데이터 전처리 시작...\n")

    # 1. 이메일 오버라이드 로드
    email_overrides = load_email_overrides()

    # 2. 엑셀 데이터 로드 (openpyxl 엔진)
    df = load_excel_data()

    # 3. 데이터프레임 정리 (한글 인코딩, 컬럼 매핑, 빈 행 제거)
    df = clean_dataframe(df)

    # 4. 연구원 데이터베이스 생성
    db = build_researcher_db(df, email_overrides)

    if not db:
        print("[에러] 생성된 연구원 데이터가 없습니다. 엑셀 파일을 확인해주세요.")
        sys.exit(1)

    # 5. JSON 저장 (ensure_ascii=False 로 한글 보존)
    save_database(db)

    # 6. 결과 요약 출력
    print_summary(db, df)


if __name__ == "__main__":
    main()
