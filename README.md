# daily-news-digest

RSS에서 뉴스를 수집하고(필요 시 본문 확보), AI로 요약/품질 판단을 수행한 뒤 일일 다이제스트(JSON)를 생성합니다.

## 구조

- `src/daily_news_digest/`: 소스 코드
- `data/`: 결과/히스토리 JSON
- `config/`: 설정 템플릿
- `tests/`: 테스트

## 빠른 시작

1) 의존성 설치

```bash
python3 -m pip install -r requirements.txt
```

2) 환경 변수 설정

- `config/.env.template`를 참고해 리포지토리 루트에 `.env`를 준비합니다.

3) 실행

```bash
python3 -m daily_news_digest
```
