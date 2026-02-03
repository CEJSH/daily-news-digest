import datetime
from typing import Any
from jinja2 import Template

def generate_html(
    grouped_items: dict[str, list[dict[str, Any]]],
    top_items: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    print("📝 HTML 뉴스레터를 생성하는 중입니다...")

    html_template = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="utf-8" />
        <title>{{ title }}</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
                background-color: #f4f4f4;
                padding: 20px;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: #ffffff;
                padding: 28px;
                border-radius: 12px;
                box-shadow: 0 5px 20px rgba(0, 0, 0, 0.05);
            }
            h1 {
                color: #1f2933;
                text-align: center;
                border-bottom: 2px solid #e5e7eb;
                padding-bottom: 16px;
                margin-top: 0;
                margin-bottom: 8px;
            }
            .date {
                text-align: center;
                color: #9ca3af;
                font-size: 13px;
                margin-bottom: 20px;
            }
            .intro {
                font-size: 14px;
                color: #4b5563;
                line-height: 1.6;
                margin-bottom: 24px;
            }

            /* TOP 3 섹션 */
            .top-section {
                margin-bottom: 28px;
                padding: 16px;
                border-radius: 10px;
                background: #f9fafb;
                border: 1px solid #e5e7eb;
            }
            .top-section-title {
                font-size: 16px;
                font-weight: 700;
                color: #111827;
                margin-bottom: 12px;
            }
            .top-list {
                display: grid;
                grid-template-columns: 1fr;
                gap: 12px;
            }
            @media (min-width: 720px) {
                .top-list {
                    grid-template-columns: 1fr 1fr;
                }
            }
            .top-item {
                padding: 12px 14px;
                border-radius: 10px;
                background: #ffffff;
                border: 1px solid #e5e7eb;
            }
            .top-rank {
                font-size: 12px;
                font-weight: 700;
                color: #2563eb;
                margin-bottom: 4px;
            }
            .top-topic {
                font-size: 11px;
                color: #6b7280;
                margin-bottom: 2px;
            }
            .top-source {
                font-size: 11px;
                color: #9ca3af;
                margin-bottom: 4px;
            }
            .top-title {
                font-size: 15px;
                font-weight: 600;
                color: #111827;
                text-decoration: none;
            }
            .top-title:hover {
                text-decoration: underline;
            }
            .top-summary {
                margin-top: 6px;
                font-size: 13px;
                color: #4b5563;
                line-height: 1.5;
            }
            .top-published {
                margin-top: 4px;
                font-size: 11px;
                color: #9ca3af;
            }

            /* 주제별 섹션 */
            .topic-section {
                margin-top: 24px;
                margin-bottom: 12px;
                padding-top: 12px;
                border-top: 1px solid #e5e7eb;
            }
            .topic-title {
                font-size: 16px;
                font-weight: 700;
                color: #111827;
                margin-bottom: 10px;
            }
            .news-item {
                margin-bottom: 18px;
            }
            .news-title {
                font-size: 14px;
                font-weight: 600;
                color: #2563eb;
                text-decoration: none;
            }
            .news-title:hover {
                text-decoration: underline;
            }
            .news-summary {
                color: #4b5563;
                font-size: 13px;
                margin-top: 4px;
                line-height: 1.5;
            }
            .published {
                font-size: 11px;
                color: #9ca3af;
                margin-top: 3px;
            }
            .source {
                font-size: 11px;
                color: #9ca3af;
                margin-top: 2px;
            }

            .ad-block {
                background-color: #fff7ed;
                border: 1px solid #fed7aa;
                color: #9a3412;
                padding: 16px;
                text-align: center;
                margin-top: 32px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 14px;
            }
            .ad-link {
                text-decoration: none;
                color: #dc2626;
            }
            .ad-link:hover {
                text-decoration: underline;
            }
            .footer {
                text-align: center;
                font-size: 11px;
                color: #9ca3af;
                margin-top: 24px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{{ title }}</h1>
            <div class="date">{{ date }}</div>

            <div class="intro">
                오늘 세계 흐름을 읽는 데 중요한
                <strong>AI · 반도체 · 에너지 · 바이오 · 규제 · 금융</strong> 뉴스를
                한 번에 모았습니다. 맨 위에는 강화된 기준으로 선별한
                <strong>TOP {{ top_count }} 핵심 뉴스</strong>가, 그 아래에는
                주제별 섹션이 이어집니다.
            </div>

            {% if top_items %}
            <div class="top-section">
                <div class="top-section-title">🔥 오늘의 핵심 TOP {{ top_count }}</div>
                <div class="top-list">
                    {% for item in top_items %}
                    <div class="top-item">
                        <div class="top-rank">TOP {{ loop.index }}</div>
                        <div class="top-topic">{{ item.topic }}</div>
                        {% if item.source %}
                        <div class="top-source">{{ item.source }}</div>
                        {% endif %}
                        <a href="{{ item.link }}" target="_blank" class="top-title">{{ item.title }}</a>
                        {% if item.published %}
                        <div class="top-published">{{ item.published }}</div>
                        {% endif %}
                        <div class="top-summary">{{ item.summary }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}

            {% for topic, items in grouped_items.items() %}
            <div class="topic-section">
                <div class="topic-title">📌 {{ topic }}</div>
                {% for item in items %}
                    <div class="news-item">
                        <a href="{{ item.link }}" class="news-title" target="_blank">👉 {{ item.title }}</a>
                        {% if item.source %}
                        <div class="source">{{ item.source }}</div>
                        {% endif %}
                        {% if item.published %}
                        <div class="published">{{ item.published }}</div>
                        {% endif %}
                        <p class="news-summary">{{ item.summary }}</p>
                    </div>
                {% endfor %}
            </div>
            {% endfor %}

            <div class="ad-block">
                <a href="{{ ad_link }}" class="ad-link" target="_blank">{{ ad_text }}</a>
            </div>

            <div class="footer">
                Automated by DAILY WORLD v1.0<br />
                이 페이지는 개인용 자동 뉴스 요약 봇이 생성했습니다.
            </div>
        </div>
    </body>
    </html>
    """

    template = Template(html_template)
    today = datetime.datetime.now().strftime("%Y년 %m월 %d일 (%a)")

    return template.render(
        title=config["newsletter_title"],
        date=today,
        grouped_items=grouped_items,
        top_items=top_items,
        top_count=len(top_items),
        ad_text=config["ad_text"],
        ad_link=config["ad_link"],
    )
