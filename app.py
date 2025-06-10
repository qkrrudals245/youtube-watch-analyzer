import io
import base64
import json
from datetime import datetime, timedelta, timezone
from dateutil import parser
from flask import Flask, request, render_template_string
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────── 한글 폰트 설정 시작 ────────────────
# (1) 한글 폰트 이름 지정
plt.rcParams['font.family'] = 'Malgun Gothic'      # Windows


# (2) 음수 기호가 깨지지 않도록
plt.rcParams['axes.unicode_minus'] = False
# ──────────────── 한글 폰트 설정 끝 ────────────────

app = Flask(__name__)


def analyze_watch_history_json(content_bytes):
    raw_data = json.loads(content_bytes.decode('utf-8'))
    records = []
    for item in raw_data:
        t_str = item.get('time')
        if not t_str:
            continue
        try:
            ts = parser.isoparse(t_str)
            ts = ts.astimezone(timezone(timedelta(hours=9)))
        except:
            continue
        subs = item.get('subtitles', [])
        channel = subs[0]['name'] if subs and 'name' in subs[0] else 'Unknown'
        title = item.get('title', 'No Title')
        title_url = item.get('titleUrl', '')
        is_short = '/shorts/' in title_url.lower()
        vid = title_url
        records.append({
            'timestamp': ts,
            'channel': channel,
            'video_title': title,
            'video_id': vid,
            'is_short': is_short
        })
    df = pd.DataFrame(records)
    df = df[df['channel'] != 'Unknown']
    if df.empty:
        raise ValueError("유효한 시청이력 데이터가 없습니다.")

    # 날짜 범위 설정
    latest_date = df['timestamp'].max()
    one_year_ago = latest_date - pd.Timedelta(days=365)
    one_month_ago = latest_date - pd.Timedelta(days=30)
    two_months_ago = one_month_ago - pd.Timedelta(days=30)

    # 최근 1년 데이터 필터링
    df_year = df[df['timestamp'] >= one_year_ago]

    # 최근 30일, 이전 30일 데이터 필터링
    df_recent_30 = df[(df['timestamp'] > one_month_ago) & (df['timestamp'] <= latest_date)]
    df_prev_30 = df[(df['timestamp'] > two_months_ago) & (df['timestamp'] <= one_month_ago)]

    # 1. 최근 1년 일반 영상 top10 채널별 시청횟수
    general_df = df_year[~df_year['is_short']]
    top_general = general_df.groupby('channel').size().sort_values(ascending=False).head(10).to_dict()

    # 그래프용 전체 top30 채널 (일반 영상 기준만)
    general_counts = general_df.groupby('channel').size()
    ch_counts_df = pd.DataFrame({'general': general_counts}).fillna(0)
    ch_counts_df['total'] = ch_counts_df['general']  # 쇼츠 제거
    top30_channels_df = ch_counts_df.sort_values('total', ascending=False).head(30)
    channel_view_counts = top30_channels_df[['general']].to_dict(orient='index')

    # 4~5. 최근 30일 대비 이전 30일 시청 비율 변화 top10 (기준: 50회 이상)
    # 4~5. 최근 30일 대비 이전 30일 증감률(top10, 기준: 이전·최근 모두 50회 이상)
    def calc_counts(df_period):
        return df_period.groupby('channel').size().reset_index(name='count')

    recent_cnt = calc_counts(df_recent_30)
    prev_cnt   = calc_counts(df_prev_30)

    ratio_df = pd.merge(recent_cnt, prev_cnt, on='channel', how='outer', suffixes=('_recent','_prev')).fillna(0)

    # 절대 증감률 (%) 계산
    # 주의: 이전 기간 count_prev 가 0인 채널은 제외하거나 처리 필요
    mask = (ratio_df['count_prev'] >= 50) & (ratio_df['count_recent'] >= 50)
    ratio_df = ratio_df[mask].copy()
    ratio_df['pct_change'] = (
        (ratio_df['count_recent'] - ratio_df['count_prev'])
        / ratio_df['count_prev']
    ) * 100

    # 상승률 TOP10, 하락률 TOP10
    inc_df = ratio_df.sort_values('pct_change', ascending=False).head(10)
    dec_df = ratio_df.sort_values('pct_change', ascending=True).head(10)

    channels_increased = inc_df[['channel','count_recent','count_prev','pct_change']].itertuples(index=False, name=None)
    channels_decreased = dec_df[['channel','count_recent','count_prev','pct_change']].itertuples(index=False, name=None)

    # 6. 시간대별 시청횟수 (최근 30일) 일반 영상만
    def hour_stats(df_period):
        df_filt = df_period[~df_period['is_short']]
        df_filt = df_filt.copy()
        df_filt['hour'] = df_filt['timestamp'].dt.hour
        stats = df_filt.groupby('hour').size().reindex(range(24), fill_value=0).to_dict()
        return stats

    time_stats_general = hour_stats(df_recent_30)

    # 8. 요일별 시청횟수 (최근 1년) 일반 영상만
    def weekday_stats(df_period):
        df_filt = df_period[~df_period['is_short']]
        df_filt = df_filt.copy()
        df_filt['weekday'] = df_filt['timestamp'].dt.dayofweek
        stats = df_filt.groupby('weekday').size().reindex(range(7), fill_value=0).to_dict()
        return stats

    time_stats_weekday = weekday_stats(df_year)

    return {
        'current_month': latest_date.strftime('%Y-%m-%d'),
        'top_general': top_general,
        'channels_increased': list(channels_increased),
        'channels_decreased': list(channels_decreased),
        'time_stats_general': time_stats_general,
        'time_stats_weekday': time_stats_weekday,
        'channel_view_counts': channel_view_counts
    }


def plot_bar(x, y, title, xlabel, ylabel, rotation=45, stacked=False, y2=None, labels=None, colors=None):
    plt.figure(figsize=(12,6))
    if stacked and y2 is not None:
        plt.bar(x, y, label=labels[0], color=colors[0])
        plt.bar(x, y2, bottom=y, label=labels[1], color=colors[1])
    else:
        plt.bar(x, y, color=colors[0] if colors else None)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    if stacked:
        plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_bytes = buf.read()
    return base64.b64encode(img_bytes).decode('utf-8')


def plot_time_stats(time_stats, title):
    x = list(range(24))
    y = [time_stats.get(h,0) for h in x]
    return plot_bar(x, y, title, 'Hour of Day', 'View Count', rotation=0)


def plot_weekday_stats(weekday_stats, title):
    days = ['월', '화', '수', '목', '금', '토', '일']
    y = [weekday_stats.get(i, 0) for i in range(7)]
    return plot_bar(days, y, title, '요일', '시청 횟수', rotation=0)


def plot_channel_view_counts(channel_view_counts):
    channels = list(channel_view_counts.keys())
    general = [channel_view_counts[ch]['general'] for ch in channels]
    return plot_bar(channels, general, "General Views", "Channels", "View Count",
                    rotation=90)


HTML_TEMPLATE = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>유튜브 시청기록 분석</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    h1 { color: #333; }
    table { border-collapse: collapse; margin-bottom:20px; }
    table, th, td { border: 1px solid #ccc; padding:8px; text-align:left; }
    th { background:#eee; }
    .processing { color: #d00; font-weight:bold; }
  </style>
  <script>
    function showProcessing() {
      document.getElementById('processing-msg').style.display = 'block';
    }
  </script>
</head>
<body>
  <h1>유튜브 시청기록 분석 웹앱</h1>
  <h2>1) Google Takeout에서 watch-history.json 내려받기</h2>
  <ol>
    <li>크롬 등 브라우저에서 <code>takeout.google.com</code>에 접속합니다.</li>
    <li>구글 계정으로 로그인 후, <strong>YouTube 및 YouTube 뮤직</strong> 항목만 선택합니다.</li>
    <li>포맷은 JSON 형식으로 체크되어 있는지 확인하고, 내보내기(export) 요청을 합니다.</li>
    <li>잠시 기다리면 ZIP 파일이 생성되고, 이를 다운로드합니다.</li>
    <li>다운로드 받은 ZIP을 압축 해제하면 그 안에 <code>watch-history.json</code> 파일이 있습니다.</li>
    <li>이 파일을 아래 업로드 폼에 선택하고 “시청기록 확인” 버튼을 누르세요.</li>
  </ol>
  <h2>2) watch-history.json 파일 업로드</h2>
  <form method="POST" enctype="multipart/form-data" onsubmit="showProcessing()">
    <input type="file" name="watch_history" accept=".json" required>
    <input type="submit" value="시청기록 확인">
  </form>
  <p id="processing-msg" class="processing" style="display:none;">
    🔄 시청기록을 분석중입니다… 잠시만 기다려 주세요.
  </p>

  {% if plot_channel %}
    <h2>▶ 최근 1년간 상위 30개 채널 (일반 영상 기준)</h2>
    <img src="data:image/png;base64,{{ plot_channel }}" alt="채널별 시청 횟수 그래프" style="width:100%; max-width:1200px;">
  {% endif %}

  {% if result %}
    <hr>
    <h2>▶ 분석 결과</h2>

    <h3>1) 최근 1년 일반 영상 상위 10개 채널</h3>
    <table>
      <tr><th>순위</th><th>채널명</th><th>시청 횟수</th></tr>
      {% for ch, cnt in result.top_general.items() %}
      <tr>
        <td>{{ loop.index }}</td>
        <td>{{ ch }}</td>
        <td>{{ cnt }}</td>
      </tr>
      {% endfor %}
    </table>

    <h3>4) 최근 30일 시청 비율 ↑ 상위 10개 채널</h3>
    <table>
      <tr><th>순위</th><th>채널명</th><th>최근 30일 횟수</th><th>이전 30일 횟수</th><th>비율 변화</th></tr>
      {% for ch, cur_cnt, prev_cnt, diff in result.channels_increased %}
      <tr>
        <td>{{ loop.index }}</td>
        <td>{{ ch }}</td>
        <td>{{ cur_cnt }}</td>
        <td>{{ prev_cnt }}</td>
        <td>{{ "%.4f"|format(diff) }}</td>
      </tr>
      {% endfor %}
    </table>

    <h3>5) 최근 30일 시청 비율 ↓ 상위 10개 채널</h3>
    <table>
      <tr><th>순위</th><th>채널명</th><th>최근 30일 횟수</th><th>이전 30일 횟수</th><th>비율 변화</th></tr>
      {% for ch, cur_cnt, prev_cnt, diff in result.channels_decreased %}
      <tr>
        <td>{{ loop.index }}</td>
        <td>{{ ch }}</td>
        <td>{{ cur_cnt }}</td>
        <td>{{ prev_cnt }}</td>
        <td>{{ "%.4f"|format(diff) }}</td>
      </tr>
      {% endfor %}
    </table>

    <h3>6) 최근 30일 일반 영상 시청시간 통계</h3>
    <img src="data:image/png;base64,{{ plot_general }}" alt="일반 영상 시청시간 통계">

    <h3>8) 최근 1년 일반 영상 요일별 시청시간 통계</h3>
    <img src="data:image/png;base64,{{ plot_weekday }}" alt="요일별 시청시간 통계">

  {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    plot_general = None
    plot_channel = None
    plot_weekday = None

    if request.method == 'POST':
        uploaded_file = request.files.get('watch_history', None)
        if uploaded_file and uploaded_file.filename.endswith('.json'):
            try:
                content = uploaded_file.read()
                result = analyze_watch_history_json(content)
                plot_general = plot_time_stats(result['time_stats_general'], f"일반 영상 시청시간 통계 ({result['current_month']})")
                plot_channel = plot_channel_view_counts(result['channel_view_counts'])
                plot_weekday = plot_weekday_stats(result['time_stats_weekday'], f"최근 1년 일반 영상 요일별 시청시간 통계 ({result['current_month']})")
            except Exception as e:
                result = {'error': str(e)}
        else:
            result = {'error': '유효한 JSON 파일을 업로드하세요.'}

    return render_template_string(
        HTML_TEMPLATE,
        result=result,
        plot_general=plot_general,
        plot_channel=plot_channel,
        plot_weekday=plot_weekday
    )




