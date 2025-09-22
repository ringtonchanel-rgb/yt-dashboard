# Core imports
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import io
import zipfile

# Local imports
from utils import create_sparkline  # Add import for sparkline function
from youtube_api import YouTubeAPI
from database import db

# Page config with auto-refresh
st.set_page_config(
    page_title="YouTube Analytics",
    page_icon="📊",
    layout="wide"
)

# Initialize API
api = YouTubeAPI()

# Title and description
st.title("📊 YouTube Analytics Dashboard")
st.markdown("Анализ активности и статистики YouTube-каналов в реальном времени")

# Sidebar - Channel Management
with st.sidebar:
    st.header("📺 Управление каналами")

    # Add new channel
    new_channel = st.text_input(
        "ID или ссылка на канал",
        help="Введите ID канала YouTube или ссылку на канал (например, https://youtube.com/c/ChannelName)"
    )

    if st.button("➕ Добавить канал"):
        if new_channel:
            channel_info = api.get_channel_info(new_channel)
            if channel_info:
                db.add_channel(channel_info['id'], channel_info['title'])
                st.success(f"Канал '{channel_info['title']}' добавлен!")
                st.rerun()
            else:
                st.error("Не удалось найти канал. Проверьте правильность ID или ссылки.")

    # List of channels
    st.subheader("Отслеживаемые каналы")
    channels = db.get_channels()

    if channels:
        for channel in channels:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📺 {channel['title']}")
            with col2:
                if st.button("❌", key=f"delete_{channel['channel_id']}"):
                    db.remove_channel(channel['channel_id'])
                    st.rerun()
    else:
        st.info("Добавьте каналы для отслеживания")


# Create tabs for different analysis views
tab1, tab2 = st.tabs(["📊 Анализ канала", "🔄 Сравнение каналов"])

with tab1:
    # Main content
    if channels:
        # Channel selector
        selected_channel = st.selectbox(
            "Выберите канал для анализа",
            options=[c['channel_id'] for c in channels],
            format_func=lambda x: next(c['title'] for c in channels if c['channel_id'] == x)
        )

        if selected_channel:
            # Get videos
            videos = api.get_channel_videos(selected_channel)

            if videos:
                last_update = datetime.now()
                st.success(f"Последние {len(videos)} видео (обновлено: {last_update.strftime('%H:%M:%S')})")

                # Show video statistics
                for video in videos:
                    # Main content area
                    with st.container():
                        col1, col2 = st.columns([1.5, 1])  # Изменили соотношение с [2, 1] на [1.5, 1]

                        with col1:
                            # Show current thumbnail
                            if 'thumbnails' in video:
                                for quality in ['maxres', 'standard', 'high', 'medium', 'default']:
                                    if quality in video['thumbnails']:
                                        st.image(video['thumbnails'][quality]['url'], use_container_width=True)
                                        break

                        with col2:
                            # Get views history
                            views_history = db.get_video_views_history(video['id'])
                            if views_history:
                                try:
                                    # Create sparkline
                                    fig_sparkline = create_sparkline(views_history, width=200, height=60)
                                    if fig_sparkline:
                                        st.plotly_chart(fig_sparkline, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Не удалось создать график тренда: {str(e)}")

                                # Create full chart
                                df = pd.DataFrame(views_history)
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                                df['views'] = pd.to_numeric(df['views'])
                                df = df.sort_values('timestamp')

                                fig = px.line(
                                    df,
                                    x='timestamp',
                                    y='views',
                                    title="Динамика просмотров",
                                    labels={
                                        'timestamp': 'Время',
                                        'views': 'Просмотры'
                                    }
                                )

                                # Configure layout
                                fig.update_layout(
                                    showlegend=False,
                                    height=400,
                                    margin=dict(l=20, r=20, t=30, b=20),
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    xaxis=dict(
                                        showgrid=True,
                                        gridcolor='lightgray',
                                        title=None,
                                        tickformat='%d.%m %H:%M'
                                    ),
                                    yaxis=dict(
                                        showgrid=True,
                                        gridcolor='lightgray',
                                        title='Просмотры',
                                        rangemode='nonnegative',  # Ensure y-axis starts from 0
                                        range=[0, None]  # Force y-axis to start at 0
                                    )
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("История просмотров пока недоступна")

                    # Metrics section below main content
                    st.write("")  # Добавляем отступ
                    metrics_cols = st.columns([1, 1, 1])
                    with metrics_cols[0]:
                        st.metric("👁️ Просмотры", f"{video['statistics']['viewCount']:,}")
                    with metrics_cols[1]:
                        st.metric("👍 Лайки", f"{video['statistics']['likeCount']:,}")
                    with metrics_cols[2]:
                        st.metric("💬 Комментарии", f"{video['statistics']['commentCount']:,}")

                    # Detailed information in expander
                    with st.expander(f"📺 {video['title']}", expanded=False):
                        # Create tabs for different types of analysis
                        metadata_tab, comments_tab = st.tabs(["📝 Метаданные", "💬 Комментарии"])

                        with metadata_tab:
                            # Metadata section
                            st.subheader("Метаданные")
                            st.write(f"**Дата публикации:** {video['published_at'].strftime('%d.%m.%Y %H:%M')}")
                            st.write("**Описание:**")
                            st.text(video['description'][:500] + "..." if len(video['description']) > 500 else video['description'])

                            if video.get('tags'):
                                st.write("**Теги:**")
                                st.write(", ".join(video['tags']))

                            # Thumbnail history section
                            metadata = db.get_video_metadata(video['id'])
                            if metadata and metadata.get('thumbnail_history'):
                                st.subheader("История изменений обложки")
                                for thumbnail in metadata['thumbnail_history']:
                                    st.image(thumbnail['url'], use_container_width=True)
                                    st.caption(f"Изменено: {thumbnail['changed_at'].strftime('%d.%m.%Y %H:%M')}")

                            # Title history section
                            if metadata and metadata.get('title_history'):
                                st.subheader("История изменений названия")
                                for title_change in metadata['title_history']:
                                    st.write(f"**{title_change['changed_at'].strftime('%d.%m.%Y %H:%M')}:** {title_change['title']}")

                        with comments_tab:
                            # Comments analysis
                            comment_analysis = db.analyze_comments(video['id'])
                            if comment_analysis['stats']['total_comments'] > 0:
                                # General statistics
                                st.subheader("📊 Статистика комментариев")
                                stats_cols = st.columns(4)
                                with stats_cols[0]:
                                    st.metric("Всего комментариев", comment_analysis['stats']['total_comments'])
                                with stats_cols[1]:
                                    st.metric("Уникальных авторов", comment_analysis['stats']['unique_authors'])
                                with stats_cols[2]:
                                    st.metric("Среднее лайков", f"{comment_analysis['stats']['avg_likes']:.1f}")
                                with stats_cols[3]:
                                    st.metric("Максимум лайков", comment_analysis['stats']['max_likes'])

                                # Top authors
                                st.subheader("👥 Топ авторов")
                                authors_df = pd.DataFrame(comment_analysis['top_authors'])
                                st.dataframe(
                                    authors_df,
                                    column_config={
                                        "author": "Автор",
                                        "comments_count": "Количество комментариев",
                                        "total_likes": "Всего лайков"
                                    },
                                    hide_index=True
                                )

                                # Top comments
                                st.subheader("🔝 Популярные комментарии")
                                for comment in comment_analysis['top_comments']:
                                    with st.container():
                                        st.markdown(f"**{comment['author']}** • {comment['likes']} 👍")
                                        st.write(comment['text'])
                                        st.write(f"*{comment['published_at'].strftime('%d.%m.%Y %H:%M')}*")
                                        st.divider()
                            else:
                                st.info("У этого видео пока нет комментариев")

                    st.divider()

                # Analytics sections
                st.header("📊 Анализ времени публикации")
                publishing_analysis = db.analyze_optimal_publishing_time()
                if publishing_analysis['hourly_stats']:
                    best_hours = publishing_analysis['best_hours']
                    st.subheader("🎯 Рекомендуемое время публикации")
                    for hour in best_hours:
                        st.write(f"**{hour}:00** - хорошее время для публикации")

                    hourly_df = pd.DataFrame(publishing_analysis['hourly_stats'])

                    fig_views = px.bar(
                        hourly_df,
                        x='hour',
                        y='avg_views_24h',
                        title="Среднее количество просмотров по времени публикации",
                        labels={'hour': 'Час публикации', 'avg_views_24h': 'Среднее количество просмотров'}
                    )
                    fig_views.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
                    st.plotly_chart(fig_views, use_container_width=True)
                    fig_published = px.bar(
                        hourly_df,
                        x='hour',
                        y='videos_published',
                        title="Количество опубликованных видео по часам",
                        labels={'hour': 'Час публикации', 'videos_published': 'Количество видео'}
                    )
                    fig_published.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
                    st.plotly_chart(fig_published, use_container_width=True)
                else:
                    st.info("Недостаточно данных для анализа времени публикации")

                # Export section
                st.header("📥 Экспорт данных")
                export_container = st.container()
                with export_container:
                    st.write("Выберите формат для экспорта данных канала:")

                    col1, col2, col3 = st.columns(3)

                    # Prepare export data
                    export_data = db.get_channel_export_data(selected_channel)
                    channel_name = next(c['title'] for c in channels if c['channel_id'] == selected_channel)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                    filename_base = f"{channel_name}_{timestamp}"

                    with col1:
                        if st.button("📄 Экспорт в CSV"):
                            # Prepare CSV data
                            # Videos data
                            videos_df = pd.DataFrame(export_data['videos'])
                            videos_df['tags'] = videos_df['tags'].apply(lambda x: ', '.join(x) if x else '')

                            # Stats data
                            stats_df = pd.DataFrame(export_data['channel_stats'])

                            # Create a buffer for ZIP file
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                                # Add videos data
                                videos_csv = videos_df.to_csv(index=False)
                                zf.writestr(f'videos_{filename_base}.csv', videos_csv)

                                # Add stats data
                                stats_csv = stats_df.to_csv(index=False)
                                zf.writestr(f'stats_{filename_base}.csv', stats_csv)

                                # Add channel info
                                channel_info_df = pd.DataFrame([export_data['channel_info']])
                                channel_csv = channel_info_df.to_csv(index=False)
                                zf.writestr(f'channel_info_{filename_base}.csv', channel_csv)

                            zip_buffer.seek(0)
                            st.download_button(
                                label="⬇️ Скачать CSV",
                                data=zip_buffer,
                                file_name=f"{filename_base}_export.zip",
                                mime="application/zip"
                            )

                    with col2:
                        if st.button("📊 Экспорт в Excel"):
                            # Prepare Excel data
                            buffer = io.BytesIO()
                            excel_writer = pd.ExcelWriter(buffer, engine='openpyxl')
                            with excel_writer:
                                # Channel info sheet
                                pd.DataFrame([export_data['channel_info']]).to_excel(
                                    excel_writer, sheet_name='Channel Info', index=False
                                )

                                # Videos sheet
                                videos_df = pd.DataFrame(export_data['videos'])
                                videos_df['tags'] = videos_df['tags'].apply(lambda x: ', '.join(x) if x else '')
                                videos_df.to_excel(excel_writer, sheet_name='Videos', index=False)

                                # Channel stats sheet
                                stats_df = pd.DataFrame(export_data['channel_stats'])
                                stats_df.to_excel(excel_writer, sheet_name='Channel Stats', index=False)

                                # Add sheets for each video's detailed data
                                for video in export_data['videos']:
                                    # Views history
                                    if video['views_history']:
                                        views_df = pd.DataFrame(video['views_history'])
                                        sheet_name = f"Views_{video['video_id'][-6:]}"
                                        views_df.to_excel(writer, sheet_name=sheet_name, index=False)

                                    # Top comments
                                    if video['top_comments']:
                                        comments_df = pd.DataFrame(video['top_comments'])
                                        sheet_name = f"Comments_{video['video_id'][-6:]}"
                                        comments_df.to_excel(writer, sheet_name=sheet_name, index=False)

                            buffer.seek(0)
                            st.download_button(
                                label="⬇️ Скачать Excel",
                                data=buffer,
                                file_name=f"{filename_base}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                    with col3:
                        if st.button("🔄 Экспорт в JSON"):
                            # Prepare JSON data with proper formatting
                            formatted_data = {
                                'channel_info': export_data['channel_info'],
                                'export_info': {
                                    'timestamp': datetime.now().isoformat(),
                                    'format_version': '1.0'
                                },
                                'videos': [{
                                    **video,
                                    'tags': list(video['tags']) if video['tags'] else [],
                                    'views_history': [
                                        {
                                            'timestamp': hist['timestamp'].isoformat(),
                                            'views': hist['views']
                                        } for hist in video['views_history']
                                    ] if video['views_history'] else [],
                                    'top_comments': [
                                        {
                                            **comment,
                                            'published_at': comment['published_at'].isoformat()
                                        } for comment in video['top_comments']
                                    ] if video['top_comments'] else []
                                } for video in export_data['videos']],
                                'channel_stats': [{
                                    **stat,
                                    'date': stat['date'].isoformat()
                                } for stat in export_data['channel_stats']]
                            }

                            json_str = json.dumps(formatted_data, ensure_ascii=False, indent=2)
                            st.download_button(
                                label="⬇️ Скачать JSON",
                                data=json_str,
                                file_name=f"{filename_base}.json",
                                mime="application/json"
                            )

                st.header("📈 Сравнительный анализ")
                growth_stats = db.get_channel_growth_stats(selected_channel)
                if growth_stats:
                    growth_df = pd.DataFrame(growth_stats)

                    fig_total = px.line(
                        growth_df,
                        x='date',
                        y='total_views',
                        title="Общее количество просмотров по дням",
                        labels={'date': 'Дата', 'total_views': 'Общее количество просмотров'}
                    )
                    st.plotly_chart(fig_total, use_container_width=True)

                    fig_growth = px.bar(
                        growth_df,
                        x='date',
                        y='views_growth',
                        title="Ежедневный прирост просмотров",
                        labels={'date': 'Дата', 'views_growth': 'Прирост просмотров'}
                    )
                    st.plotly_chart(fig_growth, use_container_width=True)

                    # Video comparison
                    st.subheader("🆚 Сравнение видео")
                    video_options = {f"{v['title']} ({v['statistics']['viewCount']:,} просмотров)": v['id'] for v in videos}
                    selected_videos = st.multiselect(
                        "Выберите видео для сравнения",
                        options=list(video_options.keys()),
                        max_selections=5
                    )

                    if selected_videos:
                        video_ids = [video_options[title] for title in selected_videos]
                        comparison = db.compare_videos(video_ids)
                        if comparison:
                            comparison_df = pd.DataFrame(comparison)

                            # График роста просмотров
                            fig_growth = px.bar(
                                comparison_df,
                                x='title',
                                y='growth_per_hour',
                                title="Скорость роста просмотров",
                                labels={
                                    'title': 'Видео',
                                    'growth_per_hour': 'Прирост просмотров в час'
                                }
                            )
                            fig_growth.update_layout(
                                xaxis_tickangle=-45,
                                height=400,
                                margin=dict(l=20, r=20, t=40, b=100)
                            )
                            st.plotly_chart(fig_growth, use_container_width=True)

                            # График вовлеченности
                            fig_engagement = px.bar(
                                comparison_df,
                                x='title',
                                y='engagement_rate',
                                title="Уровень вовлеченности (комментарии/просмотры)",
                                labels={
                                    'title': 'Видео',
                                    'engagement_rate': 'Процент вовлеченности'
                                }
                            )
                            fig_engagement.update_layout(
                                xaxis_tickangle=-45,
                                height=400,
                                margin=dict(l=20, r=20, t=40, b=100)
                            )
                            st.plotly_chart(fig_engagement, use_container_width=True)

                            # Таблица с детальной статистикой
                            st.write("**Детальная статистика по видео:**")
                            st.dataframe(
                                comparison_df,
                                column_config={
                                    "title": "Название",
                                    "published_at": "Дата публикации",
                                    "max_views": "Максимум просмотров",
                                    "views_growth": "Прирост просмотров",
                                    "growth_per_hour": "Прирост в час",
                                    "comments_count": "Комментарии",
                                    "avg_comment_likes": "Ср. лайков на комментарий",
                                    "engagement_rate": "Вовлеченность, %",
                                    "tags_count": "Количество тегов"
                                },
                                hide_index=True
                            )

                st.header("📝 Анализ названий")
                title_stats = db.get_title_performance()
                if title_stats:
                    title_df = pd.DataFrame(title_stats)
                    fig_words = px.bar(
                        title_df.head(10),
                        x='word',
                        y='avg_views',
                        title="Топ-10 ключевых слов по просмотрам",
                        labels={'word': 'Слово', 'avg_views': 'Среднее количество просмотров'}
                    )
                    st.plotly_chart(fig_words, use_container_width=True)

                    st.write("**Детальная статистика по словам:**")
                    st.dataframe(
                        title_df,
                        column_config={
                            "word": "Слово",
                            "usage_count": "Использований",
                            "avg_views": "Среднее просмотров",
                            "min_views": "Минимум просмотров",
                            "max_views": "Максимум просмотров"
                        }
                    )
            else:
                st.error("Не удалось получить список видео")
        else:
            st.info("👈 Выберите канал для анализа")
    else:
        st.info("👈 Добавьте каналы для отслеживания в боковом меню")

with tab2:
    st.header("🔄 Сравнение каналов")

    # Channel selection for comparison
    available_channels = {c['title']: c['channel_id'] for c in channels}
    selected_channels = st.multiselect(
        "Выберите каналы для сравнения",
        options=list(available_channels.keys()),
        max_selections=5
    )

    if len(selected_channels) > 1:
        channel_ids = [available_channels[title] for title in selected_channels]

        # Basic comparison
        comparison = db.compare_channels(channel_ids)
        if comparison:
            comparison_df = pd.DataFrame(comparison)

            # Videos in 30 days
            fig_videos = px.bar(
                comparison_df,
                x='channel_title',
                y='Новых видео с 1 февраля',
                title="Количество новых видео с 1 февраля",
                labels={
                    'channel_title': 'Канал',
                    'Новых видео с 1 февраля': 'Количество видео'
                }
            )
            st.plotly_chart(fig_videos, use_container_width=True)

            # Views of new videos
            fig_new_views = px.bar(
                comparison_df,
                x='channel_title',
                y='Просмотры новых видео',
                title="Просмотры новых видео с 1 февраля",
                labels={
                    'channel_title': 'Канал',
                    'Просмотры новых видео': 'Количество просмотров'
                }
            )
            st.plotly_chart(fig_new_views, use_container_width=True)

            # Total views in 30 days
            fig_total = px.bar(
                comparison_df,
                x='channel_title',
                y='Всего просмотров',
                title="Общее количество просмотров",
                labels={
                    'channel_title': 'Канал',
                    'Всего просмотров': 'Количество просмотров'
                }
            )
            st.plotly_chart(fig_total, use_container_width=True)

            # Detailed statistics table
            st.write("**Детальная статистика каналов:**")
            st.dataframe(
                comparison_df.drop('channel_id', axis=1),
                hide_index=True
            )
        else:
            st.info("""
            Для сравнения каналов:
            1. Выберите 2 или более каналов из списка выше
            2. Сравните количество новых видео с 1 февраля
            3. Проанализируйте просмотры новых видео
            4. Оцените общую активность каналов
            """)