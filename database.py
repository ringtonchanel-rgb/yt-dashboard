import os
import psycopg2
from datetime import datetime, timedelta
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional
import time

class Database:
    def __init__(self):
        self.connect()
        self.create_tables()

    def connect(self):
        """Устанавливает соединение с базой данных с автоматическим переподключением"""
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                if hasattr(self, 'conn') and self.conn:
                    try:
                        # Проверяем, живо ли соединение
                        with self.conn.cursor() as cur:
                            cur.execute("SELECT 1")
                        return  # Соединение работает
                    except psycopg2.Error:
                        # Соединение мертво, закрываем его
                        try:
                            self.conn.close()
                        except:
                            pass

                # Создаем новое соединение
                self.conn = psycopg2.connect(
                    os.environ['DATABASE_URL'],
                    application_name='YouTube Analytics'
                )
                self.conn.autocommit = False  # Явное управление транзакциями
                print("Database connection established successfully")
                return
            except psycopg2.Error as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise Exception(f"Failed to connect to database after {max_retries} attempts: {str(e)}")
                print(f"Connection attempt {retry_count} failed, retrying in 1 second...")
                time.sleep(1)

    def ensure_connection(self):
        """Проверяет и восстанавливает соединение при необходимости"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
        except (psycopg2.Error, AttributeError):
            self.connect()

    def create_tables(self):
        with self.conn.cursor() as cur:
            # Таблица для хранения каналов
            cur.execute("""
                CREATE TABLE IF NOT EXISTS channels (
                    channel_id TEXT PRIMARY KEY,
                    title TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Таблица для хранения просмотров видео
            cur.execute("""
                CREATE TABLE IF NOT EXISTS video_views (
                    video_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    views INTEGER NOT NULL,
                    PRIMARY KEY (video_id, timestamp)
                )
            """)

            # Таблица для хранения метаданных видео
            cur.execute("""
                CREATE TABLE IF NOT EXISTS video_metadata (
                    video_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    tags TEXT[],
                    published_at TIMESTAMP NOT NULL,
                    channel_id TEXT REFERENCES channels(channel_id)
                )
            """)

            # Таблица для хранения истории изменений названий
            cur.execute("""
                CREATE TABLE IF NOT EXISTS video_title_history (
                    video_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    changed_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (video_id, changed_at)
                )
            """)

            # Таблица для хранения истории обложек
            cur.execute("""
                CREATE TABLE IF NOT EXISTS video_thumbnails (
                    video_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    changed_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (video_id, changed_at)
                )
            """)

            # Таблица для хранения уведомлений о резких изменениях
            cur.execute("""
                CREATE TABLE IF NOT EXISTS view_change_alerts (
                    id SERIAL PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    previous_views INTEGER NOT NULL,
                    current_views INTEGER NOT NULL,
                    percentage_change FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_read BOOLEAN DEFAULT FALSE
                )
            """)
            self.conn.commit()

    def save_video_metadata(self, video_id: str, title: str, description: str, tags: List[str], published_at: datetime, channel_id: str):
        """Сохраняет метаданные видео"""
        self.ensure_connection()
        with self.conn.cursor() as cur:
            # Сохраняем текущие метаданные
            cur.execute("""
                INSERT INTO video_metadata (video_id, title, description, tags, published_at, channel_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (video_id) 
                DO UPDATE SET title = EXCLUDED.title,
                            description = EXCLUDED.description,
                            tags = EXCLUDED.tags
            """, (video_id, title, description, tags, published_at, channel_id))

            # Проверяем, изменилось ли название
            cur.execute("""
                SELECT title FROM video_title_history 
                WHERE video_id = %s 
                ORDER BY changed_at DESC LIMIT 1
            """, (video_id,))
            last_title = cur.fetchone()

            if not last_title or last_title[0] != title:
                cur.execute("""
                    INSERT INTO video_title_history (video_id, title, changed_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                """, (video_id, title))

            self.conn.commit()

    def save_video_thumbnail(self, video_id: str, thumbnail_url: str):
        """Сохраняет новую обложку видео"""
        self.ensure_connection()
        with self.conn.cursor() as cur:
            # Проверяем, изменилась ли обложка
            cur.execute("""
                SELECT url FROM video_thumbnails 
                WHERE video_id = %s 
                ORDER BY changed_at DESC LIMIT 1
            """, (video_id,))
            last_thumbnail = cur.fetchone()

            if not last_thumbnail or last_thumbnail[0] != thumbnail_url:
                cur.execute("""
                    INSERT INTO video_thumbnails (video_id, url, changed_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                """, (video_id, thumbnail_url))
                self.conn.commit()

    def get_video_metadata(self, video_id: str) -> Optional[Dict]:
        """Получает все метаданные видео, включая историю изменений"""
        self.ensure_connection()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Получаем основные метаданные
            cur.execute("SELECT * FROM video_metadata WHERE video_id = %s", (video_id,))
            metadata = cur.fetchone()

            if metadata:
                # Получаем историю изменений названия
                cur.execute("""
                    SELECT title, changed_at 
                    FROM video_title_history 
                    WHERE video_id = %s 
                    ORDER BY changed_at DESC
                """, (video_id,))
                metadata['title_history'] = cur.fetchall()

                # Получаем историю обложек
                cur.execute("""
                    SELECT url, changed_at 
                    FROM video_thumbnails 
                    WHERE video_id = %s 
                    ORDER BY changed_at DESC
                """, (video_id,))
                metadata['thumbnail_history'] = cur.fetchall()

                return dict(metadata)
            return None

    def save_video_views(self, video_id: str, views: int):
        """Сохраняет количество просмотров для видео"""
        try:
            self.ensure_connection()
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO video_views (video_id, timestamp, views)
                    VALUES (%s, %s, %s)
                """, (video_id, datetime.now(), views))
                self.conn.commit()
                print(f"✅ Сохранено для видео {video_id}: {views:,} просмотров")
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Ошибка сохранения просмотров для {video_id}: {str(e)}")
            # Пробуем переподключиться и повторить операцию
            try:
                self.connect()
                with self.conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO video_views (video_id, timestamp, views)
                        VALUES (%s, %s, %s)
                    """, (video_id, datetime.now(), views))
                    self.conn.commit()
                    print(f"✅ Успешное повторное сохранение для видео {video_id}")
            except Exception as retry_error:
                self.conn.rollback()
                print(f"❌ Повторная попытка не удалась: {str(retry_error)}")
                raise

    def get_video_views_history(self, video_id: str, hours: int = 48) -> List[Dict]:
        """Получает историю просмотров видео за указанное количество часов"""
        self.ensure_connection()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                # First verify the video exists in metadata and get published_at
                cur.execute("""
                    SELECT video_id, title, published_at 
                    FROM video_metadata 
                    WHERE video_id = %s
                """, (video_id,))
                video = cur.fetchone()
                if not video:
                    print(f"Video {video_id} not found in metadata")
                    return []

                # Get view history
                cur.execute("""
                    SELECT vv.timestamp, vv.views
                    FROM video_views vv
                    WHERE vv.video_id = %s
                    AND vv.timestamp >= %s
                    ORDER BY vv.timestamp ASC
                """, (video_id, datetime.now() - timedelta(hours=hours)))

                results = cur.fetchall()

                if not results:
                    # If no recent history, try to get at least the latest view count
                    cur.execute("""
                        SELECT vv.timestamp, vv.views
                        FROM video_views vv
                        WHERE vv.video_id = %s
                        ORDER BY vv.timestamp DESC
                        LIMIT 1
                    """, (video_id,))
                    latest = cur.fetchall()
                    if latest:
                        print(f"Found only latest view count for video {video_id}")
                        # Add initial zero point
                        return [
                            {'timestamp': video['published_at'], 'views': 0},
                            *latest
                        ]
                    else:
                        print(f"No view history found for video {video_id}")
                        return []

                # Add initial zero point at published_at time
                results = [{'timestamp': video['published_at'], 'views': 0}, *results]
                print(f"Found {len(results)} view records for video {video_id}")
                return results

            except Exception as e:
                print(f"Error getting view history for video {video_id}: {str(e)}")
                self.conn.rollback()
                return []

    def add_channel(self, channel_id: str, title: str):
        """Добавляет канал в список отслеживаемых"""
        self.ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO channels (channel_id, title)
                VALUES (%s, %s)
                ON CONFLICT (channel_id) DO NOTHING
            """, (channel_id, title))
            self.conn.commit()

    def remove_channel(self, channel_id: str):
        """Удаляет канал из списка отслеживаемых"""
        self.ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM channels WHERE channel_id = %s", (channel_id,))
            self.conn.commit()

    def get_channels(self) -> List[Dict]:
        """Получает список всех отслеживаемых каналов"""
        self.ensure_connection()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM channels ORDER BY added_at DESC")
            return [dict(row) for row in cur.fetchall()]

    def check_view_changes(self, video_id: str) -> Optional[Dict]:
        """Проверяет резкие изменения в просмотрах видео"""
        self.ensure_connection()
        with self.conn.cursor() as cur:
            # Получаем историю просмотров за последние 24 часа
            cur.execute("""
                SELECT timestamp, views
                FROM video_views
                WHERE video_id = %s
                AND timestamp >= NOW() - INTERVAL '24 hours'
                ORDER BY timestamp ASC
            """, (video_id,))

            views_data = cur.fetchall()
            if len(views_data) < 2:
                return None

            # Анализируем изменение просмотров
            oldest_views = views_data[0][1]
            latest_views = views_data[-1][1]

            if oldest_views == 0:
                return None

            percentage_change = ((latest_views - oldest_views) / oldest_views) * 100

            # Если изменение больше 50%, создаем уведомление
            if abs(percentage_change) >= 50:
                cur.execute("""
                    INSERT INTO view_change_alerts 
                    (video_id, alert_type, previous_views, current_views, percentage_change)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    video_id,
                    'increase' if percentage_change > 0 else 'decrease',
                    oldest_views,
                    latest_views,
                    percentage_change
                ))
                self.conn.commit()

                return {
                    'type': 'increase' if percentage_change > 0 else 'decrease',
                    'previous_views': oldest_views,
                    'current_views': latest_views,
                    'percentage_change': percentage_change
                }

            return None

    def get_unread_alerts(self) -> List[Dict]:
        """Получает список непрочитанных уведомлений"""
        self.ensure_connection()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute("""
                    SELECT a.*, vm.title
                    FROM view_change_alerts a
                    JOIN video_metadata vm ON a.video_id = vm.video_id
                    WHERE NOT a.is_read
                    ORDER BY a.created_at DESC
                """)
                return [dict(row) for row in cur.fetchall()]
            except Exception as e:
                self.conn.rollback()
                print(f"Error in get_unread_alerts: {e}")
                return []

    def mark_alert_as_read(self, alert_id: int):
        """Отмечает уведомление как прочитанное"""
        self.ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE view_change_alerts
                SET is_read = TRUE
                WHERE id = %s
            """, (alert_id,))
            self.conn.commit()

    def analyze_optimal_publishing_time(self) -> Dict:
        """Анализирует оптимальное время публикации на основе просмотров"""
        self.ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("""
                WITH video_performance AS (
                    SELECT 
                        vm.video_id,
                        vm.published_at,
                        EXTRACT(HOUR FROM vm.published_at) as hour,
                        vv.views,
                        EXTRACT(EPOCH FROM vv.timestamp - vm.published_at) / 3600 as hours_since_published
                    FROM video_metadata vm
                    JOIN video_views vv ON vm.video_id = vv.video_id
                    WHERE vv.timestamp <= vm.published_at + INTERVAL '24 hours'
                ),
                hourly_stats AS (
                    SELECT 
                        hour,
                        COUNT(*) as videos_published,
                        AVG(views) as avg_views_24h,
                        AVG(views / NULLIF(hours_since_published, 0)) as avg_views_per_hour
                    FROM video_performance
                    GROUP BY hour
                    ORDER BY avg_views_24h DESC
                )
                SELECT 
                    hour,
                    videos_published,
                    ROUND(avg_views_24h) as avg_views_24h,
                    ROUND(avg_views_per_hour) as avg_views_per_hour
                FROM hourly_stats
            """)

            results = cur.fetchall()
            if not results:
                return {
                    'best_hours': [],
                    'hourly_stats': []
                }

            # Преобразуем результаты в удобный формат
            hourly_stats = [{
                'hour': row[0],
                'videos_published': row[1],
                'avg_views_24h': row[2],
                'avg_views_per_hour': row[3]
            } for row in results]

            # Определяем лучшие часы для публикации
            sorted_hours = sorted(hourly_stats, key=lambda x: x['avg_views_24h'], reverse=True)
            best_hours = [h['hour'] for h in sorted_hours[:3]]  # Top 3 часа

            return {
                'best_hours': best_hours,
                'hourly_stats': hourly_stats
            }

    def get_channel_growth_stats(self, channel_id: str) -> List[Dict]:
        """Получает статистику роста канала по дням"""
        self.ensure_connection()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                WITH daily_stats AS (
                    SELECT 
                        DATE_TRUNC('day', vv.timestamp) as date,
                        SUM(vv.views) as total_views,
                        COUNT(DISTINCT vm.video_id) as total_videos
                    FROM video_views vv
                    JOIN video_metadata vm ON vv.video_id = vm.video_id
                    WHERE vm.channel_id = %s
                    GROUP BY DATE_TRUNC('day', vv.timestamp)
                    ORDER BY date
                )
                SELECT 
                    date,
                    total_views,
                    total_videos,
                    LAG(total_views) OVER (ORDER BY date) as prev_day_views,
                    total_views - LAG(total_views) OVER (ORDER BY date) as views_growth
                FROM daily_stats
            """, (channel_id,))
            return cur.fetchall()

    def compare_videos(self, video_ids: List[str]) -> List[Dict]:
        """Сравнивает несколько видео по различным метрикам"""
        self.ensure_connection()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            placeholders = ','.join(['%s'] * len(video_ids))
            cur.execute(f"""
                WITH video_metrics AS (
                    SELECT 
                        vm.video_id,
                        vm.title,
                        vm.published_at,
                        vm.tags,
                        MAX(vv.views) as max_views,
                        MIN(vv.views) as min_views,
                        MAX(vv.views) - MIN(vv.views) as views_growth,
                        EXTRACT(EPOCH FROM MAX(vv.timestamp) - MIN(vv.timestamp))/3600 as hours_tracked,
                        (SELECT COUNT(*) FROM video_comments vc WHERE vc.video_id = vm.video_id) as comments_count,
                        (SELECT AVG(likes) FROM video_comments vc WHERE vc.video_id = vm.video_id) as avg_comment_likes
                    FROM video_metadata vm
                    JOIN video_views vv ON vm.video_id = vv.video_id
                    WHERE vm.video_id IN ({placeholders})
                    GROUP BY vm.video_id, vm.title, vm.published_at, vm.tags
                )
                SELECT 
                    video_id,
                    title,
                    published_at,
                    array_length(tags, 1) as tags_count,
                    max_views,
                    views_growth,
                    ROUND(views_growth / NULLIF(hours_tracked, 0), 2) as growth_per_hour,
                    comments_count,
                    ROUND(avg_comment_likes, 2) as avg_comment_likes,
                    ROUND(comments_count::float / NULLIF(max_views, 0) * 100, 2) as engagement_rate
                FROM video_metrics
                ORDER BY growth_per_hour DESC
            """, video_ids)
            return cur.fetchall()

    def get_title_performance(self) -> List[Dict]:
        """Анализирует эффективность ключевых слов в названиях"""
        self.ensure_connection()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute("""
                    WITH words AS (
                        SELECT 
                            vm.video_id,
                            regexp_split_to_table(lower(vm.title), '\s+') as word,
                            vv.views
                        FROM video_metadata vm
                        JOIN video_views vv ON vm.video_id = vv.video_id
                        WHERE length(regexp_split_to_table(lower(vm.title), '\s+')) > 3
                    )
                    SELECT 
                        word,
                        COUNT(*) as usage_count,
                        ROUND(AVG(views)) as avg_views,
                        ROUND(MIN(views)) as min_views,
                        ROUND(MAX(views)) as max_views
                    FROM words
                    GROUP BY word
                    HAVING COUNT(*) > 1
                    ORDER BY avg_views DESC
                    LIMIT 20
                """)
                return [dict(row) for row in cur.fetchall()]
            except Exception as e:
                self.conn.rollback()
                print(f"Error in get_title_performance: {e}")
                return []

    def compare_channels(self, channel_ids: List[str]) -> List[Dict]:
        """Сравнивает статистику нескольких каналов с 1 февраля"""
        self.ensure_connection()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            placeholders = ','.join(['%s'] * len(channel_ids))
            cur.execute(f"""
                WITH video_latest_views AS (
                    SELECT 
                        video_id,
                        views,
                        ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY timestamp DESC) as rn
                    FROM video_views
                ),
                channel_stats AS (
                    SELECT 
                        c.channel_id,
                        c.title as channel_title,
                        COUNT(DISTINCT CASE 
                            WHEN vm.published_at >= '2025-02-01 00:00:00'::timestamp
                            THEN vm.video_id 
                            END
                        ) as new_videos_30d,
                        COALESCE(SUM(CASE 
                            WHEN vm.published_at >= '2025-02-01 00:00:00'::timestamp
                            THEN vlv.views 
                            END
                        ), 0) as new_videos_views_30d,
                        COALESCE(SUM(vlv.views), 0) as total_views_30d
                    FROM channels c
                    LEFT JOIN video_metadata vm ON c.channel_id = vm.channel_id
                    LEFT JOIN video_latest_views vlv ON vm.video_id = vlv.video_id AND vlv.rn = 1
                    WHERE c.channel_id IN ({placeholders})
                    GROUP BY c.channel_id, c.title
                )
                SELECT 
                    channel_id,
                    channel_title,
                    new_videos_30d as "Новых видео с 1 февраля",
                    new_videos_views_30d as "Просмотры новых видео",
                    total_views_30d as "Всего просмотров"
                FROM channel_stats
                ORDER BY total_views_30d DESC
            """, channel_ids)
            return cur.fetchall()

    def get_channels_growth_comparison(self, channel_ids: List[str]) -> List[Dict]:
        """Получает сравнительную статистику роста каналов по дням"""
        self.ensure_connection()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            placeholders = ','.join(['%s'] * len(channel_ids))
            cur.execute(f"""
                WITH daily_channel_stats AS (
                    SELECT 
                        c.channel_id,
                        c.title as channel_title,
                        DATE_TRUNC('day', vv.timestamp) as date,
                        SUM(vv.views) as daily_views,
                        COUNT(DISTINCT vm.video_id) as videos_published
                    FROM channels c
                    JOIN video_metadata vm ON vm.channel_id = c.channel_id
                    JOIN video_views vv ON vv.video_id = vm.video_id
                    WHERE c.channel_id IN ({placeholders})
                    GROUP BY c.channel_id, c.title, DATE_TRUNC('day', vv.timestamp)
                    ORDER BY date
                )
                SELECT 
                    *,
                    LAG(daily_views) OVER (PARTITION BY channel_id ORDER BY date) as prev_day_views,
                    daily_views - LAG(daily_views) OVER (PARTITION BY channel_id ORDER BY date) as views_growth
                FROM daily_channel_stats
            """, channel_ids)
            return cur.fetchall()

    def get_channels_publishing_patterns(self, channel_ids: List[str]) -> List[Dict]:
        """Анализирует паттерны публикации видео для нескольких каналов"""
        self.ensure_connection()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            placeholders = ','.join(['%s'] * len(channel_ids))
            cur.execute(f"""
                WITH publishing_stats AS (
                    SELECT 
                        c.channel_id,
                        c.title as channel_title,
                        EXTRACT(HOUR FROM vm.published_at) as hour,
                        EXTRACT(DOW FROM vm.published_at) as day_of_week,
                        COUNT(*) as videos_count,
                        ROUND(AVG(vv.views)) as avg_views
                    FROM channels c
                    JOIN video_metadata vm ON vm.channel_id = c.channel_id
                    JOIN video_views vv ON vv.video_id = vm.video_id
                    WHERE c.channel_id IN ({placeholders})
                    GROUP BY c.channel_id, c.title, 
                            EXTRACT(HOUR FROM vm.published_at),
                            EXTRACT(DOW FROM vm.published_at)
                )
                SELECT 
                    channel_id,
                    channel_title,
                    hour,
                    day_of_week,
                    videos_count,
                    avg_views,
                    ROUND(100.0 * videos_count / SUM(videos_count) OVER (PARTITION BY channel_id)) as percentage
                FROM publishing_stats
                ORDER BY channel_id, day_of_week, hour
            """, channel_ids)
            return cur.fetchall()

    def save_comments(self, video_id: str, comments: List[Dict]):
        """Сохраняет комментарии к видео"""
        self.ensure_connection()
        with self.conn.cursor() as cur:
            # Создаем таблицу, если еще не существует
            cur.execute("""
                CREATE TABLE IF NOT EXISTS video_comments (
                    comment_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    author TEXT NOT NULL,
                    text TEXT NOT NULL,
                    likes INTEGER DEFAULT 0,
                    published_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (video_id) REFERENCES video_metadata(video_id)
                )
            """)

            # Сохраняем комментарии
            for comment in comments:
                cur.execute("""
                    INSERT INTO video_comments 
                    (comment_id, video_id, author, text, likes, published_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (comment_id) DO UPDATE SET
                        likes = EXCLUDED.likes
                """, (
                    comment['id'],
                    video_id,
                    comment['author'],
                    comment['text'],
                    comment.get('likes', 0),
                    comment['published_at']
                ))
            self.conn.commit()

    def analyze_comments(self, video_id: str) -> Dict:
        """Анализирует комментарии к видео"""
        self.ensure_connection()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                # Общая статистика
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_comments,
                        COUNT(DISTINCT author) as unique_authors,
                        AVG(likes) as avg_likes,
                        MAX(likes) as max_likes
                    FROM video_comments
                    WHERE video_id = %s
                """, (video_id,))
                stats = cur.fetchone() or {
                    'total_comments': 0,
                    'unique_authors': 0,
                    'avg_likes': 0,
                    'max_likes': 0
                }

                # Если нет комментариев, возвращаем пустые данные
                if stats['total_comments'] == 0:
                    return {
                        'stats': stats,
                        'top_authors': [],
                        'top_comments': []
                    }

                # Топ авторов по количеству комментариев
                cur.execute("""
                    SELECT 
                        author,
                        COUNT(*) as comments_count,
                        SUM(likes) as total_likes
                    FROM video_comments
                    WHERE video_id = %s
                    GROUP BY author
                    ORDER BY comments_count DESC
                    LIMIT 5
                """, (video_id,))
                top_authors = cur.fetchall()

                # Комментарии с наибольшим количеством лайков
                cur.execute("""
                    SELECT *
                    FROM video_comments
                    WHERE video_id = %s
                    ORDER BY likes DESC
                    LIMIT 5
                """, (video_id,))
                top_comments = cur.fetchall()

                self.conn.commit()
                return {
                    'stats': stats,
                    'top_authors': top_authors,
                    'top_comments': top_comments
                }
            except Exception as e:
                self.conn.rollback()
                print(f"Error in analyze_comments: {e}")
                return {
                    'stats': {
                        'total_comments': 0,
                        'unique_authors': 0,
                        'avg_likes': 0,
                        'max_likes': 0
                    },
                    'top_authors': [],
                    'top_comments': []
                }

    def get_channel_engagement(self, channel_id: str) -> Dict:
        """Анализирует взаимодействие с каналом"""
        self.ensure_connection()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                WITH video_stats AS (
                    SELECT 
                        vm.video_id,
                        vm.title,
                        vv.views,
                        (SELECT COUNT(*) FROM video_comments vc WHERE vc.video_id = vm.video_id) as comments_count,
                        (SELECT SUM(likes) FROM video_comments vc WHERE vc.video_id = vm.video_id) as comments_likes
                    FROM video_metadata vm
                    LEFT JOIN video_views vv ON vm.video_id = vv.video_id
                    WHERE vm.channel_id = %s
                    AND vv.timestamp = (
                        SELECT MAX(timestamp)
                        FROM video_views vv2
                        WHERE vv2.video_id = vm.video_id
                    )
                )
                SELECT 
                    AVG(CAST(comments_count AS FLOAT) / NULLIF(views, 0) * 100) as comment_rate,
                    AVG(comments_count) as avg_comments,
                    AVG(comments_likes) as avg_comment_likes,
                    MAX(comments_count) as max_comments,
                    MIN(comments_count) as min_comments
                FROM video_stats
            """, (channel_id,))

            engagement = cur.fetchone()

            # Получаем топ видео по вовлеченности
            cur.execute("""
                WITH video_stats AS (
                    SELECT 
                        vm.video_id,
                        vm.title,
                        vv.views,
                        (SELECT COUNT(*) FROM video_comments vc WHERE vc.video_id = vm.video_id) as comments_count,
                        (SELECT SUM(likes) FROM video_comments vc WHERE vc.video_id = vm.video_id) as comments_likes
                    FROM video_metadata vm
                    LEFT JOIN video_views vv ON vm.video_id = vv.video_id
                    WHERE vm.channel_id = %s
                    AND vv.timestamp = (
                        SELECT MAX(timestamp)
                        FROM video_views vv2
                        WHERE vv2.video_id = vm.video_id
                    )
                )
                SELECT 
                    video_id,
                    title,
                    views,
                    comments_count,
                    comments_likes,
                    ROUND(CAST(comments_count AS FLOAT) / NULLIF(views, 0) * 100, 2) as engagement_rate
                FROM video_stats
                ORDER BY engagement_rate DESC
                LIMIT 5
            """, (channel_id,))

            top_videos = cur.fetchall()

            return {
                'engagement': engagement,
                'top_videos': top_videos
            }

    def get_channel_export_data(self, channel_id: str) -> Dict:
        """Получает все данные канала для экспорта"""
        self.ensure_connection()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Базовая информация о канале
            cur.execute("""
                SELECT title, added_at
                FROM channels
                WHERE channel_id = %s
            """, (channel_id,))
            channel_info = cur.fetchone()

            # Информация о видео
            cur.execute("""
                WITH video_latest_views AS (
                    SELECT 
                        video_id,
                        views,
                        ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY timestamp DESC) as rn
                    FROM video_views
                )
                SELECT 
                    vm.video_id,
                    vm.title,
                    vm.published_at,
                    vm.description,
                    vm.tags,
                    vlv.views as current_views,
                    (SELECT COUNT(*) FROM video_comments vc WHERE vc.video_id = vm.video_id) as comments_count,
                    (SELECT AVG(likes) FROM video_comments vc WHERE vc.video_id = vm.video_id) as avg_comment_likes
                FROM video_metadata vm
                LEFT JOIN video_latest_views vlv ON vm.video_id = vlv.video_id AND vlv.rn = 1
                WHERE vm.channel_id = %s
                ORDER BY vm.published_at DESC
            """, (channel_id,))
            videos = cur.fetchall()

            # Добавляем историю просмотров для каждого видео
            for video in videos:
                cur.execute("""
                    SELECT timestamp, views
                    FROM video_views
                    WHERE video_id = %s
                    ORDER BY timestamp ASC
                """, (video['video_id'],))
                video['views_history'] = cur.fetchall()

                # Добавляем топ комментарии
                cur.execute("""
                    SELECT author, text, likes, published_at
                    FROM video_comments
                    WHERE video_id = %s
                    ORDER BY likes DESC
                    LIMIT 5
                """, (video['video_id'],))
                video['top_comments'] = cur.fetchall()

            # Добавляем статистику канала
            cur.execute("""
                WITH daily_stats AS (
                    SELECT 
                        DATE_TRUNC('day', vv.timestamp) as date,
                        COUNT(DISTINCT vm.video_id) as videos_published,
                        SUM(vv.views) as total_views,
                        AVG(vv.views) as avg_views
                    FROM video_metadata vm
                    JOIN video_views vv ON vm.video_id = vv.video_id
                    WHERE vm.channel_id = %s
                    GROUP BY DATE_TRUNC('day', vv.timestamp)
                    ORDER BY date DESC
                )
                SELECT 
                    date,
                    videos_published,
                    total_views,
                    avg_views
                FROM daily_stats
            """, (channel_id,))
            channel_stats = cur.fetchall()

            channel_info_dict = dict(channel_info) if channel_info else {}
            return {
                'channel_info': {
                    **channel_info_dict,
                    'channel_id': channel_id,
                    'export_date': datetime.now(),
                    'total_videos': len(videos)
                },
                'videos': [dict(video) for video in videos],
                'channel_stats': [dict(stat) for stat in channel_stats]
            }

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
            print("Database connection closed")

db = Database()