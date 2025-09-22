import time
from youtube_api import YouTubeAPI
from database import db
from datetime import datetime, timedelta

def collect_data():
    """Собирает данные о видео каждые 15 минут"""
    api = YouTubeAPI()
    collection_interval = 900  # 15 минут

    while True:
        try:
            start_time = datetime.now()
            print(f"\n=== Начало сбора данных: {start_time.strftime('%d.%m.%Y %H:%M:%S')} ===")

            # Получаем список отслеживаемых каналов
            channels = db.get_channels()

            for channel in channels:
                try:
                    # Получаем последние видео канала
                    videos = api.get_channel_videos(channel['channel_id'])
                    if videos:
                        print(f"\nОбработка канала: {channel['title']}")
                        for video in videos:
                            # Сохраняем метаданные видео
                            db.save_video_metadata(
                                video['id'],
                                video['title'],
                                video['description'],
                                video.get('tags', []),
                                video['published_at'],
                                channel['channel_id']  # Добавляем channel_id
                            )

                            # Сохраняем текущую обложку
                            if 'thumbnails' in video:
                                for quality in ['maxres', 'standard', 'high', 'medium', 'default']:
                                    if quality in video['thumbnails']:
                                        db.save_video_thumbnail(
                                            video['id'],
                                            video['thumbnails'][quality]['url']
                                        )
                                        break

                            # Сохраняем статистику
                            try:
                                views = int(video['statistics']['viewCount'])
                                db.save_video_views(video['id'], views)
                                print(f"✅ Сохранено для видео {video['title']}: {views:,} просмотров")
                            except Exception as e:
                                print(f"❌ Ошибка сохранения статистики для {video['title']}: {str(e)}")
                                continue

                            # Получаем и сохраняем комментарии
                            try:
                                comments = api.get_video_comments(video['id'])
                                if comments:
                                    db.save_comments(video['id'], comments)
                                    print(f"💬 Сохранено {len(comments)} комментариев")
                            except Exception as e:
                                print(f"❌ Ошибка сохранения комментариев: {str(e)}")
                                continue

                            # Проверяем изменения в просмотрах
                            view_change = db.check_view_changes(video['id'])
                            if view_change:
                                print(f"\n🔔 Резкое изменение просмотров для видео {video['title']}")
                                print(f"Изменение: {view_change['percentage_change']:.1f}%")
                                print(f"С {view_change['previous_views']:,} до {view_change['current_views']:,}")
                except Exception as channel_error:
                    print(f"Ошибка при обработке канала {channel['title']}: {channel_error}")
                    continue  # Продолжаем со следующим каналом

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            print(f"\n=== Сбор данных завершен за {execution_time:.1f} секунд ===")

            # Ждем до следующего сбора данных
            wait_time = max(0, collection_interval - execution_time)
            next_update = datetime.now() + timedelta(seconds=wait_time)
            if wait_time > 0:
                print(f"Следующее обновление в {next_update.strftime('%H:%M:%S')} (через {wait_time/60:.1f} минут)")
                time.sleep(wait_time)

        except Exception as e:
            print(f"\nКритическая ошибка при сборе данных: {e}")
            time.sleep(60)  # При ошибке ждем минуту перед повторной попыткой

if __name__ == "__main__":
    print("\n=== Запуск сборщика данных ===")
    collect_data()