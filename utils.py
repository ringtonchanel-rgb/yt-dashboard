import os
import requests
import streamlit as st
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
from database import db

API_KEY = os.getenv("YOUTUBE_API_KEY", "")

def create_sparkline(data, width=150, height=50):
    """Creates an animated sparkline chart for quick trend visualization"""
    if not data or len(data) < 2:
        return None

    try:
        # Convert data to lists for plotting
        timestamps = [datetime.fromisoformat(str(point['timestamp'])) for point in data]
        views = [point['views'] for point in data]

        # Create figure
        fig = go.Figure()

        # Add trace
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=views,
                mode='lines',
                line=dict(
                    color='#FF4B4B',
                    width=2
                ),
                hoverinfo='y'
            )
        )

        # Update layout for minimal sparkline appearance
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            width=width,
            height=height,
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[min(views), max(views)]
            )
        )

        return fig
    except Exception as e:
        print(f"Error creating sparkline: {str(e)}")
        return None

class YouTubeAPI:
    API_KEY = API_KEY

    @staticmethod
    def _make_api_request(url, error_message):
        """Helper method to make API requests with error handling"""
        try:
            st.write(f"Making API request to: {url.split('?')[0]}")
            response = requests.get(url)
            st.write(f"Response status code: {response.status_code}")

            if not response.ok:
                st.error(f"❌ {error_message} (Status: {response.status_code})")
                st.code(response.text)
                return None

            data = response.json()
            st.write(f"Got response with items: {len(data.get('items', []))}")
            return data

        except Exception as e:
            st.error(f"❌ {error_message}: {str(e)}")
            return None

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_videos(channel_id):
        """Get videos from channel's uploads playlist"""
        try:
            # Get channel's uploads playlist
            url = f"https://www.googleapis.com/youtube/v3/channels?key={YouTubeAPI.API_KEY}&id={channel_id}&part=contentDetails"
            data = YouTubeAPI._make_api_request(url, "Ошибка при получении информации о канале")

            if not data or not data.get('items'):
                st.error(f"❌ Канал не найден: {channel_id}")
                return None

            uploads_playlist = data['items'][0]['contentDetails']['relatedPlaylists']['uploads']

            # Get videos from playlist
            url = f"https://www.googleapis.com/youtube/v3/playlistItems?key={YouTubeAPI.API_KEY}&playlistId={uploads_playlist}&part=snippet&maxResults=50"
            data = YouTubeAPI._make_api_request(url, "Ошибка при получении списка видео")

            if not data or not data.get('items'):
                st.warning("⚠️ В плейлисте нет видео")
                return None

            # Process videos
            videos = []
            video_ids = []

            for item in data['items']:
                try:
                    video_id = item['snippet']['resourceId']['videoId']
                    video_ids.append(video_id)
                    videos.append({
                        'video_id': video_id,
                        'title': item['snippet']['title'],
                        'published_at': datetime.strptime(
                            item['snippet']['publishedAt'],
                            '%Y-%m-%dT%H:%M:%SZ'
                        )
                    })
                except Exception as e:
                    st.error(f"❌ Ошибка при обработке видео: {str(e)}")
                    continue

            if not videos:
                return None

            # Get statistics for all videos in one batch
            stats_url = f"https://www.googleapis.com/youtube/v3/videos?key={YouTubeAPI.API_KEY}&id={','.join(video_ids)}&part=statistics"
            stats_data = YouTubeAPI._make_api_request(stats_url, "Ошибка при получении статистики видео")

            if not stats_data:
                return None

            # Update videos with view counts
            for item in stats_data['items']:
                video_id = item['id']
                views = int(item['statistics'].get('viewCount', 0))

                # Save current views to database
                db.save_video_views(video_id, views)

                # Find corresponding video in our list
                for video in videos:
                    if video['video_id'] == video_id:
                        # Get view history
                        history = db.get_video_views_history(video_id)

                        if history:
                            # Calculate views in last 48h
                            oldest_views = history[0]['views']
                            video['views_48h'] = views - oldest_views
                            # Calculate views in last hour
                            hour_ago = datetime.now() - timedelta(hours=1)
                            hour_views = [h['views'] for h in history if h['timestamp'] >= hour_ago]
                            video['views_60m'] = hour_views[-1] - hour_views[0] if len(hour_views) > 1 else 0
                        else:
                            # First record for this video
                            video['views_48h'] = views
                            video['views_60m'] = 0

            # Sort by 48h views
            videos.sort(key=lambda x: x.get('views_48h', 0), reverse=True)
            return videos[:10]  # Return top 10 videos

        except Exception as e:
            st.error(f"❌ Ошибка при получении видео: {str(e)}")
            return None

    @st.cache_data(ttl=60)  # Cache for 1 minute
    def get_realtime_stats(channel_id):
        """Get real-time statistics for videos"""
        videos = YouTubeAPI.get_videos(channel_id)
        if not videos:
            return None

        # Get history for all videos
        time_series = []
        now = datetime.now()
        start_time = now - timedelta(hours=48)

        # Create hourly intervals
        for hour in range(48):
            point_time = start_time + timedelta(hours=hour)
            total_views = 0

            for video in videos:
                # Get views at this point in time
                history = db.get_video_views_history(video['video_id'])
                if history:
                    # Find closest measurement
                    closest = min(history, 
                                key=lambda x: abs((datetime.fromisoformat(str(x['timestamp'])) - point_time).total_seconds()))
                    total_views += closest['views']

            time_series.append({
                'timestamp': point_time,
                'views': total_views
            })

        return {
            'time_series': time_series,
            'videos': videos
        }