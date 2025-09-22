import os
import requests
from datetime import datetime
from typing import Dict, List, Optional
import re

class YouTubeAPI:
    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("YouTube API key not found")

    def _make_request(self, url: str) -> Optional[Dict]:
        """Helper method to make API requests with proper error handling"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API request error: {str(e)}")
            return None

    def extract_channel_id(self, channel_input: str) -> Optional[str]:
        """Extract channel ID from various YouTube URL formats or return the ID if directly provided"""
        print(f"Trying to extract channel ID from: {channel_input}")

        # Clean the input
        channel_input = channel_input.strip()

        # If input is already a channel ID (starts with UC and 24 chars long)
        if re.match(r'^UC[\w-]{22}$', channel_input):
            print("Input appears to be a channel ID")
            return channel_input

        # Try to extract from URL
        patterns = [
            (r'youtube\.com/channel/(UC[\w-]{22})', 'standard'),  # Standard channel URL
            (r'youtube\.com/c/([^/]+)', 'custom'),                # Custom URL
            (r'youtube\.com/@([^/]+)', 'handle'),                 # Handle URL
            (r'youtube\.com/user/([^/]+)', 'legacy'),             # Legacy username URL
        ]

        for pattern, url_type in patterns:
            match = re.search(pattern, channel_input)
            if match:
                print(f"Matched {url_type} URL pattern")
                if url_type == 'standard':
                    return match.group(1)
                else:
                    # For custom URL, handle, or legacy username, first search for the channel
                    username = match.group(1)
                    print(f"Looking up channel ID for username: {username}")

                    # Search for the channel
                    url = f"https://www.googleapis.com/youtube/v3/search?key={self.api_key}&q={username}&type=channel&part=id,snippet"
                    response = self._make_request(url)

                    if response and response.get('items'):
                        for item in response['items']:
                            channel_id = item['id']['channelId']
                            print(f"Found channel ID: {channel_id}")
                            return channel_id

                    print("No matching channel found")

        # If not found by URL patterns, try direct username search
        if channel_input.startswith('@'):
            username = channel_input[1:]  # Remove @ symbol
            print(f"Trying direct search for username: {username}")

            url = f"https://www.googleapis.com/youtube/v3/search?key={self.api_key}&q={username}&type=channel&part=id,snippet"
            response = self._make_request(url)

            if response and response.get('items'):
                for item in response['items']:
                    channel_id = item['id']['channelId']
                    print(f"Found channel ID through direct search: {channel_id}")
                    return channel_id

        print("No matching URL pattern found")
        return None

    def get_channel_info(self, channel_input: str) -> Optional[Dict]:
        """Get basic channel information from ID or URL"""
        channel_id = channel_input if re.match(r'^UC[\w-]{22}$', channel_input) else self.extract_channel_id(channel_input)
        if not channel_id:
            print(f"Could not extract channel ID from input: {channel_input}")
            return None

        url = f"https://www.googleapis.com/youtube/v3/channels?key={self.api_key}&id={channel_id}&part=snippet,brandingSettings"
        response = self._make_request(url)

        if not response or not response.get('items'):
            print(f"No channel found for ID: {channel_id}")
            return None

        channel = response['items'][0]
        return {
            'id': channel_id,
            'title': channel['snippet']['title'],
            'description': channel['snippet']['description'],
            'customUrl': channel['snippet'].get('customUrl', '')
        }

    def get_channel_videos(self, channel_id: str, max_results: int = 10) -> Optional[List[Dict]]:
        """Get list of latest videos from a channel"""
        url = f"https://www.googleapis.com/youtube/v3/channels?key={self.api_key}&id={channel_id}&part=contentDetails"
        response = self._make_request(url)

        if not response or not response.get('items'):
            return None

        uploads_playlist = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        url = f"https://www.googleapis.com/youtube/v3/playlistItems?key={self.api_key}&playlistId={uploads_playlist}&part=snippet&maxResults={max_results}"
        response = self._make_request(url)

        if not response or not response.get('items'):
            return None

        videos = []
        for item in response['items']:
            video_id = item['snippet']['resourceId']['videoId']
            # Получаем подробную информацию о видео
            video_info = self.get_video_details(video_id)
            if video_info:
                videos.append(video_info)

        return videos

    def get_video_details(self, video_id: str) -> Optional[Dict]:
        """Get detailed information about a video including tags and thumbnails"""
        url = f"https://www.googleapis.com/youtube/v3/videos?key={self.api_key}&id={video_id}&part=snippet,statistics"
        response = self._make_request(url)

        if not response or not response.get('items'):
            return None

        video = response['items'][0]
        snippet = video['snippet']
        stats = video['statistics']

        return {
            'id': video_id,
            'title': snippet['title'],
            'description': snippet['description'],
            'published_at': datetime.strptime(
                snippet['publishedAt'],
                '%Y-%m-%dT%H:%M:%SZ'
            ),
            'tags': snippet.get('tags', []),
            'thumbnails': snippet['thumbnails'],
            'statistics': {
                'viewCount': int(stats.get('viewCount', 0)),
                'likeCount': int(stats.get('likeCount', 0)),
                'commentCount': int(stats.get('commentCount', 0))
            }
        }

    def get_video_stats(self, video_id: str) -> Optional[Dict]:
        """Get current statistics for a video"""
        url = f"https://www.googleapis.com/youtube/v3/videos?key={self.api_key}&id={video_id}&part=statistics"
        response = self._make_request(url)

        if not response or not response.get('items'):
            return None

        stats = response['items'][0]['statistics']
        return {
            'viewCount': int(stats.get('viewCount', 0)),
            'likeCount': int(stats.get('likeCount', 0)),
            'commentCount': int(stats.get('commentCount', 0)),
            'impressionCount': int(stats.get('impressionCount', 0))  # YouTube API не предоставляет эти данные напрямую
        }

    def get_video_comments(self, video_id: str, max_results: int = 100) -> Optional[List[Dict]]:
        """Get comments for a video"""
        url = f"https://www.googleapis.com/youtube/v3/commentThreads?key={self.api_key}&videoId={video_id}&part=snippet&maxResults={max_results}"
        response = self._make_request(url)

        if not response or not response.get('items'):
            return None

        comments = []
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'id': item['id'],
                'author': comment['authorDisplayName'],
                'text': comment['textDisplay'],
                'likes': comment.get('likeCount', 0),
                'published_at': datetime.strptime(
                    comment['publishedAt'],
                    '%Y-%m-%dT%H:%M:%SZ'
                )
            })

        return comments