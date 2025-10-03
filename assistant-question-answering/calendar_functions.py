import os

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ['https://www.googleapis.com/auth/calendar']

def intialize_service():
    """
    구글 캘린더 API 서비스 객체를 초기화합니다.
    - 기존 토큰(`./res/token.json`)이 있으면 재사용
    - 없으면 OAuth 플로우로 토큰 생성 후 저장
    반환: googleapiclient.discovery.Resource
    """
    if os.path.exists('./res/token.json'):
        creds = Credentials.from_authorized_user_file('./res/token.json', SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            './res/credentials.json', SCOPES
        )
        creds = flow.run_local_server(port=0)
        with open('./res/token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('calendar', 'v3', credentials=creds)
    return service


def create_event(summary, start, end):
    """요약/시작/종료를 받아 기본 캘린더에 이벤트를 생성하고 결과를 반환."""
    service = intialize_service()

    event = {
        'summary': summary,
        'start': {
            'dateTime': start,
            'timeZone': 'Asia/Seoul',
        },
        'end': {
            'dateTime': end,
            'timeZone': 'Asia/Seoul',
        }
    }
    event = service.events().insert(calendarId='primary', body=event).execute()
    print('Event created: %s' % (event.get('htmlLink')))
    return event


def check_event(start, end):
    """특정 기간[start, end]의 이벤트 목록을 조회하여 API 응답을 반환."""
    service = intialize_service()

    events_result = service.events().list(
        calendarId='primary',
        timeMin=start,
        timeMax=end,
        maxResults=5,
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    return events_result


def delete_event(id):
    """이벤트 ID로 기본 캘린더의 이벤트를 삭제하고 API 응답을 반환."""
    service = intialize_service()

    event = service.events().delete(calendarId='primary', eventId=id).execute()
    print('Event deleted')
    return event
