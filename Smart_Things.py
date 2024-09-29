# smartthings
import requests
from Data_Time import datetime, timedelta

# SmartThings API 토큰
API_KEY = 'eb1b5c53-9f68-4df8-b668-9e774bde63e0'

# 제어하려는 디바이스의 ID
DEVICE_ID1 = '981d4c95-ea89-4888-a07e-b9c7230af257'
DEVICE_ID2 = '22e93c50-759e-46ab-9854-d4efd381f22a'

# 명령 전송을 위한 API URL
url1 = f'https://api.smartthings.com/v1/devices/{DEVICE_ID1}/commands'
url2 = f'https://api.smartthings.com/v1/devices/{DEVICE_ID2}/commands'

# 현재 시간을 기준으로 ISO8601 형식의 시간을 반환하는 함수
def get_iso8601_time():
    current_time = datetime.utcnow()
    start_time = current_time
    capture_time = current_time + timedelta(minutes=5)  # 5분 후로 설정
    end_time = current_time + timedelta(minutes=10)     # 10분 후로 설정

    # ISO8601 형식으로 변환 (Z는 UTC를 나타냄)
    start_time_iso = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    capture_time_iso = capture_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_time_iso = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

    return start_time_iso, capture_time_iso, end_time_iso


# 전원을 켜거나 끄는 명령 정의
def send_command(command):
    print(command)
    
    device, command = command.split(' ')
    capability = "switch" if command in ["on", "off"] else "videoCamera"  # 전원 관련은 switch, 그 외는 videoCamera
    data = {
        "commands": [
            {
                "component": "main",
                "capability": capability,
                "command": command
            }
        ]
    }

    # 요청 헤더 (Accept 헤더에 정확한 값 설정)
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
        'Accept': 'application/vnd.smartthings+json;v=1'  # 벤더 형식 사용
    }

    # POST 요청을 보내 전원 제어 명령 실행
    if device == 'plug':
        response = requests.post(url1, json=data, headers=headers)
    elif device == 'camera':
        response = requests.post(url2, json=data, headers=headers)

    # 결과 출력
    if response.status_code == 200:
        print(f"Successfully sent '{command}' command to device {device}.")
    else:
        print(f"Failed to send command: {response.status_code} - {response.text}")

def send_camera_move_command(direction):
    # 음소거 관련 명령어로 이동 명령어를 임시로 매핑 (이동 명령어는 올바르게 확인되어야 함)
    direction_map = {
        'left': 'mute',
        'right': 'unmute'
        # 카메라 이동 관련 명령어는 따로 추가되어야 함
    }

    if direction in direction_map:
        command = direction_map[direction]
        
        data = {
            "commands": [
                {
                    "component": "main",
                    "capability": "videoCamera",  # 올바른 capability 사용
                    "command": command
                }
            ]
        }

        # 요청 헤더 (Accept 헤더에 정확한 값 설정)
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json',
            'Accept': 'application/vnd.smartthings+json;v=1'  # 벤더 형식 사용
        }

        response = requests.post(url2, json=data, headers=headers)

        # 결과 출력
        if response.status_code == 200:
            print(f"Successfully sent '{command}' command to device camera.")
        else:
            print(f"Failed to send command: {response.status_code} - {response.text}")


# # 카메라 음소거/음소거 해제 테스트
# send_command("camera mute")
# send_command("camera unmute")

# 이미지 캡처 명령을 보내는 함수
def send_image_capture_command():
    data = {
        "commands": [
            {
                "component": "main",
                "capability": "videoCamera",
                "command": "imageCapture"  # imageCapture 기능에서 이미지 캡처 명령은 'capture'
            }
        ]
    }

    # 요청 헤더 (API 키 설정)
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
        'Accept': 'application/vnd.smartthings+json;v=1'  # 벤더 형식 사용
    }

    # POST 요청을 보내 이미지 캡처 명령 실행
    response = requests.post(url2, json=data, headers=headers)

    # 결과 출력
    if response.status_code == 200:
        print(f"Successfully sent 'image capture' command to device camera.")
    else:
        print(f"Failed to send command: {response.status_code} - {response.text}")