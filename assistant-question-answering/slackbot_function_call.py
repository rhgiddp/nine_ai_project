import json
import os

import anthropic
from slack import WebClient
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from calendar_functions import create_event, delete_event, check_event


MESSAGES = []

# Event API & Web API
app = App(token=os.environ['SLACK_BOT_TOKEN']) 
slack_client = WebClient(os.environ['SLACK_BOT_TOKEN'])
anthropic_client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])


def process_tool_call(tool_name, tool_input):
    """툴 이름에 따라 실제 함수 호출을 라우팅."""
    if tool_name == "create_event":
        return create_event(**tool_input)
    elif tool_name == "delete_event":
        return delete_event(**tool_input)
    elif tool_name == "check_event":
        return check_event(**tool_input)

# This gets activated when the bot is tagged in a channel    
@app.event("app_mention")
def handle_message_events(body, logger):
    """
    멘션 수신 시:
      - 사용자 메시지를 프롬프트로 구성
      - 대기 메시지 전송
      - Anthropic에 툴 정의와 함께 요청
      - 필요 시 툴 실행 결과를 이어붙여 최종 응답 회신
    """
    # Log message
    print(str(body["event"]["text"]).split(">")[1])
    
    # Create prompt for ChatGPT
    prompt = str(body["event"]["text"]).split(">")[1]
    
    # Let the user know that we are busy with the request 
    response = slack_client.chat_postMessage(
        channel=body["event"]["channel"],
        thread_ts=body["event"]["event_ts"],
        text="안녕하세요, 개인 비서 슬랙봇입니다! :robot_face: \n곧 전달 주신 문의사항 처리하겠습니다!"
    )
    print(f"\n{'='*50}\nUser Message: {prompt}\n{'='*50}")
    
    # 2. tools 정의
    tools = [
        {
            "name": "create_event",
            "description": "Create new Google Calender Event",
            "input_schema": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Name of Google Calender Event"
                    },
                    "start": {
                        "type": "string",
                        "description": "Starting date of Google Calender Event in UTC+9 Time i.e. 2024-08-08T09:00:00+09:00"
                    },
                    "end": {
                        "type": "string",
                        "description": "Starting date of Google Calender Event in UTC+9 Time i.e. 2024-08-08T09:00:00+09:00"
                    }
                },
                "required": ["summary", "start"]
            }
        },
        {
            "name": "check_event",
            "description": "Check Google Calender Events",
            "input_schema": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "Starting date of Google Calender Event in UTC+9 Time i.e. 2024-08-08T09:00:00+09:00"
                    },
                    "end": {
                        "type": "string",
                        "description": "Starting date of Google Calender Event in UTC+9 Time i.e. 2024-08-08T09:00:00+09:00"
                    }
                },
                "required": ["start", "end"]
            }
        },
        {
            "name": "delete_event",
            "description": "Delete Google Calender Event. Delete immediately if you already have Calendar Event ID.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique ID of Calendar Event. Can be fetched using check_event()"
                    }
                },
                "required": ["id"]
            }
        }
    ]

    # 3. LLM API 호출
    MESSAGES.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    message = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        tools=tools,
        messages=MESSAGES,
    )
    print(f"\nInitial Response:")
    print(f"Stop Reason: {message.stop_reason}")
    print(f"Content: {message.content}")

    if message.stop_reason == "tool_use":
        tool_use = next(block for block in message.content if block.type == "tool_use")
        tool_name = tool_use.name
        tool_input = tool_use.input

        print(f"\nTool Used: {tool_name}")
        print(f"Tool Input: {tool_input}")

        tool_result = process_tool_call(tool_name, tool_input)

        print(f"Tool Result: {tool_result}")

        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            messages=MESSAGES+[
                {"role": "assistant", "content": message.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": json.dumps(tool_result),
                        }
                    ],
                },
            ],
            tools=tools,
        )

    else:
        response = message

    final_response = next(
        (block.text for block in response.content if hasattr(block, "text")),
        None,
    )
    print(response.content)
    print(f"\nFinal Response: {final_response}")

    tool_use = next(
        (block for block in response.content if getattr(block, 'type', None) == "tool_use"),
        None
    )
    if tool_use:
        tool_result = process_tool_call(tool_use.name, tool_use.input)
        print(tool_result)

    # Reply to thread 
    response = slack_client.chat_postMessage(channel=body["event"]["channel"], 
                                       thread_ts=body["event"]["event_ts"],
                                       text=f"{final_response}")
    MESSAGES.append({"role": "assistant", "content": final_response})

    return response


if __name__ == "__main__":
    SocketModeHandler(app, os.environ['SLACK_APP_TOKEN']).start()