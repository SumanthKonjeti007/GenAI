# from __future__ import annotations
# from typing import Literal, TypedDict
# import asyncio
# import os

# import streamlit as st
# import json
# import logfire
# from supabase import Client
# from openai import AsyncOpenAI

# # Import all the message part classes
# from pydantic_ai.messages import (
#     ModelMessage,
#     ModelRequest,
#     ModelResponse,
#     SystemPromptPart,
#     UserPromptPart,
#     TextPart,
#     ToolCallPart,
#     ToolReturnPart,
#     RetryPromptPart,
#     ModelMessagesTypeAdapter
# )
# from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps

# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# supabase: Client = Client(
#     os.getenv("SUPABASE_URL"),
#     os.getenv("SUPABASE_SERVICE_KEY")
# )

# # Configure logfire to suppress warnings (optional)
# logfire.configure(send_to_logfire='never')

# class ChatMessage(TypedDict):
#     """Format of messages sent to the browser/API."""

#     role: Literal['user', 'model']
#     timestamp: str
#     content: str


# def display_message_part(part):
#     """
#     Display a single part of a message in the Streamlit UI.
#     Customize how you display system prompts, user prompts,
#     tool calls, tool returns, etc.
#     """
#     # system-prompt
#     if part.part_kind == 'system-prompt':
#         with st.chat_message("system"):
#             st.markdown(f"**System**: {part.content}")
#     # user-prompt
#     elif part.part_kind == 'user-prompt':
#         with st.chat_message("user"):
#             st.markdown(part.content)
#     # text
#     elif part.part_kind == 'text':
#         with st.chat_message("assistant"):
#             st.markdown(part.content)          


# async def run_agent_with_streaming(user_input: str):
#     """
#     Run the agent with streaming text for the user_input prompt,
#     while maintaining the entire conversation in `st.session_state.messages`.
#     """
#     # Prepare dependencies
#     deps = PydanticAIDeps(
#         supabase=supabase,
#         openai_client=openai_client
#     )

#     # Run the agent in a stream
#     async with pydantic_ai_expert.run_stream(
#         user_input,
#         deps=deps,
#         message_history= st.session_state.messages[:-1],  # pass entire conversation so far
#     ) as result:
#         # We'll gather partial text to show incrementally
#         partial_text = ""
#         message_placeholder = st.empty()

#         # Render partial text as it arrives
#         async for chunk in result.stream_text(delta=True):
#             partial_text += chunk
#             message_placeholder.markdown(partial_text)

#         # Now that the stream is finished, we have a final result.
#         # Add new messages from this run, excluding user-prompt messages
#         filtered_messages = [msg for msg in result.new_messages() 
#                             if not (hasattr(msg, 'parts') and 
#                                     any(part.part_kind == 'user-prompt' for part in msg.parts))]
#         st.session_state.messages.extend(filtered_messages)

#         # Add the final response to the messages
#         st.session_state.messages.append(
#             ModelResponse(parts=[TextPart(content=partial_text)])
#         )


# async def main():
#     st.title("Pydantic AI Agentic RAG")
#     st.write("Ask any question about Pydantic AI, the hidden truths of the beauty of this framework lie within.")

#     # Initialize chat history in session state if not present
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Display all messages from the conversation so far
#     # Each message is either a ModelRequest or ModelResponse.
#     # We iterate over their parts to decide how to display them.
#     for msg in st.session_state.messages:
#         if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
#             for part in msg.parts:
#                 display_message_part(part)

#     # Chat input for the user
#     user_input = st.chat_input("What questions do you have about Pydantic AI?")

#     if user_input:
#         # We append a new request to the conversation explicitly
#         st.session_state.messages.append(
#             ModelRequest(parts=[UserPromptPart(content=user_input)])
#         )
        
#         # Display user prompt in the UI
#         with st.chat_message("user"):
#             st.markdown(user_input)

#         # Display the assistant's partial response while streaming
#         with st.chat_message("assistant"):
#             # Actually run the agent now, streaming the text
#             await run_agent_with_streaming(user_input)


# if __name__ == "__main__":
#     asyncio.run(main())

from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import Client
from pydantic_ai.messages import ModelResponse
# from openai import AsyncOpenAI  # ← removed

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter,
)
from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps  # deps only has `supabase`

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Supabase client (keep your existing pattern) ---
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire="never")

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal["user", "model"]
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    # if part.part_kind == "system-prompt":
    #     with st.chat_message("system"):
    #         st.markdown(f"**System**: {part.content}")
    # # user-prompt
    # elif part.part_kind == "user-prompt":
    #     with st.chat_message("user"):
    #         st.markdown(part.content)
    # # text
    # elif part.part_kind == "text":
    #     with st.chat_message("assistant"):
    #         st.markdown(part.content)
    if part.part_kind == "system-prompt":
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    elif part.part_kind == "user-prompt":
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == "text":
        with st.chat_message("assistant"):
            st.markdown(part.content)
    elif part.part_kind in ("tool-call", "tool-return"):
        # Hide in user-facing mode; comment this out to debug
        return


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies (only supabase now)
    deps = PydanticAIDeps(supabase=supabase)

    # Run the agent in a stream
    async with pydantic_ai_expert.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],  # pass entire conversation so far
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Add new messages from this run, excluding user-prompt messages
        # filtered_messages = [
        #     msg
        #     for msg in result.new_messages()
        #     if not (hasattr(msg, "parts") and any(part.part_kind == "user-prompt" for part in msg.parts))
        # ]
        filtered_messages = []
        for msg in result.new_messages():
            # Keep only model responses (not requests/tool-calls)
            if isinstance(msg, ModelResponse):
                text_parts = [p for p in msg.parts if p.part_kind == "text"]
                if text_parts:
                    filtered_messages.append(ModelResponse(parts=text_parts))
        
        st.session_state.messages.extend(filtered_messages)

        # Add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )


async def main():
    st.title("Pydantic AI Agentic RAG")
    st.write("Ask any question about Pydantic AI, the hidden truths of the beauty of this framework lie within.")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What questions do you have about Pydantic AI?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input)


if __name__ == "__main__":
    asyncio.run(main())
