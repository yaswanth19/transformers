from transformers import LlamaTokenizerFast
from transformers.models.janus.image_processing_janus import JanusImageProcessor
from transformers.models.janus.processing_janus import JanusProcessor


def main():
    conversation_1 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Draw animals fleeing from a storm in a futuristic setting"},
            ],
        },
    ]

    """
    In Janus code, this is equivalent to this user-side input:
    [
        {
            "role": "<|User|>",
            "content": "Draw animals fleeing from a storm in a futuristic setting",
        },
        {
            "role": "<|Assistant|>", 
            "content": ""
        },
    ]
    """

    conversation_2 = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What’s shown in this image?"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "This image shows a red stop sign."}, ]
        },
        {

            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image in more details."},
            ],
        },
    ]

    """
    In Janus code, this is equivalent to this user-side input:
    [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\nWhat’s shown in this image?",
            "images": ["./images/sign.png"],
        },
        {
            "role": "<|Assistant|>", 
            "content": "This image shows a red stop sign."
        },
        {
            "role": "<|User|>", 
            "content": "Describe the image in more details."
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    """

    chat_template = "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"


    chat_template_yxxx = "{% for message in messages %}{{ message['role'].capitalize() + ': '}}{% for content in message['content'] %}{% if content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}{% if not loop.last %}\n\n{% endif %}{% endfor %}"

    chat_template_mine = "{%set seps=['\n\n','<｜end▁of▁sentence｜>']%}{%set i=0%}{%for message in messages%}{%if message['role']!='system'%}{%if message['role']=='user'%}<|User|>: {%else%}<|Assistant|>: {%endif%}{%endif%}{%for content in message['content']%}{%if content['type']=='image'%}{%if not loop.first%}{{'\n'}}{%endif%}<image_placeholder>{%if not loop.last%}{{'\n'}}{%endif%}{%elif content['type']=='text'%}{%set text=content['text']%}{%if loop.first%}{%set text=text.lstrip()%}{%endif%}{%if loop.last%}{%set text=text.rstrip()%}{%endif%}{%if not loop.first and message['content'][loop.index0-1]['type']=='text'%}{{' '+text}}{%else%}{{text}}{%endif%}{%endif%}{%endfor%}{%if not loop.last or add_generation_prompt%}{%if message['role']=='system'%}{{seps[0]}}{%elif message['role']=='user'%}{{seps[0]}}{%else%}{{seps[1]}}{%endif%}{%endif%}{%endfor%}{%if add_generation_prompt%}<|Assistant|>:{%endif%}"


    tokenizer = LlamaTokenizerFast.from_pretrained('deepseek-ai/Janus-Pro-7B')

    # Image processor does not matter for this example
    processor = JanusProcessor(JanusImageProcessor(), tokenizer)

    for cv in [conversation_1, conversation_2]:
        print(processor.apply_chat_template(cv, chat_template=chat_template, add_generation_prompt=True))
        # Print something to separate
        print("-" * 80)
        print(processor.apply_chat_template(cv, chat_template=chat_template_yxxx, add_generation_prompt=True))
        print("-" * 80)
        print(processor.apply_chat_template(cv, chat_template=chat_template_mine, add_generation_prompt=True))
        print("-" * 80)
    exit()

    prompt = processor(conversation_1, return_tensors="pt", return_for_image_generation=True)

    print(prompt)
    pass


if __name__ == '__main__':
    main()
