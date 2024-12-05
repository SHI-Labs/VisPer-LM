import os
import base64
import json
import cv2
import time
import numpy as np
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential, get_bearer_token_provider

class GPTModel:
    def __init__(self, model_name="gpt-4o") -> None:
        self.client = self._get_client()
        self.model_name = model_name
    
    def _get_client(self, resource_name="dl-openai-1"):
        endpoint = f"https://{resource_name}.openai.azure.com/"
        api_version = "2024-02-15-preview"  # Replace with the appropriate API version
        
        # ChainedTokenCredential example borrowed from
        # https://github.com/technology-and-research/msr-azure/blob/main/knowledge-base/how-to/Access-Storage-Without-Keys-in-Azure-ML.md
        # Attribution: AI4Science
        azure_credential = ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential(
                exclude_cli_credential=True,
                # Exclude other credentials we are not interested in.
                exclude_environment_credential=True,
                exclude_shared_token_cache_credential=True,
                exclude_developer_cli_credential=True,
                exclude_powershell_credential=True,
                exclude_interactive_browser_credential=True,
                exclude_visual_studio_code_credentials=True,
                # DEFAULT_IDENTITY_CLIENT_ID is a variable exposed in
                # Azure ML Compute jobs that has the client id of the
                # user-assigned managed identity in it.
                # See https://learn.microsoft.com/en-us/azure/machine-learning/how-to-identity-based-service-authentication#compute-cluster
                # In case it is not set the ManagedIdentityCredential will
                # default to using the system-assigned managed identity, if any.
                managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
            )
        )
        
        # token_provider = get_bearer_token_provider(azure_credential,
        #     "https://cognitiveservices.azure.com/.default")
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_ad_token_provider=self.get_token
        )
        
        return client

    def get_token(self):
        token = os.environ["AZURE_TOKEN"]
        return token

    def __load_image(self, img_path):
        '''
        load png images
        '''
        img = cv2.imread(img_path)
        img_encoded_bytes = base64.b64encode(cv2.imencode('.jpg', img)[1])
        img_encoded_str = img_encoded_bytes.decode('utf-8')
        return img_encoded_str

    def get_response(self, image_path, messages, temperature=0.0, max_tokens=128):
        if image_path is not None:
            image_tag = self.__load_image(image_path)
        
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_tag
                        },
                        {
                            "type": "text",
                            "text": conversation_text[0]["content"]
                        }
                    ]
                },
            ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result_str = json.loads(completion.to_json())["choices"][0]["message"]["content"] # .strip('```python').strip('```').strip())
        except Exception as e:
            result_str = f"API Error: {str(e)}"
        
        # if error code is 429, it means the API call limit has been reached, sleep for 90s and retry
        while "Error code: 429" in result_str:
            time.sleep(90)
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                result_str = json.loads(completion.to_json())["choices"][0]["message"]["content"]
            except Exception as e:
                result_str = f"API Error: {str(e)}"

        return result_str

    def get_response_for_images(self, images, conversation_text):
        # encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
        base64Frames = []
        # sample at most 5 frames uniformly
        sample_idx = np.linspace(0, len(images)-1, num=5, dtype=int)
        images = [images[i] for i in sample_idx]
        for frame in images:
            # get frame size
            height, width, _ = frame.shape
            # resize frame's shortest side to 512 and keep original aspect ratio
            if height < width:
                frame = cv2.resize(frame, (int(width * 512 / height), 512))
            else:
                frame = cv2.resize(frame, (512, int(height * 512 / width)))
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        messages=[
            {
                "role": "system",
                "content": "You are teaching others to do some daily tasks through videos. With the task in mind, please describe the procedure to finish the task in the video."
            },
            {
                "role": "user", 
                "content": [
                    *map(lambda x:{"type":"image_url", "image_url":{"url":f'data:image/jpg;base64,{x}'}}, base64Frames), 
                    {
                        "type": "text",
                        "text": conversation_text
                    },                                                                      
                ],
            }            
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            result_str = json.loads(completion.to_json())["choices"][0]["message"]["content"] # .strip('```python').strip('```').strip())
        except Exception as e:
            result_str = f"API Error: {str(e)}"
        
        # if error code is 429, it means the API call limit has been reached, sleep for 90s and retry
        while "Error code: 429" in result_str:
            time.sleep(90)
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                result_str = json.loads(completion.to_json())["choices"][0]["message"]["content"]
            except Exception as e:
                result_str = f"API Error: {str(e)}"

        return result_str        

if __name__ == "__main__":
    model = GPTModel()
    image_path = "test.jpg"
    conversation_text = "You are teaching others to do some daily tasks through videos. With the task in mind, please describe the procedure to finish the task in the video."
    response = model.get_response(image_path, conversation_text)
    print(response)